from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


NRI_FEATURE_SCHEMA_VERSION = "nri_pair_keypoints_v1"
NRI_VALENCE_FEATURE_SCHEMA_VERSION = "nri_type_keypoints_v1"


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


def _frame_mask_or_ones(frame_mask: Optional[torch.Tensor], batch: int, frames: int, device: torch.device) -> torch.Tensor:
    if frame_mask is None:
        return torch.ones(batch, frames, dtype=torch.bool, device=device)
    return frame_mask.to(device=device, dtype=torch.bool)


def build_pair_mask(person_mask: torch.Tensor) -> torch.Tensor:
    if person_mask.ndim != 2:
        raise ValueError(f"person_mask must have shape [B,N], got {tuple(person_mask.shape)}")
    batch, persons = person_mask.shape
    device = person_mask.device
    pair_mask = person_mask[:, :, None] & person_mask[:, None, :]
    not_diag = ~torch.eye(persons, dtype=torch.bool, device=device)[None, :, :]
    return pair_mask & not_diag


def _upper_pair_mask(pair_mask: torch.Tensor) -> torch.Tensor:
    persons = pair_mask.shape[-1]
    upper = torch.triu(torch.ones(persons, persons, dtype=torch.bool, device=pair_mask.device), diagonal=1)
    return pair_mask & upper[None, :, :]


def valid_pairs_from_mask(pair_mask: torch.Tensor) -> list[list[tuple[int, int]]]:
    valid = _upper_pair_mask(pair_mask.to(dtype=torch.bool))
    out: list[list[tuple[int, int]]] = []
    for b in range(valid.shape[0]):
        rows = []
        for i, j in valid[b].nonzero(as_tuple=False).detach().cpu().tolist():
            rows.append((int(i), int(j)))
        out.append(rows)
    return out


def _normalize_pair(pair: tuple[int, int]) -> tuple[int, int]:
    i, j = int(pair[0]), int(pair[1])
    return (i, j) if i <= j else (j, i)


def noisy_or_aggregate(p_pair: torch.Tensor, pair_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    valid = _upper_pair_mask(pair_mask)
    p = torch.where(valid, p_pair, torch.zeros_like(p_pair)).clamp(0.0, 1.0 - eps)
    log_no_interaction = torch.log1p(-p).flatten(1).sum(dim=1)
    return (1.0 - torch.exp(log_no_interaction)).masked_fill(~valid.flatten(1).any(dim=1), 0.0)


def topk_noisy_or_aggregate(
    p_pair: torch.Tensor,
    pair_mask: torch.Tensor,
    top_k: int = 3,
    eps: float = 1e-6,
) -> torch.Tensor:
    valid = _upper_pair_mask(pair_mask)
    flat_p = p_pair.flatten(1)
    flat_valid = valid.flatten(1)
    if flat_p.shape[1] == 0:
        return torch.zeros(p_pair.shape[0], dtype=p_pair.dtype, device=p_pair.device)
    k = max(1, min(int(top_k), flat_p.shape[1]))
    values = flat_p.masked_fill(~flat_valid, -1.0).topk(k, dim=1).values
    values = torch.where(values >= 0.0, values.clamp(0.0, 1.0 - eps), torch.zeros_like(values))
    p_video = 1.0 - torch.prod(1.0 - values, dim=1)
    return p_video.masked_fill(~flat_valid.any(dim=1), 0.0)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int, eps: float = 1e-6) -> torch.Tensor:
    mask_f = mask.to(dtype=values.dtype)
    return (values * mask_f).sum(dim=dim) / mask_f.sum(dim=dim).clamp_min(eps)


def _masked_min(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    inf = torch.full_like(values, float("inf"))
    out = torch.where(mask, values, inf).min(dim=dim).values
    return torch.where(torch.isfinite(out), out, torch.zeros_like(out))


def _gather_time(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    gather_index = indices[:, None, :, :].expand(-1, 1, values.shape[2], values.shape[3])
    return values.gather(1, gather_index).squeeze(1)


class KeypointFeatureBuilder(nn.Module):
    def __init__(self, keypoint_dim: int, compute_motion: bool = True, eps: float = 1e-6):
        super().__init__()
        self.keypoint_dim = int(keypoint_dim)
        self.compute_motion = bool(compute_motion)
        self.eps = float(eps)
        self.out_dim = self.keypoint_dim + (4 if self.compute_motion else 0)

    def forward(self, x: torch.Tensor, frame_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"X must have shape [B,T,N,K,C], got {tuple(x.shape)}")
        batch, frames, persons, joints, channels = x.shape
        if channels != self.keypoint_dim:
            raise ValueError(f"Expected keypoint_dim={self.keypoint_dim}, got C={channels}")

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if channels < 2:
            raise ValueError(f"Expected keypoint_dim >= 2, got {channels}")
        frame_mask_t = _frame_mask_or_ones(frame_mask, batch, frames, x.device)

        xy = x[..., :2]
        if channels >= 3:
            conf = x[..., 2:3].clamp(0.0, 1.0)
        else:
            conf = torch.ones(batch, frames, persons, joints, 1, dtype=x.dtype, device=x.device)
        valid = conf * frame_mask_t[:, :, None, None, None].to(dtype=x.dtype)
        xy_mean = (xy * valid).sum(dim=(1, 2, 3), keepdim=True) / valid.sum(dim=(1, 2, 3), keepdim=True).clamp_min(self.eps)
        centered = xy - xy_mean
        scale = torch.sqrt((centered.square().sum(dim=-1, keepdim=True) * valid).sum(dim=(1, 2, 3), keepdim=True) / valid.sum(dim=(1, 2, 3), keepdim=True).clamp_min(self.eps))
        scale = scale.clamp_min(self.eps)

        x_norm = x.clone()
        x_norm[..., :2] = centered / scale

        if not self.compute_motion:
            return x_norm

        vel = torch.zeros(batch, frames, persons, joints, 2, dtype=x.dtype, device=x.device)
        vel[:, 1:] = (xy[:, 1:] - xy[:, :-1]) / scale
        acc = torch.zeros_like(vel)
        acc[:, 1:] = vel[:, 1:] - vel[:, :-1]
        return torch.cat([x_norm, vel, acc], dim=-1)


class PersonEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        person_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.keypoint_mlp = _mlp(in_dim, hidden_dim, hidden_dim, dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=max(hidden_dim * 2, person_dim),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, person_dim))

    def forward(self, x_feat: torch.Tensor, person_mask: torch.Tensor, frame_mask: Optional[torch.Tensor]) -> torch.Tensor:
        batch, frames, persons, _joints, _channels = x_feat.shape
        frame_mask_t = _frame_mask_or_ones(frame_mask, batch, frames, x_feat.device)

        h = self.keypoint_mlp(x_feat).mean(dim=3)
        h = h.permute(0, 2, 1, 3).contiguous().view(batch * persons, frames, -1)

        valid = (person_mask[:, :, None] & frame_mask_t[:, None, :]).reshape(batch * persons, frames)
        padding = ~valid
        all_padding = padding.all(dim=1)
        if all_padding.any():
            padding = padding.clone()
            padding[all_padding, 0] = False

        h = self.temporal_encoder(h, src_key_padding_mask=padding)
        valid_f = valid.to(dtype=h.dtype).unsqueeze(-1)
        pooled = (h * valid_f).sum(dim=1) / valid_f.sum(dim=1).clamp_min(1.0)
        pooled = self.out(pooled).view(batch, persons, -1)
        return pooled * person_mask[:, :, None].to(dtype=pooled.dtype)


class PairFeatureBuilder(nn.Module):
    rel_dim = 7

    def __init__(self, person_dim: int, eps: float = 1e-6):
        super().__init__()
        self.person_dim = int(person_dim)
        self.eps = float(eps)
        self.out_dim = 3 * self.person_dim + self.rel_dim

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        person_mask: torch.Tensor,
        frame_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, frames, persons, joints, channels = x.shape
        frame_mask_t = _frame_mask_or_ones(frame_mask, batch, frames, x.device)
        pair_mask = build_pair_mask(person_mask)

        xy = torch.nan_to_num(x[..., :2], nan=0.0, posinf=0.0, neginf=0.0)
        if channels >= 3:
            conf = torch.nan_to_num(x[..., 2:3], nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
        else:
            conf = torch.ones(batch, frames, persons, joints, 1, dtype=x.dtype, device=x.device)
        frame_valid = frame_mask_t[:, :, None, None, None].to(dtype=x.dtype)
        conf = conf * frame_valid
        conf_sum = conf.sum(dim=3).clamp_min(self.eps)
        center = (xy * conf).sum(dim=3) / conf_sum

        valid_person_frame = (conf.sum(dim=(3, 4)) > 0.0) & frame_mask_t[:, :, None]
        valid_pair_frame = (
            valid_person_frame[:, :, :, None]
            & valid_person_frame[:, :, None, :]
            & pair_mask[:, None, :, :]
        )

        xy_scale = torch.sqrt(
            ((xy - (xy * conf).sum(dim=(1, 2, 3), keepdim=True) / conf.sum(dim=(1, 2, 3), keepdim=True).clamp_min(self.eps)).square().sum(dim=-1, keepdim=True) * conf)
            .sum(dim=(1, 2, 3), keepdim=True)
            / conf.sum(dim=(1, 2, 3), keepdim=True).clamp_min(self.eps)
        ).clamp_min(self.eps)

        center_i = center[:, :, :, None, :]
        center_j = center[:, :, None, :, :]
        center_delta = center_j - center_i
        center_dist = torch.norm(center_delta, dim=-1) / xy_scale.view(batch, 1, 1, 1)

        center_vel = torch.zeros_like(center)
        center_vel[:, 1:] = center[:, 1:] - center[:, :-1]
        rel_vel = center_vel[:, :, None, :, :] - center_vel[:, :, :, None, :]
        rel_vel_mag = torch.norm(rel_vel, dim=-1) / xy_scale.view(batch, 1, 1, 1)

        kxy_i = xy[:, :, :, None, :, :]
        kxy_j = xy[:, :, None, :, :, :]
        kdist = torch.norm(kxy_j - kxy_i, dim=-1) / xy_scale.view(batch, 1, 1, 1, 1)
        kvalid = (conf[:, :, :, None, :, 0] > 0.0) & (conf[:, :, None, :, :, 0] > 0.0)
        kdist_min = _masked_min(kdist, kvalid, dim=4)

        mean_center = _masked_mean(center_dist, valid_pair_frame, dim=1)
        min_center = _masked_min(center_dist, valid_pair_frame, dim=1)
        mean_rel_vel = _masked_mean(rel_vel_mag, valid_pair_frame, dim=1)
        min_kpt_dist = _masked_min(kdist_min, valid_pair_frame, dim=1)
        mean_kpt_dist = _masked_mean(kdist_min, valid_pair_frame, dim=1)

        time_idx = torch.arange(frames, device=x.device, dtype=torch.long)[None, :, None, None]
        first_idx = torch.where(valid_pair_frame, time_idx, torch.full_like(time_idx, frames)).min(dim=1).values.clamp_max(frames - 1)
        last_idx = torch.where(valid_pair_frame, time_idx, torch.zeros_like(time_idx)).max(dim=1).values
        first_center = _gather_time(center_dist, first_idx)
        final_center = _gather_time(center_dist, last_idx)
        delta_center = final_center - first_center

        rel = torch.stack(
            [mean_center, min_center, final_center, delta_center, mean_rel_vel, min_kpt_dist, mean_kpt_dist],
            dim=-1,
        )
        rel = torch.nan_to_num(rel, nan=0.0, posinf=0.0, neginf=0.0)

        h_i = h[:, :, None, :].expand(-1, persons, persons, -1)
        h_j = h[:, None, :, :].expand(-1, persons, persons, -1)
        pair_feat = torch.cat([h_i + h_j, torch.abs(h_i - h_j), h_i * h_j, rel], dim=-1)
        return pair_feat * pair_mask[:, :, :, None].to(dtype=pair_feat.dtype), pair_mask


class NRILayer(nn.Module):
    def __init__(self, person_dim: int, edge_dim: int, msg_dim: int, dropout: float):
        super().__init__()
        self.msg = _mlp(person_dim * 2 + edge_dim, max(msg_dim, edge_dim), msg_dim, dropout)
        self.node = _mlp(person_dim + msg_dim, max(person_dim, msg_dim), person_dim, dropout)
        self.edge = _mlp(person_dim * 2 + edge_dim, max(person_dim, edge_dim), edge_dim, dropout)
        self.node_norm = nn.LayerNorm(person_dim)
        self.edge_norm = nn.LayerNorm(edge_dim)

    def forward(self, h: torch.Tensor, e: torch.Tensor, pair_mask: torch.Tensor, person_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        persons = h.shape[1]
        h_i = h[:, :, None, :].expand(-1, persons, persons, -1)
        h_j = h[:, None, :, :].expand(-1, persons, persons, -1)
        pair_mask_f = pair_mask[:, :, :, None].to(dtype=h.dtype)

        msg = self.msg(torch.cat([h_i, h_j, e], dim=-1)) * pair_mask_f
        agg = msg.sum(dim=2)
        h = self.node_norm(h + self.node(torch.cat([h, agg], dim=-1)))
        h = h * person_mask[:, :, None].to(dtype=h.dtype)

        h_i = h[:, :, None, :].expand(-1, persons, persons, -1)
        h_j = h[:, None, :, :].expand(-1, persons, persons, -1)
        e = self.edge_norm(e + self.edge(torch.cat([h_i, h_j, e], dim=-1)))
        return h, e * pair_mask_f


class NRIInteractionHead(nn.Module):
    def __init__(
        self,
        keypoint_dim: int,
        num_keypoints: int,
        hidden_dim: int = 128,
        person_dim: int = 256,
        edge_dim: int = 256,
        msg_dim: int = 256,
        num_temporal_layers: int = 2,
        num_nri_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        aggregator: str = "topk_noisy_or",
        top_k: int = 3,
        compute_motion: bool = True,
    ):
        super().__init__()
        self.keypoint_dim = int(keypoint_dim)
        self.num_keypoints = int(num_keypoints)
        self.person_dim = int(person_dim)
        self.edge_dim = int(edge_dim)
        self.aggregator = str(aggregator)
        self.top_k = int(top_k)

        self.feature_builder = KeypointFeatureBuilder(keypoint_dim, compute_motion=compute_motion)
        self.person_encoder = PersonEncoder(
            in_dim=self.feature_builder.out_dim,
            hidden_dim=hidden_dim,
            person_dim=person_dim,
            num_layers=num_temporal_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.pair_builder = PairFeatureBuilder(person_dim)
        self.edge_in = _mlp(self.pair_builder.out_dim, edge_dim, edge_dim, dropout)
        self.nri_layers = nn.ModuleList(
            [NRILayer(person_dim=person_dim, edge_dim=edge_dim, msg_dim=msg_dim, dropout=dropout) for _ in range(num_nri_layers)]
        )
        self.pair_head = nn.Sequential(
            nn.LayerNorm(edge_dim),
            nn.Linear(edge_dim, edge_dim // 2 if edge_dim >= 2 else edge_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(edge_dim // 2 if edge_dim >= 2 else edge_dim, 1),
        )

    def forward(
        self,
        X: torch.Tensor,
        person_mask: torch.Tensor,
        frame_mask: Optional[torch.Tensor] = None,
        y_video: Optional[torch.Tensor] = None,
        pos_pairs: Optional[list[list[tuple[int, int]]]] = None,
        return_loss: bool = False,
    ) -> dict:
        if X.ndim != 5:
            raise ValueError(f"X must have shape [B,T,N,K,C], got {tuple(X.shape)}")
        batch, frames, persons, joints, channels = X.shape
        if joints != self.num_keypoints:
            raise ValueError(f"Expected num_keypoints={self.num_keypoints}, got K={joints}")
        if channels != self.keypoint_dim:
            raise ValueError(f"Expected keypoint_dim={self.keypoint_dim}, got C={channels}")
        if person_mask.ndim == 3:
            person_mask = person_mask.any(dim=1)
        person_mask = person_mask.to(device=X.device, dtype=torch.bool)
        frame_mask_t = _frame_mask_or_ones(frame_mask, batch, frames, X.device)

        X_feat = self.feature_builder(X, frame_mask_t)
        h = self.person_encoder(X_feat, person_mask, frame_mask_t)
        pair_feat, pair_mask = self.pair_builder(X, h, person_mask, frame_mask_t)
        e = self.edge_in(pair_feat) * pair_mask[:, :, :, None].to(dtype=pair_feat.dtype)
        for layer in self.nri_layers:
            h, e = layer(h, e, pair_mask, person_mask)

        pair_logits = self.pair_head(e).squeeze(-1)
        pair_logits = 0.5 * (pair_logits + pair_logits.transpose(1, 2))
        pair_logits = pair_logits.masked_fill(~pair_mask, 0.0)
        p_pair = torch.sigmoid(pair_logits)
        p_pair = 0.5 * (p_pair + p_pair.transpose(1, 2))
        p_pair = p_pair * pair_mask.to(dtype=p_pair.dtype)

        if self.aggregator == "noisy_or":
            p_video = noisy_or_aggregate(p_pair, pair_mask)
        elif self.aggregator == "topk_noisy_or":
            p_video = topk_noisy_or_aggregate(p_pair, pair_mask, self.top_k)
        else:
            raise ValueError(f"unsupported aggregator: {self.aggregator!r}")

        out = {
            "p_video": p_video,
            "p_video_interaction": p_video,
            "p_pair": p_pair,
            "p_pair_interaction": p_pair,
            "pair_logits": pair_logits,
            "pair_mask": pair_mask,
            "valid_pairs": valid_pairs_from_mask(pair_mask),
            "h_node": h,
            "e_pair": e,
        }
        if return_loss or y_video is not None:
            if y_video is None:
                raise ValueError("y_video is required when return_loss=True")
            loss_dict = compute_interaction_loss(p_video, p_pair, pair_mask, y_video, pos_pairs)
            out["loss"] = loss_dict["total_loss"]
            out["loss_dict"] = loss_dict
        return out


class CandidatePairSelector:
    @staticmethod
    def select_inference(
        p_pair_interaction: torch.Tensor,
        valid_pairs: list[list[tuple[int, int]]],
        top_m: int = 3,
    ) -> list[list[tuple[int, int]]]:
        selected: list[list[tuple[int, int]]] = []
        for b, pairs in enumerate(valid_pairs):
            scored = [
                (float(p_pair_interaction[b, i, j].detach().cpu().item()), (int(i), int(j)))
                for i, j in pairs
            ]
            scored.sort(key=lambda item: item[0], reverse=True)
            selected.append([pair for _score, pair in scored[: max(0, int(top_m))]])
        return selected

    @staticmethod
    def select_train(
        p_pair_interaction: torch.Tensor,
        valid_pairs: list[list[tuple[int, int]]],
        pos_pairs: Optional[list[list[tuple[int, int]]]],
        top_m: int = 3,
    ) -> list[list[tuple[int, int]]]:
        selected = CandidatePairSelector.select_inference(p_pair_interaction, valid_pairs, top_m=top_m)
        if pos_pairs is None:
            return selected
        out: list[list[tuple[int, int]]] = []
        for b, pairs in enumerate(selected):
            valid_set = {_normalize_pair(pair) for pair in valid_pairs[b]}
            ordered = [_normalize_pair(pair) for pair in pairs]
            seen = set(ordered)
            for raw_pair in pos_pairs[b]:
                pair = _normalize_pair(raw_pair)
                if pair in valid_set and pair not in seen:
                    ordered.append(pair)
                    seen.add(pair)
            out.append(ordered)
        return out


class PairTemporalEncoder(nn.Module):
    def __init__(
        self,
        keypoint_dim: int,
        num_keypoints: int,
        hidden_dim: int,
        out_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.keypoint_dim = int(keypoint_dim)
        self.num_keypoints = int(num_keypoints)
        self.in_dim = self.num_keypoints * self.keypoint_dim * 3
        self.out_dim = int(out_dim)
        self.frame_mlp = _mlp(self.in_dim, hidden_dim, hidden_dim, dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=max(hidden_dim * 2, self.out_dim),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, self.out_dim))

    def forward(
        self,
        X: torch.Tensor,
        selected_pairs: list[list[tuple[int, int]]],
        frame_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if X.ndim != 5:
            raise ValueError(f"X must have shape [B,T,N,K,C], got {tuple(X.shape)}")
        batch, frames, persons, joints, channels = X.shape
        if joints != self.num_keypoints:
            raise ValueError(f"Expected num_keypoints={self.num_keypoints}, got K={joints}")
        if channels != self.keypoint_dim:
            raise ValueError(f"Expected keypoint_dim={self.keypoint_dim}, got C={channels}")
        max_pairs = max((len(pairs) for pairs in selected_pairs), default=0)
        if max_pairs == 0:
            return (
                torch.zeros(batch, 0, self.out_dim, dtype=X.dtype, device=X.device),
                torch.zeros(batch, 0, dtype=torch.bool, device=X.device),
            )

        frame_mask_t = _frame_mask_or_ones(frame_mask, batch, frames, X.device)
        pair_seq = torch.zeros(batch, max_pairs, frames, self.in_dim, dtype=X.dtype, device=X.device)
        selected_mask = torch.zeros(batch, max_pairs, dtype=torch.bool, device=X.device)
        clean_x = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        for b, pairs in enumerate(selected_pairs):
            for e_idx, raw_pair in enumerate(pairs[:max_pairs]):
                i, j = _normalize_pair(raw_pair)
                if i < 0 or j < 0 or i >= persons or j >= persons or i == j:
                    continue
                xi = clean_x[b, :, i].reshape(frames, -1)
                xj = clean_x[b, :, j].reshape(frames, -1)
                pair_seq[b, e_idx] = torch.cat([xi, xj, xi - xj], dim=-1)
                selected_mask[b, e_idx] = True

        h = self.frame_mlp(pair_seq.view(batch * max_pairs, frames, self.in_dim))
        valid = (selected_mask[:, :, None] & frame_mask_t[:, None, :]).view(batch * max_pairs, frames)
        padding = ~valid
        all_padding = padding.all(dim=1)
        if all_padding.any():
            padding = padding.clone()
            padding[all_padding, 0] = False
        h = self.temporal_encoder(h, src_key_padding_mask=padding)
        valid_f = valid.to(dtype=h.dtype).unsqueeze(-1)
        pooled = (h * valid_f).sum(dim=1) / valid_f.sum(dim=1).clamp_min(1.0)
        pooled = self.out(pooled).view(batch, max_pairs, self.out_dim)
        return pooled * selected_mask[:, :, None].to(dtype=pooled.dtype), selected_mask


class TypeNRILayer(nn.Module):
    def __init__(self, person_dim: int, type_dim: int, msg_dim: int, dropout: float):
        super().__init__()
        msg_in = person_dim * 2 + type_dim
        self.msg_friendly = _mlp(msg_in, max(msg_dim, type_dim), msg_dim, dropout)
        self.msg_unfriendly = _mlp(msg_in, max(msg_dim, type_dim), msg_dim, dropout)
        self.node = _mlp(person_dim + msg_dim, max(person_dim, msg_dim), person_dim, dropout)
        self.edge = _mlp(person_dim * 2 + type_dim + msg_dim, max(person_dim, type_dim, msg_dim), type_dim, dropout)
        self.node_norm = nn.LayerNorm(person_dim)
        self.edge_norm = nn.LayerNorm(type_dim)

    def forward(
        self,
        z: torch.Tensor,
        a: torch.Tensor,
        q_pair_type: torch.Tensor,
        p_selected: torch.Tensor,
        selected_pairs: list[list[tuple[int, int]]],
        selected_mask: torch.Tensor,
        detach_gate: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, persons, _ = z.shape
        max_pairs = a.shape[1]
        if max_pairs == 0:
            return z, a

        zi = torch.zeros(batch, max_pairs, z.shape[-1], dtype=z.dtype, device=z.device)
        zj = torch.zeros_like(zi)
        for b, pairs in enumerate(selected_pairs):
            for e_idx, raw_pair in enumerate(pairs[:max_pairs]):
                i, j = _normalize_pair(raw_pair)
                if 0 <= i < persons and 0 <= j < persons and i != j:
                    zi[b, e_idx] = z[b, i]
                    zj[b, e_idx] = z[b, j]

        msg_in = torch.cat([zi, zj, a], dim=-1)
        msg_f = self.msg_friendly(msg_in)
        msg_u = self.msg_unfriendly(msg_in)
        gate = p_selected.detach() if detach_gate else p_selected
        msg = gate[:, :, None] * (
            q_pair_type[:, :, 0:1] * msg_f + q_pair_type[:, :, 1:2] * msg_u
        )
        msg = msg * selected_mask[:, :, None].to(dtype=msg.dtype)

        agg = torch.zeros(batch, persons, msg.shape[-1], dtype=z.dtype, device=z.device)
        for b, pairs in enumerate(selected_pairs):
            for e_idx, raw_pair in enumerate(pairs[:max_pairs]):
                if not bool(selected_mask[b, e_idx]):
                    continue
                i, j = _normalize_pair(raw_pair)
                if 0 <= i < persons and 0 <= j < persons and i != j:
                    agg[b, i] = agg[b, i] + msg[b, e_idx]
                    agg[b, j] = agg[b, j] + msg[b, e_idx]

        z = self.node_norm(z + self.node(torch.cat([z, agg], dim=-1)))

        zi = torch.zeros(batch, max_pairs, z.shape[-1], dtype=z.dtype, device=z.device)
        zj = torch.zeros_like(zi)
        for b, pairs in enumerate(selected_pairs):
            for e_idx, raw_pair in enumerate(pairs[:max_pairs]):
                i, j = _normalize_pair(raw_pair)
                if 0 <= i < persons and 0 <= j < persons and i != j:
                    zi[b, e_idx] = z[b, i]
                    zj[b, e_idx] = z[b, j]
        a = self.edge_norm(a + self.edge(torch.cat([zi, zj, a, msg], dim=-1)))
        return z, a * selected_mask[:, :, None].to(dtype=a.dtype)


class TypeNRIHead(nn.Module):
    def __init__(
        self,
        keypoint_dim: int,
        num_keypoints: int,
        person_dim: int = 256,
        edge_dim: int = 256,
        pair_temporal_dim: int = 256,
        hidden_dim: int = 256,
        msg_dim: int = 256,
        num_temporal_layers: int = 1,
        num_nri_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        top_m: int = 3,
        detach_interaction_gate: bool = True,
    ):
        super().__init__()
        self.keypoint_dim = int(keypoint_dim)
        self.num_keypoints = int(num_keypoints)
        self.person_dim = int(person_dim)
        self.edge_dim = int(edge_dim)
        self.pair_temporal_dim = int(pair_temporal_dim)
        self.top_m = int(top_m)
        self.detach_interaction_gate = bool(detach_interaction_gate)
        self.pair_temporal = PairTemporalEncoder(
            keypoint_dim=keypoint_dim,
            num_keypoints=num_keypoints,
            hidden_dim=hidden_dim,
            out_dim=pair_temporal_dim,
            num_heads=num_heads,
            num_layers=num_temporal_layers,
            dropout=dropout,
        )
        self.edge_in = _mlp(person_dim * 3 + edge_dim + 1 + pair_temporal_dim, hidden_dim, edge_dim, dropout)
        self.layers = nn.ModuleList(
            [TypeNRILayer(person_dim=person_dim, type_dim=edge_dim, msg_dim=msg_dim, dropout=dropout) for _ in range(num_nri_layers)]
        )
        self.type_out = nn.Sequential(
            nn.LayerNorm(edge_dim),
            nn.Linear(edge_dim, max(2, edge_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(2, edge_dim // 2), 2),
        )

    def forward(
        self,
        X: torch.Tensor,
        h_node: torch.Tensor,
        e_pair: torch.Tensor,
        p_pair_interaction: torch.Tensor,
        valid_pairs: list[list[tuple[int, int]]],
        selected_pairs: list[list[tuple[int, int]]],
        frame_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        batch, _frames, persons, _joints, _channels = X.shape
        max_pairs = max((len(pairs) for pairs in selected_pairs), default=0)
        if max_pairs == 0:
            q_video = torch.full((batch, 2), 0.5, dtype=X.dtype, device=X.device)
            return {
                "q_pair_type": torch.zeros(batch, 0, 2, dtype=X.dtype, device=X.device),
                "q_pair_type_logits": torch.zeros(batch, 0, 2, dtype=X.dtype, device=X.device),
                "q_video_type": q_video,
                "selected_pairs": selected_pairs,
                "selected_pair_mask": torch.zeros(batch, 0, dtype=torch.bool, device=X.device),
            }

        u_pair, selected_mask = self.pair_temporal(X, selected_pairs, frame_mask=frame_mask)
        h_i = torch.zeros(batch, max_pairs, self.person_dim, dtype=h_node.dtype, device=h_node.device)
        h_j = torch.zeros_like(h_i)
        e_sel = torch.zeros(batch, max_pairs, self.edge_dim, dtype=e_pair.dtype, device=e_pair.device)
        p_sel = torch.zeros(batch, max_pairs, dtype=p_pair_interaction.dtype, device=p_pair_interaction.device)
        valid_sets = [{_normalize_pair(pair) for pair in pairs} for pairs in valid_pairs]

        for b, pairs in enumerate(selected_pairs):
            for e_idx, raw_pair in enumerate(pairs[:max_pairs]):
                i, j = _normalize_pair(raw_pair)
                if i < 0 or j < 0 or i >= persons or j >= persons or i == j:
                    selected_mask[b, e_idx] = False
                    continue
                if (i, j) not in valid_sets[b]:
                    selected_mask[b, e_idx] = False
                    continue
                h_i[b, e_idx] = h_node[b, i]
                h_j[b, e_idx] = h_node[b, j]
                e_sel[b, e_idx] = e_pair[b, i, j]
                p_sel[b, e_idx] = p_pair_interaction[b, i, j]

        edge_input = torch.cat([h_i + h_j, torch.abs(h_i - h_j), h_i * h_j, e_sel, p_sel[:, :, None], u_pair], dim=-1)
        a = self.edge_in(edge_input) * selected_mask[:, :, None].to(dtype=edge_input.dtype)
        logits = self.type_out(a)
        q = F.softmax(logits, dim=-1) * selected_mask[:, :, None].to(dtype=logits.dtype)
        z = h_node
        for layer in self.layers:
            z, a = layer(
                z,
                a,
                q,
                p_sel,
                selected_pairs,
                selected_mask,
                detach_gate=self.detach_interaction_gate,
            )
            logits = self.type_out(a)
            q = F.softmax(logits, dim=-1) * selected_mask[:, :, None].to(dtype=logits.dtype)

        weights = p_sel * selected_mask.to(dtype=p_sel.dtype)
        denom = weights.sum(dim=1, keepdim=True)
        q_weighted = (q * weights[:, :, None]).sum(dim=1) / denom.clamp_min(1e-6)
        default_q = torch.full_like(q_weighted, 0.5)
        q_video = torch.where((denom > 0.0).expand_as(q_weighted), q_weighted, default_q)
        return {
            "q_pair_type": q,
            "q_pair_type_logits": logits,
            "q_video_type": q_video,
            "selected_pairs": selected_pairs,
            "selected_pair_mask": selected_mask,
        }


def compute_type_loss(
    q_pair_type: torch.Tensor,
    q_video_type: torch.Tensor,
    selected_pairs: list[list[tuple[int, int]]],
    y_interaction: torch.Tensor,
    y_type: Optional[torch.Tensor],
    pos_pairs: Optional[list[list[tuple[int, int]]]],
    pair_type_labels: Optional[list[dict[tuple[int, int], int]]] = None,
    lambda_type_pair: float = 1.0,
    lambda_type_video: float = 0.5,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    device = q_video_type.device
    y_inter = y_interaction.to(device=device, dtype=torch.float32).view(-1)
    zero = q_video_type.sum() * 0.0
    if y_type is None:
        return {"total_loss": zero, "pair_loss": zero, "video_loss": zero}
    y_t = y_type.to(device=device, dtype=torch.long).view(-1)
    if pos_pairs is None:
        pos_pairs = [[] for _ in range(q_video_type.shape[0])]
    if pair_type_labels is None:
        pair_type_labels = [{} for _ in range(q_video_type.shape[0])]

    pair_losses: list[torch.Tensor] = []
    video_losses: list[torch.Tensor] = []
    q_pair_c = q_pair_type.clamp(eps, 1.0)
    q_video_c = q_video_type.clamp(eps, 1.0)

    for b in range(q_video_type.shape[0]):
        if y_inter[b] < 0.5:
            continue
        label = int(y_t[b].detach().cpu().item())
        video_losses.append(F.nll_loss(torch.log(q_video_c[b : b + 1]), torch.tensor([label], dtype=torch.long, device=device)))
        selected_index = {_normalize_pair(pair): e_idx for e_idx, pair in enumerate(selected_pairs[b])}
        normalized_pair_labels = {_normalize_pair(pair): int(value) for pair, value in pair_type_labels[b].items()}
        for raw_pair in pos_pairs[b]:
            pair = _normalize_pair(raw_pair)
            e_idx = selected_index.get(pair)
            if e_idx is None or e_idx >= q_pair_type.shape[1]:
                continue
            pair_label = int(normalized_pair_labels.get(pair, label))
            pair_losses.append(
                F.nll_loss(torch.log(q_pair_c[b : b + 1, e_idx, :]), torch.tensor([pair_label], dtype=torch.long, device=device))
            )

    pair_loss = torch.stack(pair_losses).mean() if pair_losses else zero
    video_loss = torch.stack(video_losses).mean() if video_losses else zero
    total = float(lambda_type_pair) * pair_loss + float(lambda_type_video) * video_loss
    return {"total_loss": total, "pair_loss": pair_loss, "video_loss": video_loss}


class CascadeInteractionModel(nn.Module):
    def __init__(
        self,
        interaction_head: NRIInteractionHead,
        type_head: TypeNRIHead,
        lambda_type: float = 1.0,
    ):
        super().__init__()
        self.interaction_head = interaction_head
        self.type_head = type_head
        self.lambda_type = float(lambda_type)

    def forward(
        self,
        X: torch.Tensor,
        person_mask: torch.Tensor,
        frame_mask: Optional[torch.Tensor] = None,
        y_interaction: Optional[torch.Tensor] = None,
        pos_pairs: Optional[list[list[tuple[int, int]]]] = None,
        y_type: Optional[torch.Tensor] = None,
        pair_type_labels: Optional[list[dict[tuple[int, int], int]]] = None,
        mode: str = "train",
    ) -> dict:
        inter = self.interaction_head(X, person_mask, frame_mask=frame_mask)
        if mode == "train" and pos_pairs is not None:
            selected_pairs = CandidatePairSelector.select_train(
                inter["p_pair"],
                inter["valid_pairs"],
                pos_pairs,
                top_m=self.type_head.top_m,
            )
        else:
            selected_pairs = CandidatePairSelector.select_inference(inter["p_pair"], inter["valid_pairs"], top_m=self.type_head.top_m)
        type_out = self.type_head(
            X,
            inter["h_node"],
            inter["e_pair"],
            inter["p_pair"],
            inter["valid_pairs"],
            selected_pairs,
            frame_mask=frame_mask,
        )

        p_interaction = inter["p_video"]
        q_video = type_out["q_video_type"]
        p_no = (1.0 - p_interaction).clamp(0.0, 1.0)
        p_friendly = p_interaction * q_video[:, 0]
        p_unfriendly = p_interaction * q_video[:, 1]
        out = {
            "p_interaction": p_interaction,
            "p_video_interaction": p_interaction,
            "p_pair_interaction": inter["p_pair"],
            "valid_pairs": inter["valid_pairs"],
            "pair_mask": inter["pair_mask"],
            "h_node": inter["h_node"],
            "e_pair": inter["e_pair"],
            "p_friendly_given_interaction": q_video[:, 0],
            "p_unfriendly_given_interaction": q_video[:, 1],
            "p_no_interaction": p_no,
            "p_friendly_interaction": p_friendly,
            "p_unfriendly_interaction": p_unfriendly,
            **type_out,
        }
        losses: dict[str, torch.Tensor] = {}
        if y_interaction is not None:
            interaction_losses = compute_interaction_loss(inter["p_video"], inter["p_pair"], inter["pair_mask"], y_interaction, pos_pairs)
            losses["interaction_loss"] = interaction_losses["total_loss"]
            out["interaction_loss_dict"] = interaction_losses
        if y_interaction is not None and y_type is not None:
            type_losses = compute_type_loss(
                type_out["q_pair_type"],
                type_out["q_video_type"],
                selected_pairs,
                y_interaction,
                y_type,
                pos_pairs,
                pair_type_labels=pair_type_labels,
            )
            losses["type_loss"] = type_losses["total_loss"]
            out["type_loss_dict"] = type_losses
        if losses:
            total = losses.get("interaction_loss", p_interaction.sum() * 0.0) + self.lambda_type * losses.get("type_loss", p_interaction.sum() * 0.0)
            out.update(losses)
            out["total_loss"] = total
        return out


class NRIValenceHead(nn.Module):
    """NRI-based friendly/unfriendly classifier for already-candidate interaction pairs."""

    def __init__(
        self,
        keypoint_dim: int,
        num_keypoints: int,
        hidden_dim: int = 128,
        person_dim: int = 256,
        edge_dim: int = 256,
        msg_dim: int = 256,
        pair_temporal_dim: int = 256,
        num_temporal_layers: int = 2,
        num_nri_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        aggregator: str = "topk_noisy_or",
        top_k: int = 3,
        top_m: int = 3,
        compute_motion: bool = True,
        detach_interaction_gate: bool = True,
    ):
        super().__init__()
        self.context_head = NRIInteractionHead(
            keypoint_dim=keypoint_dim,
            num_keypoints=num_keypoints,
            hidden_dim=hidden_dim,
            person_dim=person_dim,
            edge_dim=edge_dim,
            msg_dim=msg_dim,
            num_temporal_layers=num_temporal_layers,
            num_nri_layers=num_nri_layers,
            num_heads=num_heads,
            dropout=dropout,
            aggregator=aggregator,
            top_k=top_k,
            compute_motion=compute_motion,
        )
        self.type_head = TypeNRIHead(
            keypoint_dim=keypoint_dim,
            num_keypoints=num_keypoints,
            person_dim=person_dim,
            edge_dim=edge_dim,
            pair_temporal_dim=pair_temporal_dim,
            hidden_dim=hidden_dim,
            msg_dim=msg_dim,
            num_temporal_layers=max(1, min(2, int(num_temporal_layers))),
            num_nri_layers=num_nri_layers,
            num_heads=num_heads,
            dropout=dropout,
            top_m=top_m,
            detach_interaction_gate=detach_interaction_gate,
        )

    def forward(
        self,
        X: torch.Tensor,
        person_mask: torch.Tensor,
        frame_mask: Optional[torch.Tensor] = None,
        pos_pairs: Optional[list[list[tuple[int, int]]]] = None,
        mode: str = "inference",
    ) -> dict:
        inter = self.context_head(X, person_mask, frame_mask=frame_mask)
        if mode == "train" and pos_pairs is not None:
            selected_pairs = CandidatePairSelector.select_train(
                inter["p_pair"],
                inter["valid_pairs"],
                pos_pairs,
                top_m=self.type_head.top_m,
            )
        else:
            selected_pairs = CandidatePairSelector.select_inference(inter["p_pair"], inter["valid_pairs"], top_m=self.type_head.top_m)
        type_out = self.type_head(
            X,
            inter["h_node"],
            inter["e_pair"],
            inter["p_pair"],
            inter["valid_pairs"],
            selected_pairs,
            frame_mask=frame_mask,
        )
        logits = torch.log(type_out["q_video_type"].clamp(1e-6, 1.0))
        return {
            "logits": logits,
            "probs": type_out["q_video_type"],
            "p_pair_interaction": inter["p_pair"],
            "valid_pairs": inter["valid_pairs"],
            **type_out,
        }


def compute_interaction_loss(
    p_video: torch.Tensor,
    p_pair: torch.Tensor,
    pair_mask: torch.Tensor,
    y_video: torch.Tensor,
    pos_pairs: Optional[list[list[tuple[int, int]]]],
    lambda_bag: float = 1.0,
    lambda_pos_pair: float = 2.0,
    lambda_neg_video: float = 1.0,
    lambda_neg_pair: float = 1.0,
    lambda_sparse: float = 0.0,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    device = p_video.device
    y = y_video.to(device=device, dtype=torch.float32).view(-1)
    batch, persons, _ = p_pair.shape
    if y.shape[0] != batch:
        raise ValueError(f"y_video length {y.shape[0]} does not match batch {batch}")
    if pos_pairs is None:
        pos_pairs = [[] for _ in range(batch)]

    valid_upper = _upper_pair_mask(pair_mask)
    zero = p_video.sum() * 0.0
    bag_losses: list[torch.Tensor] = []
    pos_pair_losses: list[torch.Tensor] = []
    neg_pair_losses: list[torch.Tensor] = []
    sparse_losses: list[torch.Tensor] = []

    p_video_c = p_video.clamp(eps, 1.0 - eps)
    p_pair_c = p_pair.clamp(eps, 1.0 - eps)

    for b in range(batch):
        if y[b] >= 0.5:
            bag_losses.append(-torch.log(p_video_c[b]))
            known = torch.zeros(persons, persons, dtype=torch.bool, device=device)
            for raw_i, raw_j in pos_pairs[b]:
                i, j = sorted((int(raw_i), int(raw_j)))
                if i == j or i < 0 or j < 0 or i >= persons or j >= persons:
                    continue
                known[i, j] = True
                known[j, i] = True
                if pair_mask[b, i, j]:
                    pos_pair_losses.append(-torch.log(p_pair_c[b, i, j]))
            if lambda_sparse > 0:
                unknown = valid_upper[b] & ~known
                if unknown.any():
                    sparse_losses.append(p_pair[b][unknown].mean())
        else:
            bag_losses.append(-torch.log((1.0 - p_video_c[b]).clamp_min(eps)))
            neg_pairs = valid_upper[b]
            if neg_pairs.any():
                neg_pair_losses.append(-torch.log((1.0 - p_pair_c[b][neg_pairs]).clamp_min(eps)).mean())

    bag_loss = torch.stack(bag_losses).mean() if bag_losses else zero
    pos_pair_loss = torch.stack(pos_pair_losses).mean() if pos_pair_losses else zero
    neg_pair_loss = torch.stack(neg_pair_losses).mean() if neg_pair_losses else zero
    sparse_loss = torch.stack(sparse_losses).mean() if sparse_losses else zero

    pos_mask = y >= 0.5
    neg_mask = ~pos_mask
    pos_bag = torch.stack([bag_losses[i] for i in range(batch) if pos_mask[i]]).mean() if pos_mask.any() else zero
    neg_bag = torch.stack([bag_losses[i] for i in range(batch) if neg_mask[i]]).mean() if neg_mask.any() else zero
    total = (
        float(lambda_bag) * pos_bag
        + float(lambda_pos_pair) * pos_pair_loss
        + float(lambda_sparse) * sparse_loss
        + float(lambda_neg_video) * neg_bag
        + float(lambda_neg_pair) * neg_pair_loss
    )
    return {
        "total_loss": total,
        "bag_loss": bag_loss,
        "pos_pair_loss": pos_pair_loss,
        "neg_pair_loss": neg_pair_loss,
        "sparse_loss": sparse_loss,
    }


def get_top_pairs(p_pair: torch.Tensor, pair_mask: torch.Tensor, top_k: int = 5) -> list[list[dict]]:
    valid = _upper_pair_mask(pair_mask.to(dtype=torch.bool))
    out: list[list[dict]] = []
    for b in range(p_pair.shape[0]):
        indices = valid[b].nonzero(as_tuple=False)
        if indices.numel() == 0:
            out.append([])
            continue
        probs = p_pair[b][valid[b]]
        k = min(int(top_k), int(probs.numel()))
        values, order = probs.topk(k)
        rows = []
        for value, idx in zip(values.detach().cpu().tolist(), order.detach().cpu().tolist()):
            i, j = indices[idx].detach().cpu().tolist()
            rows.append({"person_i": int(i), "person_j": int(j), "p_interaction": float(value)})
        out.append(rows)
    return out


if __name__ == "__main__":
    x = torch.randn(2, 8, 4, 17, 3)
    person_mask = torch.ones(2, 4, dtype=torch.bool)
    y = torch.tensor([1.0, 0.0])
    model = NRIInteractionHead(keypoint_dim=3, num_keypoints=17, hidden_dim=32, person_dim=64, edge_dim=64, msg_dim=64)
    output = model(x, person_mask, y_video=y, pos_pairs=[[(0, 2)], []], return_loss=True)
    print({k: tuple(v.shape) if torch.is_tensor(v) else v for k, v in output.items() if k != "loss_dict"})
