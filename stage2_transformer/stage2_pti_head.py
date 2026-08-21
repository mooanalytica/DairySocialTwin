from __future__ import annotations

import itertools

import torch
import torch.nn as nn


PTI_FEATURE_SCHEMA_VERSION = "pti_pair_keypoints_v1"


class PTIHead(nn.Module):
    """Pair-Temporal Interaction Transformer Head."""

    def __init__(
        self,
        num_joints: int = 27,
        max_persons: int = 2,
        max_frames: int = 64,
        d_model: int = 64,
        num_heads: int = 4,
        temporal_layers: int = 2,
        pair_layers: int = 1,
        ffn_dim: int = 128,
        dropout: float = 0.1,
        causal: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_joints = int(num_joints)
        self.max_persons = int(max_persons)
        self.max_frames = int(max_frames)
        self.d_model = int(d_model)
        self.causal = bool(causal)
        self.eps = float(eps)

        pair_feat_dim = 10 * self.num_joints + 8
        self.input_mlp = nn.Sequential(
            nn.LayerNorm(pair_feat_dim),
            nn.Linear(pair_feat_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.d_model),
        )
        self.time_pos = nn.Parameter(torch.zeros(1, self.max_frames, self.d_model))

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=temporal_layers)

        self.global_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        pair_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.pair_encoder = nn.TransformerEncoder(pair_layer, num_layers=pair_layers)

        self.pair_head = nn.Linear(self.d_model, 1)
        self.global_head = nn.Linear(self.d_model, 1)
        self.fusion = nn.Linear(2, 1)
        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.trunc_normal_(self.time_pos, std=0.02)
        nn.init.trunc_normal_(self.global_token, std=0.02)

    def _make_pairs(self, persons: int, device: torch.device) -> torch.Tensor:
        pairs = list(itertools.combinations(range(persons), 2))
        if not pairs:
            return torch.empty(0, 2, dtype=torch.long, device=device)
        return torch.tensor(pairs, dtype=torch.long, device=device)

    def _causal_mask(self, frames: int, device: torch.device) -> torch.Tensor | None:
        if not self.causal:
            return None
        mask = torch.full((frames, frames), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def forward(self, keypoints: torch.Tensor, person_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        batch, frames, persons, joints, channels = keypoints.shape
        device = keypoints.device
        dtype = keypoints.dtype
        eps = self.eps

        if joints != self.num_joints:
            raise ValueError(f"Expected num_joints={self.num_joints}, got J={joints}")
        if persons > self.max_persons:
            raise ValueError(f"P={persons} exceeds max_persons={self.max_persons}")
        if frames > self.max_frames:
            raise ValueError(f"T={frames} exceeds max_frames={self.max_frames}")
        if channels < 2:
            raise ValueError(f"Expected keypoint channels C>=2, got C={channels}")

        if person_mask is None:
            person_mask = torch.ones(batch, frames, persons, dtype=torch.bool, device=device)
        else:
            person_mask = person_mask.to(device=device, dtype=torch.bool)

        pairs = self._make_pairs(persons, device=device)
        pair_count = int(pairs.shape[0])
        if pair_count == 0:
            logit = torch.full((batch,), -20.0, dtype=dtype, device=device)
            return {
                "prob": torch.sigmoid(logit),
                "logit": logit,
                "pair_prob": torch.zeros(batch, 0, dtype=dtype, device=device),
                "pair_logits": torch.zeros(batch, 0, dtype=dtype, device=device),
                "pairs": pairs,
            }

        xy = torch.nan_to_num(keypoints[..., :2], nan=0.0, posinf=0.0, neginf=0.0)
        if channels >= 3:
            conf = torch.nan_to_num(keypoints[..., 2:3], nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
        else:
            conf = torch.ones(batch, frames, persons, joints, 1, dtype=dtype, device=device)
        conf = conf * person_mask[:, :, :, None, None].to(dtype=dtype)

        conf_sum = conf.sum(dim=3).clamp_min(eps)
        center = (xy * conf).sum(dim=3) / conf_sum
        centered_xy = xy - center.unsqueeze(3)
        scale = torch.sqrt(
            ((centered_xy.square().sum(dim=-1, keepdim=True) * conf).sum(dim=3) / conf_sum) + eps
        )

        vxy = torch.zeros_like(xy)
        vxy[:, 1:] = xy[:, 1:] - xy[:, :-1]
        vcenter = torch.zeros_like(center)
        vcenter[:, 1:] = center[:, 1:] - center[:, :-1]

        i_idx = pairs[:, 0]
        j_idx = pairs[:, 1]
        xy_i = xy[:, :, i_idx]
        xy_j = xy[:, :, j_idx]
        conf_i = conf[:, :, i_idx]
        conf_j = conf[:, :, j_idx]
        vxy_i = vxy[:, :, i_idx]
        vxy_j = vxy[:, :, j_idx]

        center_i = center[:, :, i_idx]
        center_j = center[:, :, j_idx]
        vcenter_i = vcenter[:, :, i_idx]
        vcenter_j = vcenter[:, :, j_idx]
        scale_i = scale[:, :, i_idx]
        scale_j = scale[:, :, j_idx]

        pair_center = 0.5 * (center_i + center_j)
        center_delta = center_j - center_i
        center_dist = torch.norm(center_delta, dim=-1, keepdim=True)
        pair_scale = torch.maximum(torch.maximum(scale_i, scale_j), center_dist).clamp_min(eps)

        joint_feat = torch.cat(
            [
                (xy_i - pair_center.unsqueeze(3)) / pair_scale.unsqueeze(3),
                conf_i,
                vxy_i / pair_scale.unsqueeze(3),
                (xy_j - pair_center.unsqueeze(3)) / pair_scale.unsqueeze(3),
                conf_j,
                vxy_j / pair_scale.unsqueeze(3),
            ],
            dim=-1,
        ).flatten(start_dim=3)

        rel_center = center_delta / pair_scale
        rel_vel = (vcenter_j - vcenter_i) / pair_scale
        norm_dist = torch.norm(rel_center, dim=-1, keepdim=True)
        approaching_speed = -torch.sum(rel_center * rel_vel, dim=-1, keepdim=True) / (norm_dist + eps)
        global_pair_feat = torch.cat(
            [rel_center, rel_vel, norm_dist, approaching_speed, conf_i.mean(dim=3), conf_j.mean(dim=3)],
            dim=-1,
        )

        pair_feat = torch.cat([joint_feat, global_pair_feat], dim=-1)
        mask_i = person_mask[:, :, i_idx]
        mask_j = person_mask[:, :, j_idx]
        pair_frame_mask = mask_i & mask_j

        pair_feat = pair_feat.permute(0, 2, 1, 3).contiguous().view(batch * pair_count, frames, -1)
        temporal_valid = pair_frame_mask.permute(0, 2, 1).contiguous().view(batch * pair_count, frames)
        temporal_padding = ~temporal_valid
        all_padding = temporal_padding.all(dim=1)
        if all_padding.any():
            temporal_padding[all_padding, 0] = False

        x = self.input_mlp(pair_feat) + self.time_pos[:, :frames]
        x = self.temporal_encoder(
            x,
            mask=self._causal_mask(frames, device=device),
            src_key_padding_mask=temporal_padding,
        )
        valid_float = (~temporal_padding).to(dtype=x.dtype).unsqueeze(-1)
        pair_token = (x * valid_float).sum(dim=1) / valid_float.sum(dim=1).clamp_min(1.0)
        pair_token = pair_token.view(batch, pair_count, self.d_model)

        pair_clip_mask = pair_frame_mask.any(dim=1)
        pair_tokens = torch.cat([self.global_token.expand(batch, 1, self.d_model), pair_token], dim=1)
        pair_padding = torch.cat(
            [torch.zeros(batch, 1, dtype=torch.bool, device=device), ~pair_clip_mask],
            dim=1,
        )
        pair_tokens = self.pair_encoder(pair_tokens, src_key_padding_mask=pair_padding)
        global_logit = self.global_head(pair_tokens[:, 0]).squeeze(-1)
        pair_logits = self.pair_head(pair_tokens[:, 1:]).squeeze(-1).masked_fill(~pair_clip_mask, -20.0)
        pair_prob = torch.sigmoid(pair_logits).masked_fill(~pair_clip_mask, 0.0)

        p_any = 1.0 - torch.prod(1.0 - pair_prob.clamp(0.0, 1.0 - eps), dim=1)
        noisy_or_logit = torch.logit(p_any.clamp(eps, 1.0 - eps))
        final_logit = self.fusion(torch.stack([global_logit, noisy_or_logit], dim=-1)).squeeze(-1)
        return {
            "prob": torch.sigmoid(final_logit),
            "logit": final_logit,
            "pair_prob": pair_prob,
            "pair_logits": pair_logits,
            "pairs": pairs,
        }


def predict_interaction(
    model: PTIHead,
    keypoints: torch.Tensor,
    person_mask: torch.Tensor | None = None,
    threshold: float = 0.5,
) -> dict[str, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        out = model(keypoints, person_mask)
    return {
        "interaction_prob": out["prob"],
        "interaction_decision": out["prob"] >= float(threshold),
        "pair_prob": out["pair_prob"],
        "pairs": out["pairs"],
    }
