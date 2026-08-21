from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import time
import traceback
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


def _disable_user_site_packages() -> None:
    """Keep user-site packages from shadowing the active conda environment."""
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    try:
        import site

        candidates = set()
        user_site = getattr(site, "USER_SITE", None)
        if user_site:
            candidates.add(str(Path(user_site).resolve()))
        try:
            candidates.add(str(Path(site.getusersitepackages()).resolve()))
        except Exception:
            pass

        filtered = []
        for entry in sys.path:
            if not entry:
                filtered.append(entry)
                continue
            try:
                resolved = str(Path(entry).resolve())
            except Exception:
                resolved = entry
            if resolved in candidates:
                continue
            if f"{os.sep}.local{os.sep}lib{os.sep}python" in resolved:
                continue
            filtered.append(entry)
        sys.path[:] = filtered
    except Exception:
        pass


_disable_user_site_packages()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from stage2_pair_features import (
    FEATURE_SCHEMA_VERSION,
    encode_pair_frame,
    expected_in_features as valence_expected_in_features,
    swap_pair_sequence as swap_valence_pair_sequence,
)
from stage2_dynamic_threshold import coerce_dynamic_threshold_config, dynamic_valence_conf_threshold
from stage2_pti_head import PTIHead, PTI_FEATURE_SCHEMA_VERSION


try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


BASE = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = Path(r"D:\DairyCowSNA\output_4_categories")
DEFAULT_OUTPUT_ROOT = BASE / "output_s2"
DEFAULT_INTERACTION_GATE_CKPT = BASE / "models_new" / "stage2_interaction_gate_best.pt"
DEFAULT_VALENCE_CKPT = BASE / "models_new" / "stage2_valence_transformer_best.pt"
DEFAULT_LOG_DIR = BASE / "run_logs"

RUN_TIMESTAMP_FORMAT = "%Y-%m-%d-%H:%M:%S"
VIDEO_EXTS = {".mp4"}

# Interaction existence and event tuning for the default two-stage cascade.
INTERACT_MIN_SEC = 0.5
INTERACT_DIAG_SIM_RATIO = 0.75
TIME_WINDOW_SEC = 30.0
COOLDOWN_SEC = 30.0
CONF_MASK_THR = 0.20

MI_CROPS_SEC: list[float | str] = [1, 2, 4, "full"]
RESAMPLE_TO_MODEL_FPS = True
BIDIRECTIONAL_PAIR_INFERENCE = True

PROX_HARD_CAP = 0.55
Q_PROX = 0.15
Q_WINDOWS_SEC = [1, 2, 4]

GATE_STABLE_M = 12
GATE_STABLE_K = 8
INTERACTION_GATE_STABLE_M = 12
INTERACTION_GATE_STABLE_K = 8
VALENCE_VOTE_N = 9
VALENCE_MIN_CONF = 0.60
VALENCE_MIN_START_VOTES = 2

LOG_PROX_DISTS = True
PROX_LOG_EVERY_SEC = 10
CLASSIFY_EVERY_HZ = 6
BUF_SEC_MIN = 5.0
GAP_TOL_SEC = 0.5
MAX_SUSPECT_PAIRS = 12

FRIENDLY_VALENCE = "friendly"
UNFRIENDLY_VALENCE = "unfriendly"
NO_INTERACTION_LABEL = "no_interaction"
NEUTRAL_VALENCE = "neutral"
POSITIVE_INTERACTIONS = {"friendly", "licking", "grooming", "allogrooming"}
NEGATIVE_INTERACTIONS = {"unfriendly", "displacement", "headbutting"}

DATA_SOURCE = "mooanalytica.com & Agnovix.com"
FORMAT_TIER = "lrv_proxy_mp4"
GOPRO_ID = 1
GOPRO_ID_NOTE = "one gopro each position"


class TeeLogger:
    def __init__(self, path: Path | None):
        self.path = path
        self.fh = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.fh = path.open("a", encoding="utf-8", buffering=1)

    def close(self) -> None:
        if self.fh is not None:
            self.fh.flush()
            self.fh.close()
            self.fh = None

    def log(self, message: str) -> None:
        print(message, flush=True)
        if self.fh is not None:
            self.fh.write(message + "\n")
            self.fh.flush()


LOGGER = TeeLogger(None)


def log(message: str) -> None:
    LOGGER.log(message)


class GlobalStage2Error(RuntimeError):
    pass


class FolderStage2Error(RuntimeError):
    pass


@dataclass
class Box:
    tid: int
    x: float
    y: float
    w: float
    h: float
    score: float = 0.0
    identity: str = ""
    id_conf: float = 0.0

    @property
    def area_scale(self) -> float:
        area = max(0.0, self.w) * max(0.0, self.h)
        return math.sqrt(area) if area > 0 else float("nan")

    @property
    def center(self) -> tuple[float, float]:
        return self.x + 0.5 * self.w, self.y + 0.5 * self.h


@dataclass
class LoadedModels:
    interaction_gate_model: nn.Module
    valence_model: nn.Module
    interaction_gate_model_type: str
    interaction_gate_thresh: float
    interaction_gate_max_frames: int
    valence_train_fps: float
    valence_max_frames: int
    required_keypoints: list[str]
    interaction_gate_id_to_label: dict[int, str]
    valence_id_to_label: dict[int, str]
    valence_num_classes: int
    valence_interaction_thresh: float = 0.0
    dynamic_valence_threshold: dict | None = None


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _resolve_nonexistent(path: Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _ensure_output_under_base(path: Path) -> Path:
    resolved = _resolve_nonexistent(path)
    base = BASE.resolve()
    if resolved == base or not _is_relative_to(resolved, base):
        raise RuntimeError(f"output root must be under project root {base}; got {resolved}")
    return resolved


def safe_clear_output_root(output_root: Path) -> Path:
    root = _ensure_output_under_base(output_root)
    root.mkdir(parents=True, exist_ok=True)
    for child in list(root.iterdir()):
        child_resolved = _resolve_nonexistent(child)
        if child_resolved == root or not _is_relative_to(child_resolved, root):
            raise RuntimeError(f"refusing to remove path outside output root: {child_resolved}")
        if child_resolved.is_dir() and not child_resolved.is_symlink():
            shutil.rmtree(child_resolved, onerror=_remove_readonly_and_retry)
        else:
            try:
                child_resolved.unlink()
            except PermissionError:
                os.chmod(child_resolved, 0o666)
                child_resolved.unlink()
    return root


def _remove_readonly_and_retry(func, path, exc_info):
    try:
        os.chmod(path, 0o666)
        func(path)
    except Exception:
        raise exc_info[1]


def safe_make_output_dir(output_root: Path, relative_dir: Path) -> Path:
    root = _ensure_output_under_base(output_root)
    target = _resolve_nonexistent(root / relative_dir)
    if target == root or not _is_relative_to(target, root):
        raise RuntimeError(f"refusing to create output path outside output root: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def csv_has_data_rows(path: Path) -> bool:
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            saw_header = False
            for row in reader:
                if not any(str(cell).strip() for cell in row):
                    continue
                if not saw_header:
                    saw_header = True
                    continue
                return True
    except OSError:
        return False
    return False


def normalize_legacy_path_string(value: str) -> str:
    s = str(value)
    replacements = [
        (
            r"G:\DROPBOX\Isolated_Interaction_Clips\Interaction_Type",
            r"F:\FULLDATA\Isolated_Interaction_Clips\Interaction_Type",
        ),
        (
            "G:/DROPBOX/Isolated_Interaction_Clips/Interaction_Type",
            "F:/FULLDATA/Isolated_Interaction_Clips/Interaction_Type",
        ),
        (r"E:\DROPBOX_LRV\Dairy Farm Videos", r"F:\FULLDATA\Dairy Farm Videos"),
        (r"E:\DROPBOX_LRV\Dairy Farm Video", r"F:\FULLDATA\Dairy Farm Videos"),
        ("E:/DROPBOX_LRV/Dairy Farm Videos", "F:/FULLDATA/Dairy Farm Videos"),
        ("E:/DROPBOX_LRV/Dairy Farm Video", "F:/FULLDATA/Dairy Farm Videos"),
    ]
    for old, new in replacements:
        if s.lower().startswith(old.lower()):
            return new + s[len(old) :]
    return s


def normalize_legacy_paths(obj):
    if isinstance(obj, dict):
        return {k: normalize_legacy_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize_legacy_paths(v) for v in obj]
    if isinstance(obj, str):
        return normalize_legacy_path_string(obj)
    return obj


def load_json_if_present(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_inter_label(lbl: str) -> tuple[str, str]:
    lab = str(lbl).strip().lower()
    if lab == "allogrooming":
        lab = "grooming"
    return lab, lab.capitalize()


def interaction_valence(lbl: str) -> str:
    lbl_norm, _ = normalize_inter_label(lbl)
    if lbl_norm in POSITIVE_INTERACTIONS:
        return FRIENDLY_VALENCE
    if lbl_norm in NEGATIVE_INTERACTIONS:
        return UNFRIENDLY_VALENCE
    return NEUTRAL_VALENCE


def _center_and_diag(box: Box) -> tuple[float, float, float]:
    cx = box.x + 0.5 * box.w
    cy = box.y + 0.5 * box.h
    return cx, cy, math.hypot(box.w, box.h)


def normalized_center_distance(a: Box, b: Box) -> tuple[float, float, float]:
    ax, ay, ad = _center_and_diag(a)
    bx, by, bd = _center_and_diag(b)
    denom = max(1e-6, 0.5 * (ad + bd))
    dist = math.hypot(ax - bx, ay - by) / denom
    return float(dist), float(ad), float(bd)


def boxes_overlap(a: Box, b: Box) -> tuple[bool, float]:
    ax1, ay1, ax2, ay2 = a.x, a.y, a.x + a.w, a.y + a.h
    bx1, by1, bx2, by2 = b.x, b.y, b.x + b.w, b.y + b.h
    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    overlap = (inter_w > 0.0) and (inter_h > 0.0)
    if not overlap:
        return False, 0.0
    inter_area = inter_w * inter_h
    area_a = max(0.0, a.w) * max(0.0, a.h)
    area_b = max(0.0, b.w) * max(0.0, b.h)
    iou = inter_area / max(1e-6, area_a + area_b - inter_area)
    return True, float(iou)


def _swap_pair_sequence(x_np: np.ndarray) -> np.ndarray:
    if x_np is None:
        return None
    if x_np.ndim != 2 or x_np.shape[1] < 2 or (x_np.shape[1] - 2) % 8 != 0:
        return np.array(x_np, copy=True)
    out = np.empty_like(x_np)
    num_kpts = (x_np.shape[1] - 2) // 8
    for j in range(num_kpts):
        c = 8 * j
        out[:, c : c + 6] = x_np[:, [c + 3, c + 4, c + 5, c + 0, c + 1, c + 2]]
        out[:, c + 6] = -x_np[:, c + 6]
        out[:, c + 7] = -x_np[:, c + 7]
    out[:, -2] = x_np[:, -1]
    out[:, -1] = x_np[:, -2]
    return out


def _resample_feature_sequence(x_np: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    if x_np is None:
        return None
    if x_np.ndim != 2 or x_np.shape[0] <= 1:
        return np.array(x_np, copy=True)
    if (not np.isfinite(src_fps)) or (not np.isfinite(dst_fps)) or src_fps <= 0 or dst_fps <= 0:
        return np.array(x_np, copy=True)
    if abs(float(src_fps) - float(dst_fps)) < 1e-6:
        return np.array(x_np, copy=True)
    t_old = x_np.shape[0]
    t_new = max(2, int(round(t_old * float(dst_fps) / float(src_fps))))
    if t_new == t_old:
        return np.array(x_np, copy=True)
    old_grid = np.linspace(0.0, 1.0, t_old, dtype=np.float32)
    new_grid = np.linspace(0.0, 1.0, t_new, dtype=np.float32)
    out = np.empty((t_new, x_np.shape[1]), dtype=np.float32)
    for col in range(x_np.shape[1]):
        out[:, col] = np.interp(new_grid, old_grid, x_np[:, col]).astype(np.float32)
    return out


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float32)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.clip(ex.sum(), 1e-9, None)


def masked_mean_pool_sequence(h: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    _b, t, _c = h.shape
    mask = torch.arange(t, device=h.device).unsqueeze(0) < lengths.unsqueeze(1)
    mask_f = mask.to(dtype=h.dtype).unsqueeze(-1)
    denom = mask_f.sum(dim=1).clamp(min=1.0)
    return (h * mask_f).sum(dim=1) / denom


class ValenceTransformer(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        max_frames: int = 64,
        d_model: int = 64,
        num_heads: int = 4,
        temporal_layers: int = 2,
        ffn_dim: int = 128,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        if int(max_frames) < 2:
            raise ValueError(f"max_frames must be >= 2, got {max_frames}")
        if int(d_model) % int(num_heads) != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        self.in_features = int(in_features)
        self.max_frames = int(max_frames)
        self.d_model = int(d_model)
        self.causal = bool(causal)
        self.input_norm = nn.LayerNorm(self.in_features)
        self.in_proj = nn.Linear(self.in_features, self.d_model)
        self.time_pos = nn.Parameter(torch.zeros(1, self.max_frames, self.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(num_heads),
            dim_feedforward=int(ffn_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(temporal_layers))
        self.out_norm = nn.LayerNorm(self.d_model)
        self.head = nn.Sequential(nn.Dropout(float(dropout)), nn.Linear(self.d_model, int(num_classes)))
        nn.init.trunc_normal_(self.time_pos, std=0.02)

    def _causal_mask(self, size: int, device: torch.device) -> torch.Tensor | None:
        if not self.causal:
            return None
        return torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected x shape [B,T,F], got {tuple(x.shape)}")
        _batch, frames, features = x.shape
        if int(features) != self.in_features:
            raise ValueError(f"expected {self.in_features} input features, got {features}")
        if int(frames) > self.max_frames:
            raise ValueError(f"sequence length {frames} exceeds max_frames={self.max_frames}")
        lengths = lengths.to(device=x.device, dtype=torch.long).clamp(min=1, max=max(1, int(frames)))
        padding_mask = torch.arange(frames, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        h = self.in_proj(self.input_norm(x))
        h = h + self.time_pos[:, :frames, :].to(device=x.device, dtype=h.dtype)
        h = self.encoder(h, mask=self._causal_mask(int(frames), x.device), src_key_padding_mask=padding_mask)
        h = self.out_norm(h)
        pooled = masked_mean_pool_sequence(h, lengths)
        return self.head(pooled)


def resolve_device(device_arg: str) -> torch.device:
    raw = str(device_arg or "cuda").strip().lower()
    if raw == "auto":
        raw = "cuda"
    if raw == "cpu":
        raise GlobalStage2Error("CUDA/GPU is required for this CSV interaction analysis script; refusing CPU mode.")
    if not torch.cuda.is_available():
        raise GlobalStage2Error(
            "CUDA/GPU is required for this CSV interaction analysis script, but torch.cuda.is_available() is false."
        )
    if raw in {"cuda", "cuda:0"}:
        return torch.device(raw)
    if re.fullmatch(r"cuda:\d+", raw):
        return torch.device(raw)
    raise GlobalStage2Error(f"unsupported device: {device_arg!r}; use cuda or cuda:N")


def profile_defaults(profile: str) -> dict[str, int]:
    selected = profile
    if selected == "auto":
        selected = "8gb"
        if torch.cuda.is_available():
            gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            selected = "40gb" if gb >= 20 else "8gb"
    if selected == "40gb":
        return {"inference_batch_size": 256}
    return {"inference_batch_size": 64}


def _load_ckpt(path: Path, device: torch.device) -> dict:
    if not path.is_file():
        raise GlobalStage2Error(f"missing checkpoint: {path}")
    return torch.load(str(path), map_location=device)


def load_models(args, device: torch.device) -> LoadedModels:
    interaction_gate_ckpt = _load_ckpt(Path(args.interaction_gate_ckpt), device)
    valence_ckpt = _load_ckpt(Path(args.valence_ckpt), device)

    valence_in_features = int(valence_ckpt.get("in_features", -1))
    if valence_in_features <= 0:
        raise GlobalStage2Error("Valence checkpoint missing in_features")

    interaction_gate_thresh = interaction_gate_ckpt.get("threshold", None)
    if interaction_gate_thresh is None:
        interaction_gate_thresh = interaction_gate_ckpt.get("interaction_threshold", None)
    if isinstance(interaction_gate_thresh, dict):
        interaction_gate_thresh = interaction_gate_thresh.get("thr", None)
    if isinstance(interaction_gate_thresh, torch.Tensor):
        interaction_gate_thresh = float(interaction_gate_thresh.detach().cpu().item())
    if interaction_gate_thresh is None:
        interaction_gate_thresh = 0.5
    interaction_gate_thresh = float(interaction_gate_thresh)

    interaction_gate_kps = list(interaction_gate_ckpt.get("keypoints", []))
    valence_kps = list(valence_ckpt.get("keypoints", []))
    required_kps = valence_kps or interaction_gate_kps
    if not required_kps:
        inferred = (valence_in_features - 2) // 8
        if inferred <= 0:
            raise GlobalStage2Error("checkpoints do not contain keypoints and in_features cannot infer them")
        required_kps = [f"kpt_{i}" for i in range(inferred)]

    valence_schema = str(valence_ckpt.get("feature_schema", "legacy_absolute_v0"))
    if str(valence_ckpt.get("model", "")) != "ValenceTransformer":
        raise GlobalStage2Error(
            f"Valence checkpoint must be ValenceTransformer; got model={valence_ckpt.get('model', '')!r}. "
            "Retrain the valence classifier with train_stage2_valence_inception.py --task valence_2class."
        )
    if valence_schema != FEATURE_SCHEMA_VERSION:
        raise GlobalStage2Error(
            f"Valence checkpoint feature_schema={valence_schema!r}; expected {FEATURE_SCHEMA_VERSION!r}. "
            "Retrain the valence classifier with train_stage2_valence_inception.py."
        )
    expected_valence_features = valence_expected_in_features(len(required_kps))
    if valence_in_features != expected_valence_features:
        raise GlobalStage2Error(
            f"Valence checkpoint in_features={valence_in_features}; expected {expected_valence_features} "
            f"for {len(required_kps)} keypoints and schema {FEATURE_SCHEMA_VERSION}."
        )

    interaction_gate_schema = str(interaction_gate_ckpt.get("feature_schema", "legacy_absolute_v0"))
    if (
        str(interaction_gate_ckpt.get("model", "")) != "PTIHead"
        and interaction_gate_schema != PTI_FEATURE_SCHEMA_VERSION
    ):
        raise GlobalStage2Error(
            f"Interaction gate checkpoint must be PTIHead/{PTI_FEATURE_SCHEMA_VERSION}; "
            f"got model={interaction_gate_ckpt.get('model', '')!r} feature_schema={interaction_gate_schema!r}. "
            "Retrain the gate with train_stage2_valence_inception.py --task interaction_gate_2class."
        )
    interaction_gate_model_type = "pti"
    raw_gate_id_to_label = interaction_gate_ckpt.get("id_to_label", {0: "no_interaction", 1: "interaction"})
    interaction_gate_id_to_label = {int(k): str(v) for k, v in raw_gate_id_to_label.items()}
    pti_cfg = dict(interaction_gate_ckpt.get("pti", {}))
    if not pti_cfg:
        pti_cfg = {
            "num_joints": int(interaction_gate_ckpt.get("num_joints", len(required_kps))),
            "max_persons": int(interaction_gate_ckpt.get("max_persons", 2)),
            "max_frames": int(interaction_gate_ckpt.get("max_frames", 64)),
            "d_model": 64,
            "num_heads": 4,
            "temporal_layers": 2,
            "pair_layers": 1,
            "ffn_dim": 128,
            "dropout": 0.1,
            "causal": False,
        }
    if int(pti_cfg.get("num_joints", -1)) != len(required_kps):
        raise GlobalStage2Error(
            f"PTI gate num_joints={pti_cfg.get('num_joints')}; expected {len(required_kps)} from valence keypoints."
        )
    interaction_gate_model = PTIHead(**pti_cfg).to(device)
    interaction_gate_max_frames = int(pti_cfg["max_frames"])
    interaction_gate_model.load_state_dict(interaction_gate_ckpt["state_dict"], strict=True)
    interaction_gate_model.eval()

    label_to_id = valence_ckpt.get("label_to_id", {"friendly": 0, "unfriendly": 1})
    valence_num_classes = len(label_to_id)
    transformer_cfg = dict(valence_ckpt.get("transformer", {}))
    if not transformer_cfg:
        raise GlobalStage2Error("ValenceTransformer checkpoint missing transformer config")
    transformer_cfg["in_features"] = int(transformer_cfg.get("in_features", valence_in_features))
    if int(transformer_cfg["in_features"]) != valence_in_features:
        raise GlobalStage2Error(
            f"ValenceTransformer in_features mismatch: transformer={transformer_cfg['in_features']} "
            f"top_level={valence_in_features}"
        )
    valence_max_frames = int(transformer_cfg.get("max_frames", 0))
    if valence_max_frames <= 0:
        raise GlobalStage2Error("ValenceTransformer checkpoint missing positive max_frames")
    valence_model = ValenceTransformer(num_classes=valence_num_classes, **transformer_cfg).to(device)
    valence_model.load_state_dict(valence_ckpt["state_dict"], strict=True)
    valence_model.eval()

    raw_id_to_label = valence_ckpt.get("id_to_label", {i: str(i) for i in range(valence_num_classes)})
    valence_id_to_label = {int(k): str(v) for k, v in raw_id_to_label.items()}
    valence_interaction_thresh = float(valence_ckpt.get("interaction_threshold", 0.0) or 0.0)
    dynamic_valence_threshold = coerce_dynamic_threshold_config(
        valence_ckpt.get("dynamic_valence_conf_threshold"),
        fallback_fixed=VALENCE_MIN_CONF,
        gate_threshold=interaction_gate_thresh,
    )

    log(f"[info] device: {torch.cuda.get_device_name(device)}")
    log(
        f"[info] Interaction gate loaded: threshold={interaction_gate_thresh:.6f} "
        f"train_fps={interaction_gate_ckpt.get('fps', 0)} feature_schema={interaction_gate_schema} "
        f"model_type={interaction_gate_model_type} classes={interaction_gate_id_to_label}"
    )
    log(
        f"[info] Valence classifier loaded: classes={valence_id_to_label} "
        f"train_fps={valence_ckpt.get('fps', 0)} feature_schema={valence_schema} "
        f"model=ValenceTransformer max_frames={valence_max_frames}"
    )
    if any(str(v).lower() == NO_INTERACTION_LABEL for v in valence_id_to_label.values()):
        log(f"[info] unified Stage2 no-interaction class enabled; interaction_threshold={valence_interaction_thresh:.3f}")
    log(
        "[info] Dynamic valence threshold: "
        f"enabled={bool(dynamic_valence_threshold.get('enabled', False))} "
        f"base={float(dynamic_valence_threshold['base_valence_conf']):.3f} "
        f"min={float(dynamic_valence_threshold['min_valence_conf']):.3f} "
        f"exp={float(dynamic_valence_threshold['gate_exponent']):.3f}"
    )
    gate_shape = f"pti_joints={len(required_kps)} max_frames={interaction_gate_max_frames}"
    log(f"[info] keypoints={len(required_kps)} {gate_shape} valence_features={valence_in_features}")

    return LoadedModels(
        interaction_gate_model=interaction_gate_model,
        valence_model=valence_model,
        interaction_gate_model_type=interaction_gate_model_type,
        interaction_gate_thresh=interaction_gate_thresh,
        interaction_gate_max_frames=interaction_gate_max_frames,
        valence_train_fps=float(valence_ckpt.get("fps", 0) or 0.0),
        valence_max_frames=valence_max_frames,
        required_keypoints=required_kps,
        interaction_gate_id_to_label=interaction_gate_id_to_label,
        valence_id_to_label=valence_id_to_label,
        valence_num_classes=valence_num_classes,
        valence_interaction_thresh=valence_interaction_thresh,
        dynamic_valence_threshold=dynamic_valence_threshold,
    )


def folder_sort_key(path: Path) -> tuple[int, str]:
    name = path.name
    lower = name.lower()
    for prefix in ("v_", "e_"):
        if lower.startswith(prefix):
            try:
                return int(name.split("_", 1)[1]), name
            except Exception:
                pass
    try:
        return int(name), name
    except Exception:
        return 10**9, str(path).lower()


def discover_input_folders(input_root: Path) -> list[Path]:
    root = Path(input_root)
    if not root.is_dir():
        raise FileNotFoundError(f"input root not found: {root}")
    folders: set[Path] = set()
    for name in ("tracking_boxes.csv", "keypoints.csv"):
        for path in root.rglob(name):
            if path.is_file():
                folders.add(path.parent)
    return sorted(folders, key=folder_sort_key)


def load_boxes(path: Path) -> tuple[dict[int, dict[int, Box]], int, str]:
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise FolderStage2Error(f"empty tracking CSV: {path}") from exc
    required = {"frame", "track_id", "x", "y", "w", "h"}
    missing = required.difference(df.columns)
    if missing:
        raise FolderStage2Error(f"{path} missing columns: {sorted(missing)}")
    if df.empty:
        raise FolderStage2Error(f"{path} has no data rows")
    df = df.dropna(subset=["frame", "track_id", "x", "y", "w", "h"])
    if df.empty:
        raise FolderStage2Error(f"{path} has no valid tracking rows after dropping NaN")

    by_frame: dict[int, dict[int, Box]] = defaultdict(dict)
    video_name = ""
    if "video" in df.columns and len(df["video"]) > 0:
        video_name = str(df["video"].iloc[0])
    for row in df.itertuples(index=False):
        frame = int(getattr(row, "frame"))
        tid = int(getattr(row, "track_id"))
        score = float(getattr(row, "score", 0.0)) if hasattr(row, "score") else 0.0
        identity = str(getattr(row, "identity", "")) if hasattr(row, "identity") else ""
        id_conf = float(getattr(row, "id_conf", 0.0)) if hasattr(row, "id_conf") else 0.0
        by_frame[frame][tid] = Box(
            tid=tid,
            x=float(getattr(row, "x")),
            y=float(getattr(row, "y")),
            w=float(getattr(row, "w")),
            h=float(getattr(row, "h")),
            score=score,
            identity=identity,
            id_conf=id_conf,
        )
    return dict(by_frame), int(len(df)), video_name


def load_keypoints(path: Path, num_kpts: int) -> tuple[dict[tuple[int, int], np.ndarray], int]:
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise FolderStage2Error(f"empty keypoints CSV: {path}") from exc
    required = {"frame", "track_id"}
    missing = required.difference(df.columns)
    if missing:
        raise FolderStage2Error(f"{path} missing columns: {sorted(missing)}")
    if df.empty:
        raise FolderStage2Error(f"{path} has no data rows")

    cols: list[str] = []
    for i in range(num_kpts):
        cols.extend([f"kpt_{i}_x", f"kpt_{i}_y", f"kpt_{i}_conf"])
    missing_kpts = [c for c in cols if c not in df.columns]
    if missing_kpts:
        raise FolderStage2Error(f"{path} missing keypoint columns, first missing: {missing_kpts[:6]}")

    df = df.dropna(subset=["frame", "track_id"])
    if df.empty:
        raise FolderStage2Error(f"{path} has no valid keypoint rows after dropping NaN")

    values = df[cols].to_numpy(dtype=np.float32, copy=True).reshape((-1, num_kpts, 3))
    frames = df["frame"].to_numpy(dtype=np.int64)
    tids = df["track_id"].to_numpy(dtype=np.int64)
    out: dict[tuple[int, int], np.ndarray] = {}
    for i in range(len(df)):
        out[(int(frames[i]), int(tids[i]))] = values[i]
    return out, int(len(df))


def _make_pair_buf(buf_len: int):
    return {
        "frames": deque(maxlen=buf_len),
        "kptsA": deque(maxlen=buf_len),
        "kptsB": deque(maxlen=buf_len),
        "centerA": deque(maxlen=buf_len),
        "centerB": deque(maxlen=buf_len),
        "scaleA": deque(maxlen=buf_len),
        "scaleB": deque(maxlen=buf_len),
    }


def _make_pair_evt():
    return {
        "active": False,
        "start_f": None,
        "cur_label": None,
        "cur_stage2_label": None,
        "last_pos_frame": -1,
        "conf_stage1_max": 0.0,
        "conf_stage2_max": 0.0,
        "conf_valence_max": 0.0,
        "valence_thresh_min": 1.0,
        "conf_friendly_max": 0.0,
        "conf_unfriendly_max": 0.0,
        "gate_hist": deque(maxlen=GATE_STABLE_M),
        "stage1_hist": deque(maxlen=INTERACTION_GATE_STABLE_M),
        "stage2_votes": deque(maxlen=VALENCE_VOTE_N),
        "last_pred_frame": -1,
    }


def _make_pair_noevt():
    return {"active": False, "start_f": None, "last_frame": -1, "p1_max": 0.0}


def reset_stage2_event(evt: dict) -> None:
    evt["active"] = False
    evt["start_f"] = None
    evt["cur_label"] = None
    evt["cur_stage2_label"] = None
    evt["conf_stage2_max"] = 0.0
    evt["conf_valence_max"] = 0.0
    evt["valence_thresh_min"] = 1.0
    evt["conf_friendly_max"] = 0.0
    evt["conf_unfriendly_max"] = 0.0
    evt["stage2_votes"].clear()


def reset_pair_after_gap(evt: dict) -> None:
    reset_stage2_event(evt)
    evt["conf_stage1_max"] = 0.0
    evt["stage1_hist"].clear()
    evt["last_pos_frame"] = -1
    evt["last_pred_frame"] = -1


def valence_seq_from_pairbuf(stbuf: dict, num_kpts: int) -> np.ndarray | None:
    k_a_list = stbuf["kptsA"]
    k_b_list = stbuf["kptsB"]
    c_a_list = stbuf["centerA"]
    c_b_list = stbuf["centerB"]
    s_a_list = stbuf["scaleA"]
    s_b_list = stbuf["scaleB"]
    t_len = len(k_a_list)
    if t_len == 0:
        return None
    fdim = valence_expected_in_features(num_kpts)
    x = np.zeros((t_len, fdim), dtype=np.float32)

    for t in range(t_len):
        x[t, :] = encode_pair_frame(
            k_a_list[t],
            k_b_list[t],
            c_a_list[t] if t < len(c_a_list) else None,
            c_b_list[t] if t < len(c_b_list) else None,
            s_a_list[t] if t < len(s_a_list) else None,
            s_b_list[t] if t < len(s_b_list) else None,
            num_kpts,
            conf_thr=CONF_MASK_THR,
        )

    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _clean_pti_person(kpts: np.ndarray | None, num_kpts: int) -> tuple[np.ndarray, bool]:
    out = np.zeros((num_kpts, 3), dtype=np.float32)
    if kpts is None:
        return out, False
    try:
        arr = np.asarray(kpts, dtype=np.float32)
    except Exception:
        return out, False
    if arr.ndim != 2 or arr.shape[0] < num_kpts or arr.shape[1] < 2:
        return out, False
    cols = min(3, arr.shape[1])
    out[:, :cols] = arr[:num_kpts, :cols]
    if cols < 3:
        out[:, 2] = 1.0
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out, bool(np.isfinite(arr[:num_kpts, :2]).all())


def pti_seq_from_pairbuf(stbuf: dict, num_kpts: int) -> tuple[np.ndarray, np.ndarray] | None:
    k_a_list = stbuf["kptsA"]
    k_b_list = stbuf["kptsB"]
    t_len = len(k_a_list)
    if t_len == 0:
        return None
    keypoints = np.zeros((t_len, 2, num_kpts, 3), dtype=np.float32)
    person_mask = np.zeros((t_len, 2), dtype=np.bool_)
    for t in range(t_len):
        keypoints[t, 0], person_mask[t, 0] = _clean_pti_person(k_a_list[t], num_kpts)
        keypoints[t, 1], person_mask[t, 1] = _clean_pti_person(k_b_list[t], num_kpts)
    return keypoints, person_mask


def crop_pti_sequence(
    keypoints: np.ndarray,
    person_mask: np.ndarray,
    fps: float,
    crop: float | str,
    max_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    start = 0
    end = keypoints.shape[0]
    if crop != "full":
        frames = max(2, int(round(float(crop) * fps)))
        if end > frames:
            start = end - frames
            end = start + frames
    keypoints = keypoints[start:end]
    person_mask = person_mask[start:end]
    if max_frames > 0 and keypoints.shape[0] > max_frames:
        keypoints = keypoints[-max_frames:]
        person_mask = person_mask[-max_frames:]
    return keypoints, person_mask


def swap_pti_pair_sequence(keypoints: np.ndarray, person_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.ascontiguousarray(keypoints[:, [1, 0], :, :]), np.ascontiguousarray(person_mask[:, [1, 0]])


def prepare_seq_for_model(x_np: np.ndarray, fps: float, target_fps: float, max_frames: int) -> np.ndarray:
    if x_np is None:
        return None
    if RESAMPLE_TO_MODEL_FPS and target_fps > 0:
        out = _resample_feature_sequence(x_np, fps, target_fps)
    else:
        out = np.array(x_np, copy=True)
    if max_frames > 0 and out.shape[0] > max_frames:
        out = out[-max_frames:]
    return np.ascontiguousarray(out, dtype=np.float32)


def pad_batch_np(list_x: list[np.ndarray], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = np.array([x.shape[0] for x in list_x], dtype=np.int64)
    tmax = int(lengths.max()) if len(lengths) else 0
    fdim = int(list_x[0].shape[1]) if len(list_x) else 0
    xpad = np.zeros((len(list_x), tmax, fdim), dtype=np.float32)
    for i, x in enumerate(list_x):
        xpad[i, : x.shape[0], :] = x
    x_t = torch.from_numpy(xpad).to(device=device, dtype=torch.float32)
    lengths_t = torch.from_numpy(lengths).to(device=device, dtype=torch.long)
    return x_t, lengths_t


@torch.no_grad()
def pti_gate_batch_probs(
    model: nn.Module,
    list_items: list[tuple[np.ndarray, np.ndarray]],
    device: torch.device,
    batch_size: int,
    amp: bool,
) -> np.ndarray:
    outs: list[np.ndarray] = []
    for start in range(0, len(list_items), batch_size):
        chunk = list_items[start : start + batch_size]
        lengths = [item[0].shape[0] for item in chunk]
        max_len = int(max(lengths)) if lengths else 0
        persons = int(chunk[0][0].shape[1]) if chunk else 0
        joints = int(chunk[0][0].shape[2]) if chunk else 0
        channels = int(chunk[0][0].shape[3]) if chunk else 0
        keypoints = np.zeros((len(chunk), max_len, persons, joints, channels), dtype=np.float32)
        person_mask = np.zeros((len(chunk), max_len, persons), dtype=np.bool_)
        for i, (seq, mask) in enumerate(chunk):
            keypoints[i, : seq.shape[0]] = seq
            person_mask[i, : mask.shape[0]] = mask
        k_t = torch.from_numpy(keypoints).to(device=device, dtype=torch.float32)
        m_t = torch.from_numpy(person_mask).to(device=device, dtype=torch.bool)
        with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda"), dtype=torch.float16):
            probs = model(k_t, m_t)["prob"]
        outs.append(probs.detach().float().cpu().numpy())
    return np.concatenate(outs, axis=0) if outs else np.zeros((0,), dtype=np.float32)


@torch.no_grad()
def valence_batch_logits(
    model: nn.Module,
    list_x: list[np.ndarray],
    device: torch.device,
    batch_size: int,
    amp: bool,
) -> np.ndarray:
    outs: list[np.ndarray] = []
    for start in range(0, len(list_x), batch_size):
        chunk = list_x[start : start + batch_size]
        x_t, lengths_t = pad_batch_np(chunk, device)
        with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda"), dtype=torch.float16):
            logits = model(x_t, lengths_t)
        outs.append(logits.detach().float().cpu().numpy())
    if not outs:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(outs, axis=0)


def decode_valence_logits(logits: np.ndarray) -> tuple[int, float, np.ndarray]:
    probs = _softmax_np(logits)
    cid = int(np.argmax(probs))
    conf = float(probs[cid])
    return cid, conf, probs


def decode_valence_scores(probs: np.ndarray, id_to_label: dict[int, str]) -> tuple[str, float, float, float]:
    friendly_score = 0.0
    unfriendly_score = 0.0
    for cls_id, prob in enumerate(np.asarray(probs, dtype=np.float32).tolist()):
        label = id_to_label.get(int(cls_id), str(cls_id))
        valence = interaction_valence(label)
        if valence == FRIENDLY_VALENCE:
            friendly_score += float(prob)
        elif valence == UNFRIENDLY_VALENCE:
            unfriendly_score += float(prob)
    if friendly_score >= unfriendly_score:
        return FRIENDLY_VALENCE, float(friendly_score), float(friendly_score), float(unfriendly_score)
    return UNFRIENDLY_VALENCE, float(unfriendly_score), float(friendly_score), float(unfriendly_score)


def vote_mode(votes: deque) -> str | None:
    if not votes:
        return None
    return Counter(list(votes)).most_common(1)[0][0]


def output_csvs(output_dir: Path, video_name: str, inter_rows: list[list], no_inter_rows: list[list], prox_logs: list[list]) -> dict:
    if inter_rows:
        inter_df = pd.DataFrame(
            inter_rows,
            columns=[
                "video",
                "class",
                "tidA",
                "tidB",
                "start_frame",
                "end_frame",
                "duration_s",
                "weight",
                "valence_score",
                "valence_thresh_min",
                "friendly_score",
                "unfriendly_score",
                "stage2_class",
                "stage2_conf",
                "stage1_prob",
                "stage1_thresh",
            ],
        )
        inter_df.to_csv(output_dir / "interactions.csv", index=False)
        for cls in sorted([c for c in inter_df["class"].unique() if str(c).lower() != "neutral"]):
            sub = inter_df[inter_df["class"] == cls]
            adj = (
                sub.groupby(["tidA", "tidB"])["weight"]
                .sum()
                .reset_index()
                .sort_values(["weight"], ascending=False)
            )
            adj.to_csv(output_dir / f"adjacency_{cls}.csv", index=False)

    if no_inter_rows:
        no_df = pd.DataFrame(
            no_inter_rows,
            columns=["video", "tidA", "tidB", "start_frame", "end_frame", "duration_s", "stage1_prob_max"],
        )
        no_df.to_csv(output_dir / "no_interactions.csv", index=False)

    if prox_logs:
        prox_df = pd.DataFrame(
            prox_logs,
            columns=["video", "frame", "t_sec", "p1", "p5", "p10", "p15", "p25", "p50", "p75", "p90", "p95", "p99"],
        )
        prox_df.to_csv(output_dir / "proximity_percentiles.csv", index=False)

    adjacency_classes = sorted(
        p.stem[len("adjacency_") :]
        for p in output_dir.glob("adjacency_*.csv")
        if p.is_file() and csv_has_data_rows(p)
    )
    return {
        "has_interactions": int(csv_has_data_rows(output_dir / "interactions.csv")),
        "has_no_interactions": int(csv_has_data_rows(output_dir / "no_interactions.csv")),
        "has_adjacency": int(bool(adjacency_classes)),
        "adjacency_classes": adjacency_classes,
        "has_proximity_percentiles": int(csv_has_data_rows(output_dir / "proximity_percentiles.csv")),
        "video_name": video_name,
    }


def copy_visualization_mp4s(source_dir: Path, output_dir: Path) -> list[str]:
    copied = []
    for path in sorted(source_dir.iterdir(), key=lambda p: p.name.lower()):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
            dest = output_dir / path.name
            shutil.copy2(path, dest)
            try:
                os.chmod(dest, 0o666)
            except OSError:
                pass
            copied.append(path.name)
    return copied


def process_folder(
    source_dir: Path,
    output_dir: Path,
    models: LoadedModels,
    device: torch.device,
    args,
) -> dict:
    track_csv = source_dir / "tracking_boxes.csv"
    kp_csv = source_dir / "keypoints.csv"
    if not track_csv.is_file() or not kp_csv.is_file():
        raise FolderStage2Error(f"missing required CSVs in {source_dir}")

    manifest = normalize_legacy_paths(load_json_if_present(source_dir / "manifest.json"))
    video_manifest = dict(manifest.get("video_manifest", {})) if manifest else {}
    fps = float(video_manifest.get("fps", 0.0) or args.fps)
    if fps <= 0:
        fps = float(args.fps)
    video_name = str(video_manifest.get("relative_path", "")).replace("/", "\\").split("\\")[-1]

    boxes_by_frame, track_rows, track_video_name = load_boxes(track_csv)
    num_kpts = len(models.required_keypoints)
    kpts_by_key, kpt_rows = load_keypoints(kp_csv, num_kpts)
    if track_video_name:
        video_name = track_video_name
    if not video_name:
        video_name = source_dir.name

    copied_videos = copy_visualization_mp4s(source_dir, output_dir)

    all_frames = sorted(boxes_by_frame)
    if not all_frames:
        raise FolderStage2Error("no tracking frames available")
    max_frame = max(all_frames)
    manifest_frames = int(video_manifest.get("frame_count", 0) or 0)
    frame_total = max(max_frame + 1, manifest_frames)

    buf_sec = max(INTERACT_MIN_SEC, BUF_SEC_MIN)
    buf_len = int(math.ceil(buf_sec * fps))
    min_frames = int(math.ceil(INTERACT_MIN_SEC * fps))
    gap_tol_fr = int(round(GAP_TOL_SEC * fps))
    classify_every = max(1, int(round(fps / CLASSIFY_EVERY_HZ)))
    cooldown_frames = int(round(COOLDOWN_SEC * fps))

    pair_geom = defaultdict(lambda: {"suspect_frames": 0, "miss": 0, "prox": None, "last_ok_frame": -1})
    pair_prox_hist = defaultdict(lambda: {w: deque(maxlen=max(1, int(w * fps))) for w in Q_WINDOWS_SEC})
    pair_buf = defaultdict(lambda: _make_pair_buf(buf_len))
    pair_evt = defaultdict(_make_pair_evt)
    pair_noevt = defaultdict(_make_pair_noevt)
    pair_cooldown_until: dict[tuple[int, int], int] = {}

    inter_rows: list[list] = []
    no_inter_rows: list[list] = []
    prox_logs: list[list] = []

    def finalize_and_write(evt: dict, key: tuple[int, int], end_f: int) -> None:
        a_id, b_id = key
        if evt["start_f"] is None:
            return
        dur_frames = int(end_f - evt["start_f"] + 1)
        if dur_frames < min_frames:
            return
        weight = max(1, int(math.ceil((dur_frames / fps) / TIME_WINDOW_SEC)))
        inter_rows.append(
            [
                video_name,
                evt["cur_label"],
                int(a_id),
                int(b_id),
                int(evt["start_f"]),
                int(end_f),
                float(dur_frames / fps),
                int(weight),
                float(evt.get("conf_valence_max", 0.0)),
                float(evt.get("valence_thresh_min", 1.0)),
                float(evt.get("conf_friendly_max", 0.0)),
                float(evt.get("conf_unfriendly_max", 0.0)),
                evt.get("cur_stage2_label") or "",
                float(evt.get("conf_stage2_max", 0.0)),
                float(evt.get("conf_stage1_max", 0.0)),
                float(models.interaction_gate_thresh),
            ]
        )

    pbar = tqdm(
        range(frame_total),
        desc=f"{source_dir.name}",
        unit="frame",
        leave=False,
        ascii=True,
        dynamic_ncols=False,
        ncols=100,
        mininterval=1.0,
        file=sys.stdout,
    )

    for frame_idx in pbar:
        frame_boxes = boxes_by_frame.get(frame_idx, {})
        tracks = [frame_boxes[tid] for tid in sorted(frame_boxes)]
        all_pairs: list[tuple[float, tuple[int, int]]] = []
        prox_log_bucket: list[float] = []

        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                a = tracks[i]
                b = tracks[j]
                overlap, _iou = boxes_overlap(a, b)
                center_dist_norm, d1, d2 = normalized_center_distance(a, b)
                sim = min(d1, d2) / max(d1, d2) if max(d1, d2) > 0 else 0.0
                diag_ok = sim >= INTERACT_DIAG_SIM_RATIO
                key = tuple(sorted((int(a.tid), int(b.tid))))

                prox_hist = pair_prox_hist[key]
                evt = pair_evt[key]
                hard_prox_ok = center_dist_norm <= PROX_HARD_CAP
                quantile_ok = False

                if overlap and diag_ok:
                    for w in Q_WINDOWS_SEC:
                        hist = prox_hist[w]
                        hist.append(center_dist_norm)
                        if len(hist) >= max(3, int(0.5 * w * fps)):
                            qv = float(np.quantile(np.asarray(hist, dtype=np.float32), Q_PROX))
                            if qv <= PROX_HARD_CAP:
                                quantile_ok = True
                    if LOG_PROX_DISTS and (frame_idx % max(1, int(PROX_LOG_EVERY_SEC * fps)) == 0):
                        prox_log_bucket.append(center_dist_norm)
                else:
                    for hist in prox_hist.values():
                        hist.clear()

                gate_pass = bool(overlap and diag_ok and (hard_prox_ok or quantile_ok))
                evt["gate_hist"].append(1 if gate_pass else 0)
                stable_gate = sum(evt["gate_hist"]) >= GATE_STABLE_K
                if stable_gate:
                    g = pair_geom[key]
                    g["suspect_frames"] += 1
                    g["miss"] = 0
                    g["last_ok_frame"] = frame_idx
                    g["prox"] = float(center_dist_norm)
                    all_pairs.append((float(center_dist_norm), key))

        if LOG_PROX_DISTS and (frame_idx % max(1, int(PROX_LOG_EVERY_SEC * fps)) == 0) and prox_log_bucket:
            pcts = [1, 5, 10, 15, 25, 50, 75, 90, 95, 99]
            vals = np.percentile(np.asarray(prox_log_bucket, dtype=np.float32), pcts).tolist()
            prox_logs.append([video_name, int(frame_idx), float(frame_idx / fps)] + [float(v) for v in vals])

        all_pairs.sort(key=lambda x: x[0])
        suspect_pairs = {p[1] for p in all_pairs[:MAX_SUSPECT_PAIRS]}

        for key, st in list(pair_geom.items()):
            if key not in suspect_pairs:
                st["miss"] += 1
                if st["miss"] > gap_tol_fr:
                    st["suspect_frames"] = 0
                    st["prox"] = None
                    pair_prox_hist.pop(key, None)

        for key in list(pair_buf.keys()):
            st = pair_geom.get(key)
            if (key not in suspect_pairs) and (st is None or st.get("miss", 0) > gap_tol_fr):
                pair_buf.pop(key, None)

        for a_id, b_id in suspect_pairs:
            box_a = frame_boxes.get(int(a_id))
            box_b = frame_boxes.get(int(b_id))
            k_a = kpts_by_key.get((frame_idx, int(a_id)))
            k_b = kpts_by_key.get((frame_idx, int(b_id)))
            st = pair_buf[(a_id, b_id)]
            st["frames"].append(frame_idx)
            st["kptsA"].append(k_a)
            st["kptsB"].append(k_b)
            st["centerA"].append(box_a.center if box_a is not None else None)
            st["centerB"].append(box_b.center if box_b is not None else None)
            st["scaleA"].append(box_a.area_scale if box_a is not None else float("nan"))
            st["scaleB"].append(box_b.area_scale if box_b is not None else float("nan"))

        if (frame_idx % classify_every) == 0:
            eligible_keys: list[tuple[int, int]] = []
            interaction_gate_pti_map: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
            valence_xfull_map: dict[tuple[int, int], np.ndarray] = {}
            for key in suspect_pairs:
                if key in pair_cooldown_until and frame_idx < pair_cooldown_until[key]:
                    continue
                stbuf = pair_buf[key]
                if len(stbuf["frames"]) < int(min(Q_WINDOWS_SEC) * fps):
                    continue
                valence_xfull = valence_seq_from_pairbuf(stbuf, num_kpts)
                interaction_gate_pti = pti_seq_from_pairbuf(stbuf, num_kpts)
                if interaction_gate_pti is None or valence_xfull is None or valence_xfull.shape[0] < 2:
                    continue
                if interaction_gate_pti[0].shape[0] < 2:
                    continue
                eligible_keys.append(key)
                interaction_gate_pti_map[key] = interaction_gate_pti
                valence_xfull_map[key] = valence_xfull

            if eligible_keys:
                p1_max = {k: 0.0 for k in eligible_keys}
                for w in MI_CROPS_SEC:
                    keys_w: list[tuple[int, int]] = []
                    list_pti: list[tuple[np.ndarray, np.ndarray]] = []
                    for key in eligible_keys:
                        kfull, mfull = interaction_gate_pti_map[key]
                        kseq, mseq = crop_pti_sequence(kfull, mfull, fps, w, models.interaction_gate_max_frames)
                        list_pti.append((kseq, mseq))
                        keys_w.append(key)
                        if BIDIRECTIONAL_PAIR_INFERENCE:
                            list_pti.append(swap_pti_pair_sequence(kseq, mseq))
                            keys_w.append(key)
                    probs = pti_gate_batch_probs(
                        models.interaction_gate_model,
                        list_pti,
                        device=device,
                        batch_size=args.inference_batch_size,
                        amp=not args.no_amp,
                    )
                    for key, prob in zip(keys_w, probs):
                        if prob > p1_max[key]:
                            p1_max[key] = float(prob)

                stable_interaction_gate_keys: list[tuple[int, int]] = []
                for key in eligible_keys:
                    evt = pair_evt[key]
                    p1 = float(p1_max[key])
                    hit1 = p1 >= models.interaction_gate_thresh
                    evt["conf_stage1_max"] = max(evt.get("conf_stage1_max", 0.0), p1)
                    evt["stage1_hist"].append(1 if hit1 else 0)
                    stable_interaction_gate = sum(evt["stage1_hist"]) >= INTERACTION_GATE_STABLE_K

                    noevt = pair_noevt[key]
                    if not stable_interaction_gate:
                        if not noevt["active"]:
                            noevt["active"] = True
                            noevt["start_f"] = frame_idx
                            noevt["p1_max"] = p1
                        else:
                            noevt["p1_max"] = max(float(noevt.get("p1_max", 0.0)), p1)
                        noevt["last_frame"] = frame_idx
                    else:
                        if noevt.get("active", False) and (noevt.get("start_f") is not None):
                            end_f0 = frame_idx - 1
                            dur0 = int(end_f0 - int(noevt["start_f"]) + 1)
                            if dur0 >= min_frames and float(noevt.get("p1_max", 0.0)) < float(models.interaction_gate_thresh):
                                no_inter_rows.append(
                                    [
                                        video_name,
                                        int(key[0]),
                                        int(key[1]),
                                        int(noevt["start_f"]),
                                        int(end_f0),
                                        float(dur0 / fps),
                                        float(noevt.get("p1_max", 0.0)),
                                    ]
                                )
                        noevt["active"] = False
                        noevt["start_f"] = None
                        noevt["last_frame"] = -1
                        noevt["p1_max"] = 0.0

                    if stable_interaction_gate:
                        evt["last_pos_frame"] = frame_idx
                        evt["last_pred_frame"] = frame_idx
                        stable_interaction_gate_keys.append(key)

                if stable_interaction_gate_keys:
                    logits_sum: dict[tuple[int, int], np.ndarray | None] = {k: None for k in stable_interaction_gate_keys}
                    for w in MI_CROPS_SEC:
                        list_x = []
                        keys_w: list[tuple[tuple[int, int], bool]] = []
                        for key in stable_interaction_gate_keys:
                            xfull = valence_xfull_map[key]
                            if w == "full":
                                x = xfull
                            else:
                                n = int(float(w) * fps)
                                x = xfull[-n:] if xfull.shape[0] > n else xfull
                            list_x.append(
                                prepare_seq_for_model(x, fps, models.valence_train_fps, models.valence_max_frames)
                            )
                            keys_w.append((key, False))
                            if BIDIRECTIONAL_PAIR_INFERENCE:
                                list_x.append(
                                    prepare_seq_for_model(
                                        swap_valence_pair_sequence(x),
                                        fps,
                                        models.valence_train_fps,
                                        models.valence_max_frames,
                                    )
                                )
                                keys_w.append((key, True))

                        logits_batch = valence_batch_logits(
                            models.valence_model,
                            list_x,
                            device=device,
                            batch_size=args.inference_batch_size,
                            amp=not args.no_amp,
                        )

                        best_logits: dict[tuple[int, int], np.ndarray] = {}
                        best_conf: dict[tuple[int, int], float] = {}
                        for (key, _swapped), logits in zip(keys_w, logits_batch):
                            _, conf, _ = decode_valence_logits(logits)
                            if (key not in best_conf) or (conf > best_conf[key]):
                                best_conf[key] = conf
                                best_logits[key] = logits.astype(np.float32)

                        for key, logits in best_logits.items():
                            logits_sum[key] = logits if logits_sum[key] is None else logits_sum[key] + logits

                    num_windows = float(len(MI_CROPS_SEC))
                    for key in stable_interaction_gate_keys:
                        evt = pair_evt[key]
                        p1_now = float(p1_max.get(key, 0.0))
                        if logits_sum[key] is None:
                            continue
                        avg_logits = logits_sum[key] / max(1.0, num_windows)
                        cid, cconf, probs = decode_valence_logits(avg_logits)
                        stage2_label = models.valence_id_to_label.get(cid, str(cid))
                        valence_label, valence_conf, friendly_score, unfriendly_score = decode_valence_scores(
                            probs,
                            models.valence_id_to_label,
                        )
                        interaction_score = float(friendly_score + unfriendly_score)
                        valence_conf_thresh = dynamic_valence_conf_threshold(
                            p1_now,
                            models.dynamic_valence_threshold,
                            VALENCE_MIN_CONF,
                        )
                        evt["last_pred_frame"] = frame_idx

                        if (
                            stage2_label.lower() == NO_INTERACTION_LABEL
                            or interaction_score < float(models.valence_interaction_thresh)
                            or valence_conf < valence_conf_thresh
                        ):
                            if not evt["active"] and evt["stage2_votes"]:
                                evt["stage2_votes"].clear()
                            continue

                        evt["stage2_votes"].append(valence_label)
                        voted_label = vote_mode(evt["stage2_votes"]) or valence_label

                        if (not evt["active"]) and (len(evt["stage2_votes"]) < VALENCE_MIN_START_VOTES):
                            evt["conf_stage2_max"] = max(evt.get("conf_stage2_max", 0.0), cconf)
                            evt["conf_valence_max"] = max(evt.get("conf_valence_max", 0.0), valence_conf)
                            evt["valence_thresh_min"] = min(evt.get("valence_thresh_min", 1.0), valence_conf_thresh)
                            evt["conf_friendly_max"] = max(evt.get("conf_friendly_max", 0.0), friendly_score)
                            evt["conf_unfriendly_max"] = max(evt.get("conf_unfriendly_max", 0.0), unfriendly_score)
                            if cconf >= evt.get("conf_stage2_max", 0.0):
                                evt["cur_stage2_label"] = stage2_label
                            continue

                        if evt["active"] and evt["cur_label"] != voted_label:
                            end_f = frame_idx - 1
                            finalize_and_write(evt, key, end_f)
                            reset_stage2_event(evt)
                            evt["conf_stage1_max"] = p1_now
                            evt["conf_stage2_max"] = float(cconf)
                            evt["conf_valence_max"] = float(valence_conf)
                            evt["valence_thresh_min"] = float(valence_conf_thresh)
                            evt["conf_friendly_max"] = float(friendly_score)
                            evt["conf_unfriendly_max"] = float(unfriendly_score)
                            evt["cur_stage2_label"] = stage2_label
                            evt["stage2_votes"].append(valence_label)
                            voted_label = valence_label

                        if not evt["active"]:
                            evt["active"] = True
                            evt["cur_label"] = voted_label
                            evt["start_f"] = frame_idx
                        if cconf >= evt.get("conf_stage2_max", 0.0):
                            evt["cur_stage2_label"] = stage2_label
                        evt["conf_stage2_max"] = max(evt.get("conf_stage2_max", 0.0), cconf)
                        evt["conf_valence_max"] = max(evt.get("conf_valence_max", 0.0), valence_conf)
                        evt["valence_thresh_min"] = min(evt.get("valence_thresh_min", 1.0), valence_conf_thresh)
                        evt["conf_friendly_max"] = max(evt.get("conf_friendly_max", 0.0), friendly_score)
                        evt["conf_unfriendly_max"] = max(evt.get("conf_unfriendly_max", 0.0), unfriendly_score)
                        evt["last_pos_frame"] = frame_idx
                        evt["last_pred_frame"] = frame_idx

                for key in eligible_keys:
                    evt = pair_evt[key]
                    if evt["active"] and (evt["last_pos_frame"] >= 0) and (frame_idx - evt["last_pos_frame"] > gap_tol_fr):
                        end_f = int(evt["last_pos_frame"])
                        finalize_and_write(evt, key, end_f)
                        reset_pair_after_gap(evt)
                        pair_cooldown_until[key] = end_f + cooldown_frames
        else:
            for key in suspect_pairs:
                evt = pair_evt[key]
                if evt["active"] and (evt["last_pos_frame"] >= 0) and (frame_idx - evt["last_pos_frame"] > gap_tol_fr):
                    end_f = int(evt["last_pos_frame"])
                    finalize_and_write(evt, key, end_f)
                    reset_pair_after_gap(evt)
                    pair_cooldown_until[key] = end_f + cooldown_frames

        for key, noevt in list(pair_noevt.items()):
            if not noevt.get("active", False):
                continue
            if (key not in suspect_pairs) and (noevt.get("last_frame", -1) >= 0) and (frame_idx - noevt["last_frame"] > gap_tol_fr):
                end_f0 = int(noevt["last_frame"])
                dur0 = int(end_f0 - int(noevt["start_f"]) + 1)
                if dur0 >= min_frames and float(noevt.get("p1_max", 0.0)) < float(models.interaction_gate_thresh):
                    no_inter_rows.append(
                        [
                            video_name,
                            int(key[0]),
                            int(key[1]),
                            int(noevt["start_f"]),
                            int(end_f0),
                            float(dur0 / fps),
                            float(noevt.get("p1_max", 0.0)),
                        ]
                    )
                noevt["active"] = False
                noevt["start_f"] = None
                noevt["last_frame"] = -1
                noevt["p1_max"] = 0.0

    pbar.close()

    for key, evt in list(pair_evt.items()):
        if evt["active"]:
            finalize_and_write(evt, key, frame_total - 1 if frame_total > 0 else 0)
            reset_pair_after_gap(evt)

    for key, noevt in list(pair_noevt.items()):
        if noevt.get("active", False) and (noevt.get("start_f") is not None):
            end_f0 = int(noevt.get("last_frame", frame_total - 1 if frame_total > 0 else 0))
            dur0 = int(end_f0 - int(noevt["start_f"]) + 1)
            if dur0 >= min_frames and float(noevt.get("p1_max", 0.0)) < float(models.interaction_gate_thresh):
                no_inter_rows.append(
                    [
                        video_name,
                        int(key[0]),
                        int(key[1]),
                        int(noevt["start_f"]),
                        int(end_f0),
                        float(dur0 / fps),
                        float(noevt.get("p1_max", 0.0)),
                    ]
                )

    output_summary = output_csvs(output_dir, video_name, inter_rows, no_inter_rows, prox_logs)
    output_summary.update(
        {
            "duration_sec": float(frame_total / fps) if fps else video_manifest.get("duration_sec"),
            "fps": float(fps) if fps else video_manifest.get("fps"),
            "width": video_manifest.get("width"),
            "height": video_manifest.get("height"),
            "frame_count": int(frame_total),
            "file_size_bytes": video_manifest.get("file_size_bytes"),
            "has_tracking_boxes": int(track_rows > 0),
            "has_keypoints": int(kpt_rows > 0),
            "source_tracking_rows": int(track_rows),
            "source_keypoint_rows": int(kpt_rows),
            "copied_visualization_videos": copied_videos,
        }
    )
    return output_summary


def build_video_manifest(input_manifest: dict, source_dir: Path, input_root: Path, output_summary: dict, fallback_video_id: int) -> dict:
    vm = dict(input_manifest.get("video_manifest", {}))
    vm = normalize_legacy_paths(vm)
    if not vm:
        vm = {
            "video_id": int(fallback_video_id),
            "source_path": str(source_dir),
            "source_root": str(input_root),
            "relative_path": str(source_dir.relative_to(input_root)) if _is_relative_to(source_dir, input_root) else source_dir.name,
            "data_source": DATA_SOURCE,
            "farm_id": "",
            "date": "",
            "camera_id": "",
            "gopro_id": GOPRO_ID,
            "segment_index": int(fallback_video_id),
            "format_tier": FORMAT_TIER,
            "sample_domain": "",
            "debug_only": False,
            "exclude_from_final_dataset": False,
            "preferred_for_local_test": True,
            "preferred_for_cluster_run": True,
            "duration_sec": output_summary.get("duration_sec"),
            "fps": output_summary.get("fps"),
            "width": output_summary.get("width"),
            "height": output_summary.get("height"),
            "file_size_bytes": output_summary.get("file_size_bytes"),
            "notes": f"generated by CSV interaction analysis; gopro_id: {GOPRO_ID_NOTE}",
        }
    else:
        vm["video_id"] = int(vm.get("video_id", fallback_video_id))
        vm["data_source"] = vm.get("data_source") or DATA_SOURCE
        vm["gopro_id"] = vm.get("gopro_id", GOPRO_ID)
        vm["format_tier"] = vm.get("format_tier") or FORMAT_TIER
        vm["duration_sec"] = output_summary.get("duration_sec", vm.get("duration_sec"))
        vm["fps"] = output_summary.get("fps", vm.get("fps"))
        vm["width"] = output_summary.get("width", vm.get("width"))
        vm["height"] = output_summary.get("height", vm.get("height"))
        vm["file_size_bytes"] = output_summary.get("file_size_bytes", vm.get("file_size_bytes"))
        note = str(vm.get("notes", "") or "")
        suffix = "CSV interaction analysis output; tracking/keypoints are source inputs and are not copied to output_s2"
        vm["notes"] = f"{note}; {suffix}" if note else suffix
    return vm


def build_manifest(
    source_dir: Path,
    input_root: Path,
    output_dir: Path,
    run_id: str,
    output_summary: dict,
    fallback_video_id: int,
    error: Exception | None = None,
) -> dict:
    input_manifest = normalize_legacy_paths(load_json_if_present(source_dir / "manifest.json"))
    video_manifest = build_video_manifest(input_manifest, source_dir, input_root, output_summary, fallback_video_id)
    usable = error is None
    feature_csv_manifest = {
        "feature_set_id": f"{run_id}_video_{int(video_manifest.get('video_id', fallback_video_id))}",
        "run_id": run_id,
        "video_id": int(video_manifest.get("video_id", fallback_video_id)),
        "source_video_path": normalize_legacy_path_string(str(video_manifest.get("source_path", source_dir))),
        "csv_root": str(_resolve_nonexistent(output_dir)),
        "source_csv_root": str(_resolve_nonexistent(source_dir)),
        "pipeline_stage": "csv_interaction_analysis",
        "perception_stage": "precomputed_csv",
        "interaction_gate_role": "interaction_gate",
        "valence_role": "valence_classifier",
        "interaction_gate_feature_schema": PTI_FEATURE_SCHEMA_VERSION,
        "valence_feature_schema": FEATURE_SCHEMA_VERSION,
        "has_tracking_boxes": int(output_summary.get("has_tracking_boxes", 0)),
        "has_keypoints": int(output_summary.get("has_keypoints", 0)),
        "has_interactions": int(output_summary.get("has_interactions", 0)),
        "has_no_interactions": int(output_summary.get("has_no_interactions", 0)),
        "has_adjacency": int(output_summary.get("has_adjacency", 0)),
        "adjacency_classes": list(output_summary.get("adjacency_classes", [])),
        "has_proximity_percentiles": int(output_summary.get("has_proximity_percentiles", 0)),
        "pipeline_version": run_id,
        "weights_version": run_id,
        "created_at": run_id,
        "format_tier": FORMAT_TIER,
        "debug_only": False,
        "usable_for_dev": bool(usable),
        "usable_for_final": bool(usable),
        "notes": "CSV interaction analysis output; tracking/keypoints were read from source_csv_root and not copied."
        if usable
        else "folder failed; see error",
    }
    csv_summary = {
        "source_tracking_rows": int(output_summary.get("source_tracking_rows", 0)),
        "source_keypoint_rows": int(output_summary.get("source_keypoint_rows", 0)),
        "copied_visualization_videos": list(output_summary.get("copied_visualization_videos", [])),
    }
    payload = {
        "video_manifest": video_manifest,
        "feature_csv_manifest": feature_csv_manifest,
        "csv_interaction_analysis_summary": csv_summary,
        "stage2_csv_summary": csv_summary,
    }
    if error is not None:
        payload["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exception(type(error), error, error.__traceback__),
        }
    return payload


def write_error_manifest(source_dir: Path, input_root: Path, output_dir: Path, run_id: str, fallback_video_id: int, exc: Exception) -> None:
    summary = {
        "duration_sec": None,
        "fps": None,
        "width": None,
        "height": None,
        "frame_count": 0,
        "file_size_bytes": None,
        "has_tracking_boxes": int(csv_has_data_rows(source_dir / "tracking_boxes.csv")),
        "has_keypoints": int(csv_has_data_rows(source_dir / "keypoints.csv")),
        "has_interactions": 0,
        "has_no_interactions": 0,
        "has_adjacency": 0,
        "adjacency_classes": [],
        "has_proximity_percentiles": 0,
        "source_tracking_rows": 0,
        "source_keypoint_rows": 0,
        "copied_visualization_videos": [],
    }
    manifest = build_manifest(source_dir, input_root, output_dir, run_id, summary, fallback_video_id, error=exc)
    write_json(output_dir / "manifest.json", manifest)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the default two-stage Stage-2 cascade from existing tracking_boxes.csv and keypoints.csv outputs."
    )
    parser.add_argument(
        "--input-root",
        default=str(DEFAULT_INPUT_ROOT),
        help="Root containing existing Stage 1 tracking_boxes.csv and keypoints.csv outputs.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT), help="CSV interaction analysis output root. Defaults to ./output_s2.")
    parser.add_argument(
        "--interaction-gate-ckpt",
        dest="interaction_gate_ckpt",
        default=str(DEFAULT_INTERACTION_GATE_CKPT),
        help="Interaction gate checkpoint for the default two-stage cascade.",
    )
    parser.add_argument(
        "--valence-ckpt",
        dest="valence_ckpt",
        default=str(DEFAULT_VALENCE_CKPT),
        help="Valence classifier checkpoint for the default two-stage cascade.",
    )
    parser.add_argument("--device", default="cuda", help="CUDA device, e.g. cuda or cuda:0. CPU is intentionally refused.")
    parser.add_argument("--profile", choices=["auto", "8gb", "40gb"], default="auto", help="VRAM profile for batch defaults.")
    parser.add_argument("--inference-batch-size", type=int, default=None, help="Override inference batch size.")
    parser.add_argument("--fps", type=float, default=60.0, help="Fallback FPS if manifest has none.")
    parser.add_argument("--max-folders", type=int, default=0, help="Limit folders for smoke tests; 0 means all.")
    parser.add_argument("--log-file", default="", help="Optional UTF-8 log file path.")
    parser.add_argument("--verify-log-flush", action="store_true", help="Write a few flushed log lines and exit.")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA fp16 autocast for temporal inference.")
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    log_path = Path(args.log_file) if args.log_file else DEFAULT_LOG_DIR / f"stage2_from_csv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    global LOGGER
    LOGGER = TeeLogger(log_path)
    try:
        log(f"[info] log file: {log_path}")
        if args.verify_log_flush:
            for i in range(1, 4):
                log(f"[flush-check] tick={i} time={datetime.now().isoformat(timespec='seconds')}")
                time.sleep(1.0)
            log("[flush-check] ok")
            return 0

        defaults = profile_defaults(args.profile)
        if args.inference_batch_size is None:
            args.inference_batch_size = defaults["inference_batch_size"]
        log(f"[info] profile={args.profile} inference_batch_size={args.inference_batch_size} amp={not args.no_amp}")

        input_root = _resolve_nonexistent(Path(args.input_root))
        folders = discover_input_folders(input_root)
        if args.max_folders > 0:
            folders = folders[: args.max_folders]
        if not folders:
            raise RuntimeError(f"no folders with tracking_boxes.csv/keypoints.csv found under {input_root}")

        output_root = safe_clear_output_root(Path(args.output_dir))
        log(f"[info] cleared output root before neural-network load: {output_root}")

        device = resolve_device(args.device)
        models = load_models(args, device)
        run_id = datetime.now().strftime(RUN_TIMESTAMP_FORMAT)

        log(f"[info] discovered {len(folders)} folder(s) under {input_root}")
        completed = 0
        failed = 0
        pbar = tqdm(folders, desc="csv interaction folders", unit="folder", ascii=True, mininterval=1.0, file=sys.stdout)
        for idx, source_dir in enumerate(pbar, start=1):
            try:
                rel = source_dir.relative_to(input_root)
            except ValueError:
                rel = Path(source_dir.name)
            output_dir = safe_make_output_dir(output_root, rel)
            log(f"[info] folder {idx}/{len(folders)}: {source_dir} -> {output_dir}")
            try:
                summary = process_folder(source_dir, output_dir, models, device, args)
                manifest = build_manifest(source_dir, input_root, output_dir, run_id, summary, idx, error=None)
                write_json(output_dir / "manifest.json", manifest)
                completed += 1
                log(
                    "[info] done "
                    f"{source_dir.name}: interactions={summary.get('has_interactions', 0)} "
                    f"no_interactions={summary.get('has_no_interactions', 0)} "
                    f"adjacency={summary.get('adjacency_classes', [])}"
                )
            except Exception as exc:
                failed += 1
                write_error_manifest(source_dir, input_root, output_dir, run_id, idx, exc)
                log(f"[error] failed folder {source_dir}: {exc}")
                log("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))

        log(f"[done] completed={completed} failed={failed} output_root={output_root}")
        return 0 if failed == 0 else 2
    finally:
        LOGGER.close()


if __name__ == "__main__":
    raise SystemExit(main())
