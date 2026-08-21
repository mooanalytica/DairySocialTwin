from __future__ import annotations

import importlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from cowtrack.config import ContractError


DEFAULT_MODEL_ROOT = Path("/home/hyw/re-identification-models")


@dataclass(frozen=True)
class EncoderProfile:
    """A complete, local-only encoder and preprocessing contract."""

    name: str
    model_id: str
    timm_model_name: str
    checkpoint_path: Path
    checkpoint_format: Literal["pytorch_bin", "safetensors"]
    checkpoint_filter: Literal["none", "timm_swin"]
    input_size: int
    resize_mode: Literal["stretch", "shortest_edge_center_crop"]
    crop_pct: float
    interpolation: Literal["bilinear", "bicubic"]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    embedding_dim: int
    preprocessing_id: str

    def __post_init__(self) -> None:
        if not self.name or not self.model_id or not self.timm_model_name:
            raise ContractError("encoder profile identifiers cannot be blank")
        if not isinstance(self.checkpoint_path, Path):
            raise ContractError("encoder checkpoint_path must be a pathlib.Path")
        if self.checkpoint_format not in {"pytorch_bin", "safetensors"}:
            raise ContractError("unsupported encoder checkpoint format")
        if self.checkpoint_filter not in {"none", "timm_swin"}:
            raise ContractError("unsupported encoder checkpoint filter")
        if (
            isinstance(self.input_size, bool)
            or not isinstance(self.input_size, Integral)
            or isinstance(self.embedding_dim, bool)
            or not isinstance(self.embedding_dim, Integral)
            or self.input_size <= 0
            or self.embedding_dim <= 0
        ):
            raise ContractError("encoder dimensions must be positive")
        if self.resize_mode not in {"stretch", "shortest_edge_center_crop"}:
            raise ContractError("unsupported encoder resize_mode")
        if self.interpolation not in {"bilinear", "bicubic"}:
            raise ContractError("unsupported encoder interpolation")
        if (
            isinstance(self.crop_pct, bool)
            or not isinstance(self.crop_pct, Real)
            or not math.isfinite(float(self.crop_pct))
            or not 0.0 < float(self.crop_pct) <= 1.0
        ):
            raise ContractError("encoder crop_pct must be in (0, 1]")
        if self.resize_mode == "stretch" and not math.isclose(
            float(self.crop_pct), 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ContractError("stretch encoder profiles require crop_pct=1.0")
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ContractError("encoder normalization must contain three channels")
        if not all(math.isfinite(value) for value in (*self.mean, *self.std)):
            raise ContractError("encoder normalization must be finite")
        if not all(value > 0.0 for value in self.std):
            raise ContractError("encoder normalization std must be positive")
        if not self.preprocessing_id:
            raise ContractError("encoder preprocessing_id cannot be blank")


class CropEncoder(Protocol):
    """Minimal interface used by four-view S02 inference."""

    def embed(
        self, crops: Sequence[np.ndarray], *, batch_size: int
    ) -> np.ndarray: ...


def production_encoder_profiles(
    model_root: Path = DEFAULT_MODEL_ROOT,
) -> tuple[EncoderProfile, ...]:
    """Return the three fixed bake-off profiles backed by local checkpoints."""

    if not isinstance(model_root, Path):
        raise ContractError("model_root must be a pathlib.Path")
    mega_checkpoint = model_root / "MegaDescriptor-L-384" / "pytorch_model.bin"
    dino_checkpoint = model_root / "DINOv2-ViT-B-14" / "model.safetensors"
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    return (
        EncoderProfile(
            name="megadescriptor_l_384_half",
            model_id="BVRA/MegaDescriptor-L-384",
            timm_model_name="swin_large_patch4_window12_384",
            checkpoint_path=mega_checkpoint,
            checkpoint_format="pytorch_bin",
            checkpoint_filter="timm_swin",
            input_size=384,
            resize_mode="stretch",
            crop_pct=1.0,
            interpolation="bilinear",
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            embedding_dim=1536,
            preprocessing_id="rgb_stretch_384_bilinear_mean_std_0.5",
        ),
        EncoderProfile(
            name="megadescriptor_l_384_imagenet",
            model_id="BVRA/MegaDescriptor-L-384",
            timm_model_name="swin_large_patch4_window12_384",
            checkpoint_path=mega_checkpoint,
            checkpoint_format="pytorch_bin",
            checkpoint_filter="timm_swin",
            input_size=384,
            resize_mode="shortest_edge_center_crop",
            crop_pct=0.9,
            interpolation="bicubic",
            mean=imagenet_mean,
            std=imagenet_std,
            embedding_dim=1536,
            preprocessing_id=(
                "timm_rgb_shortest_edge_426_center_crop_384_bicubic_imagenet"
            ),
        ),
        EncoderProfile(
            name="dinov2_vit_b_14_timm",
            model_id="timm/vit_base_patch14_dinov2.lvd142m",
            timm_model_name="vit_base_patch14_dinov2.lvd142m",
            checkpoint_path=dino_checkpoint,
            checkpoint_format="safetensors",
            checkpoint_filter="none",
            input_size=518,
            resize_mode="shortest_edge_center_crop",
            crop_pct=1.0,
            interpolation="bicubic",
            mean=imagenet_mean,
            std=imagenet_std,
            embedding_dim=768,
            preprocessing_id=(
                "timm_rgb_shortest_edge_518_center_crop_518_bicubic_imagenet"
            ),
        ),
    )


def encoder_profile_by_name(
    name: str, model_root: Path = DEFAULT_MODEL_ROOT
) -> EncoderProfile:
    profiles = {profile.name: profile for profile in production_encoder_profiles(model_root)}
    try:
        return profiles[name]
    except KeyError as exc:
        choices = ", ".join(sorted(profiles))
        raise ContractError(f"unknown encoder profile {name!r}; expected one of {choices}") from exc


def _load_checkpoint(
    profile: EncoderProfile, torch_module: Any
) -> Mapping[str, Any]:
    path = profile.checkpoint_path
    if not path.is_file():
        raise ContractError(f"encoder checkpoint does not exist: {path}")
    expected_suffix = (
        ".bin" if profile.checkpoint_format == "pytorch_bin" else ".safetensors"
    )
    if path.suffix != expected_suffix:
        raise ContractError(
            f"encoder checkpoint {path} must use {expected_suffix} for "
            f"format {profile.checkpoint_format}"
        )

    try:
        if profile.checkpoint_format == "pytorch_bin":
            state_dict = torch_module.load(
                path, map_location="cpu", weights_only=True
            )
        else:
            safetensors_torch = importlib.import_module("safetensors.torch")
            state_dict = safetensors_torch.load_file(str(path), device="cpu")
    except Exception as exc:
        raise ContractError(f"failed to read encoder checkpoint {path}: {exc}") from exc

    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ContractError(f"encoder checkpoint is not a non-empty state dict: {path}")
    if not all(isinstance(key, str) for key in state_dict):
        raise ContractError(f"encoder checkpoint has a non-string state key: {path}")
    return state_dict


class TimmCropEncoder:
    """Strict local timm encoder with no download or model fallback path."""

    def __init__(self, profile: EncoderProfile, *, device: str) -> None:
        if not isinstance(profile, EncoderProfile):
            raise ContractError("profile must be an EncoderProfile")
        if not isinstance(device, str) or not device.strip():
            raise ContractError("encoder device cannot be blank")
        if not profile.checkpoint_path.is_file():
            # Check before importing heavyweight libraries. Missing data is a
            # contract error, not a reason to contact a model hub.
            raise ContractError(
                f"encoder checkpoint does not exist: {profile.checkpoint_path}"
            )

        try:
            torch_module = importlib.import_module("torch")
            timm_module = importlib.import_module("timm")
        except Exception as exc:
            raise ContractError(f"failed to import local encoder runtime: {exc}") from exc

        try:
            model = timm_module.create_model(
                profile.timm_model_name,
                pretrained=False,
                num_classes=0,
            )
        except Exception as exc:
            raise ContractError(
                f"failed to construct timm model {profile.timm_model_name}: {exc}"
            ) from exc

        state_dict = _load_checkpoint(profile, torch_module)
        if profile.checkpoint_filter == "timm_swin":
            try:
                swin_module = importlib.import_module(
                    "timm.models.swin_transformer"
                )
                state_dict = swin_module.checkpoint_filter_fn(state_dict, model)
            except Exception as exc:
                raise ContractError(
                    f"fixed timm Swin checkpoint conversion failed for "
                    f"{profile.name}: {exc}"
                ) from exc
        if not isinstance(state_dict, Mapping) or not state_dict or not all(
            isinstance(key, str) for key in state_dict
        ):
            raise ContractError(
                f"checkpoint filter returned an invalid state dict for {profile.name}"
            )
        try:
            incompatible = model.load_state_dict(state_dict, strict=True)
        except Exception as exc:
            raise ContractError(
                f"strict state load failed for {profile.name}: {exc}"
            ) from exc
        if incompatible.missing_keys or incompatible.unexpected_keys:
            # PyTorch strict=True normally raises first; retain an explicit
            # assertion so a non-standard model implementation cannot weaken it.
            raise ContractError(
                f"strict state load returned incompatible keys for {profile.name}"
            )

        try:
            model = model.to(device)
            model.eval()
        except Exception as exc:
            raise ContractError(
                f"failed to place encoder {profile.name} on {device}: {exc}"
            ) from exc

        self.profile = profile
        self.device = device
        self._torch = torch_module
        self._model = model

    def _preprocess_one(self, crop: np.ndarray) -> Any:
        _validate_rgb_crop(crop)
        torch_module = self._torch
        # A copy avoids read-only-array warnings and guarantees positive strides.
        tensor = torch_module.from_numpy(np.array(crop, copy=True))
        tensor = tensor.permute(2, 0, 1).to(dtype=torch_module.float32).div_(255.0)
        resize_height, resize_width, crop_top, crop_left = _resize_geometry(
            self.profile, height=int(crop.shape[0]), width=int(crop.shape[1])
        )
        tensor = torch_module.nn.functional.interpolate(
            tensor.unsqueeze(0),
            size=(resize_height, resize_width),
            mode=self.profile.interpolation,
            align_corners=False,
            antialias=True,
        ).squeeze(0)
        tensor.clamp_(0.0, 1.0)
        if self.profile.resize_mode == "shortest_edge_center_crop":
            size = self.profile.input_size
            tensor = tensor[
                :,
                crop_top : crop_top + size,
                crop_left : crop_left + size,
            ]
        if tuple(tensor.shape) != (
            3,
            self.profile.input_size,
            self.profile.input_size,
        ):
            raise ContractError(
                f"encoder {self.profile.name} preprocessing returned shape "
                f"{tuple(tensor.shape)}"
            )
        mean = tensor.new_tensor(self.profile.mean).view(3, 1, 1)
        std = tensor.new_tensor(self.profile.std).view(3, 1, 1)
        return tensor.sub_(mean).div_(std)

    def embed(
        self, crops: Sequence[np.ndarray], *, batch_size: int
    ) -> np.ndarray:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise ContractError("encoder batch_size must be an integer")
        if batch_size <= 0:
            raise ContractError("encoder batch_size must be positive")
        if not crops:
            raise ContractError("encoder requires at least one crop")

        chunks: list[np.ndarray] = []
        torch_module = self._torch
        with torch_module.inference_mode():
            for start in range(0, len(crops), batch_size):
                batch_crops = crops[start : start + batch_size]
                inputs = torch_module.stack(
                    [self._preprocess_one(crop) for crop in batch_crops], dim=0
                ).to(self.device, non_blocking=True)
                outputs = self._model(inputs)
                if not isinstance(outputs, torch_module.Tensor) or outputs.ndim != 2:
                    raise ContractError(
                        f"encoder {self.profile.name} must return a rank-2 tensor"
                    )
                if outputs.shape[0] != len(batch_crops):
                    raise ContractError(
                        f"encoder {self.profile.name} changed the batch dimension"
                    )
                if outputs.shape[1] != self.profile.embedding_dim:
                    raise ContractError(
                        f"encoder {self.profile.name} returned dimension "
                        f"{outputs.shape[1]}, expected {self.profile.embedding_dim}"
                    )
                chunks.append(outputs.detach().to(dtype=torch_module.float32).cpu().numpy())

        result = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        if not np.isfinite(result).all():
            raise ContractError(f"encoder {self.profile.name} returned non-finite values")
        return result


def build_encoder(profile: EncoderProfile, *, device: str) -> TimmCropEncoder:
    """Build a local checkpoint-backed encoder without network access."""

    return TimmCropEncoder(profile, device=device)


def _validate_rgb_crop(crop: np.ndarray) -> None:
    if not isinstance(crop, np.ndarray):
        raise ContractError("encoder crops must be numpy arrays")
    if crop.dtype != np.uint8:
        raise ContractError("encoder crops must have uint8 RGB values")
    if crop.ndim != 3 or crop.shape[2] != 3 or min(crop.shape[:2]) <= 0:
        raise ContractError("encoder crops must have shape [height, width, 3]")


def _resize_geometry(
    profile: EncoderProfile, *, height: int, width: int
) -> tuple[int, int, int, int]:
    """Return timm-compatible resize dimensions and center-crop offsets."""

    if height <= 0 or width <= 0:
        raise ContractError("encoder crop dimensions must be positive")
    size = int(profile.input_size)
    if profile.resize_mode == "stretch":
        return size, size, 0, 0

    resized_short_edge = math.floor(size / float(profile.crop_pct))
    if resized_short_edge < size:
        raise ContractError("encoder crop_pct produced an undersized resize")
    if width <= height:
        resize_width = resized_short_edge
        resize_height = int(resized_short_edge * height / width)
    else:
        resize_height = resized_short_edge
        resize_width = int(resized_short_edge * width / height)
    crop_top = int(round((resize_height - size) / 2.0))
    crop_left = int(round((resize_width - size) / 2.0))
    if (
        crop_top < 0
        or crop_left < 0
        or crop_top + size > resize_height
        or crop_left + size > resize_width
    ):
        raise ContractError("encoder center-crop geometry is invalid")
    return resize_height, resize_width, crop_top, crop_left


def _normalize_rows(vectors: np.ndarray, *, context: str) -> np.ndarray:
    result = np.asarray(vectors, dtype=np.float32)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
        raise ContractError(f"{context} embeddings must be a non-empty rank-2 array")
    if not np.isfinite(result).all():
        raise ContractError(f"{context} embeddings contain non-finite values")
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
        raise ContractError(f"{context} embeddings contain a zero-norm vector")
    normalized = result / norms
    if not np.isfinite(normalized).all():
        raise ContractError(f"{context} normalized embeddings are non-finite")
    return normalized.astype(np.float32, copy=False)


def embed_four_views(
    encoder: CropEncoder,
    raw_crops: Sequence[np.ndarray],
    masked_crops: Sequence[np.ndarray],
    *,
    batch_size: int,
) -> np.ndarray:
    """Embed raw/masked crops at 0/180 degrees and normalize their mean.

    Every one of the four view embeddings is L2-normalized before averaging;
    the averaged embedding is normalized once more. Horizontal flips are never
    generated.
    """

    if len(raw_crops) == 0 or len(raw_crops) != len(masked_crops):
        raise ContractError("raw and masked crop sequences must have equal non-zero length")
    for raw, masked in zip(raw_crops, masked_crops, strict=True):
        _validate_rgb_crop(raw)
        _validate_rgb_crop(masked)
        if raw.shape != masked.shape:
            raise ContractError("raw and masked versions must have identical shapes")

    raw_180 = [np.ascontiguousarray(np.rot90(crop, 2)) for crop in raw_crops]
    masked_180 = [np.ascontiguousarray(np.rot90(crop, 2)) for crop in masked_crops]
    view_inputs = (raw_crops, raw_180, masked_crops, masked_180)
    normalized_views: list[np.ndarray] = []
    expected_shape: tuple[int, int] | None = None
    for view_index, crops in enumerate(view_inputs):
        embeddings = encoder.embed(crops, batch_size=batch_size)
        normalized = _normalize_rows(embeddings, context=f"view {view_index}")
        if expected_shape is None:
            expected_shape = normalized.shape
        elif normalized.shape != expected_shape:
            raise ContractError("encoder returned inconsistent four-view shapes")
        normalized_views.append(normalized)

    averaged = np.mean(np.stack(normalized_views, axis=0), axis=0, dtype=np.float32)
    return _normalize_rows(averaged, context="four-view mean")
