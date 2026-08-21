"""Unified ID-based inference interface for calibrated S03 link models."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Literal, Protocol

from cowtrack.config import ContractError
from cowtrack.linking.features import LONG_FEATURE_SCHEMA, SHORT_FEATURE_SCHEMA
from cowtrack.linking.model import LinkResult, LinkScorer, load_link_model
from cowtrack.linking.pseudo_pairs import TrackletPairFeatures


LinkMode = Literal["short", "long"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ContractError(f"cannot fingerprint S03 scorer artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _verified_runtime_files(directory: Path) -> None:
    success_path = directory / "_SUCCESS.json"
    try:
        success = json.loads(success_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read S03 success marker: {exc}") from exc
    if not isinstance(success, dict) or success.get("stage") != "S03":
        raise ContractError("calibration directory is not a completed S03 output")
    config_hash = success.get("config_hash")
    if (
        not isinstance(config_hash, str)
        or len(config_hash) != 64
        or any(character not in "0123456789abcdef" for character in config_hash)
    ):
        raise ContractError("S03 success marker has an invalid config hash")
    records = success.get("output_fingerprints")
    if not isinstance(records, list):
        raise ContractError("S03 success marker lacks output fingerprints")
    by_name = {
        str(item.get("path")): item for item in records if isinstance(item, dict)
    }
    if len(by_name) != len(records):
        raise ContractError("S03 success marker has duplicate/invalid fingerprints")
    required = (
        "link_model_short.joblib",
        "link_model_long.joblib",
        "thresholds.json",
        "pair_feature_schema.json",
        "effective_config.json",
    )
    missing = sorted(set(required) - set(by_name))
    if missing:
        raise ContractError(f"S03 success marker lacks runtime artifacts: {missing}")
    for name in required:
        path = directory / name
        if not path.is_file():
            raise ContractError(f"S03 runtime artifact does not exist: {path}")
        record = by_name[name]
        if path.stat().st_size != record.get("size_bytes"):
            raise ContractError(f"completed S03 runtime artifact changed: {name} (size)")
        if _sha256(path) != record.get("sha256"):
            raise ContractError(f"completed S03 runtime artifact changed: {name} (sha256)")


class PairFeatureProvider(Protocol):
    def build_pair(
        self, source_tracklet_id: int, target_tracklet_id: int, mode: LinkMode
    ) -> TrackletPairFeatures: ...


class CalibratedLinkScorer:
    """Score tracklet IDs while preserving missing/high-overlap reliability gates."""

    def __init__(
        self,
        short_scorer: LinkScorer,
        long_scorer: LinkScorer,
        feature_provider: PairFeatureProvider,
    ) -> None:
        if short_scorer.model.mode != "short" or long_scorer.model.mode != "long":
            raise ContractError("S03 unified scorer received swapped model modes")
        if not hasattr(feature_provider, "build_pair"):
            raise ContractError("S03 unified scorer lacks a pair feature provider")
        self._scorers = {"short": short_scorer, "long": long_scorer}
        self._features = feature_provider

    @classmethod
    def from_directory(
        cls, calibration_dir: Path, feature_provider: PairFeatureProvider
    ) -> "CalibratedLinkScorer":
        directory = calibration_dir.resolve()
        _verified_runtime_files(directory)
        # This is intentionally a second, semantic validation in addition to
        # byte fingerprinting: the canonical config hash must reproduce the
        # exact hash committed by the completed S03 marker.
        from cowtrack.linking.runtime import load_s03_runtime_config

        load_s03_runtime_config(directory)
        try:
            thresholds = json.loads(
                (directory / "thresholds.json").read_text(encoding="utf-8")
            )
            feature_schema = json.loads(
                (directory / "pair_feature_schema.json").read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ContractError(f"cannot read S03 runtime JSON: {exc}") from exc
        if not isinstance(thresholds, dict):
            raise ContractError("S03 thresholds root must be an object")
        short_schema = feature_schema.get("short") if isinstance(feature_schema, dict) else None
        long_schema = feature_schema.get("long") if isinstance(feature_schema, dict) else None
        if (
            not isinstance(short_schema, dict)
            or not isinstance(long_schema, dict)
            or short_schema.get("ordered_features") != list(SHORT_FEATURE_SCHEMA)
            or long_schema.get("ordered_features") != list(LONG_FEATURE_SCHEMA)
        ):
            raise ContractError("S03 persisted feature schema differs")
        short_thresholds = thresholds.get("short")
        long_thresholds = thresholds.get("long")
        if not isinstance(short_thresholds, dict) or not isinstance(
            long_thresholds, dict
        ):
            raise ContractError("S03 thresholds lack short/long policies")
        short_model = load_link_model(directory / "link_model_short.joblib")
        long_model = load_link_model(directory / "link_model_long.joblib")
        if short_model.feature_names != SHORT_FEATURE_SCHEMA:
            raise ContractError("S03 short runtime model feature schema differs")
        if long_model.feature_names != LONG_FEATURE_SCHEMA:
            raise ContractError("S03 long runtime model feature schema differs")
        return cls(
            LinkScorer(short_model, short_thresholds),
            LinkScorer(long_model, long_thresholds),
            feature_provider,
        )

    def score_pair(
        self,
        source_tracklet_id: int,
        target_tracklet_id: int,
        mode: LinkMode,
    ) -> LinkResult:
        """Implement the S03 `score_pair(source_id, target_id, mode)` contract.

        Long-link confirmation additionally requires a candidate-set margin.
        Since the three-argument interface has no candidate set, a qualifying
        long score remains provisional.  S05 can call `score_pair_with_margin`
        after retrieval/ranking.
        """

        return self.score_pair_with_margin(
            source_tracklet_id, target_tracklet_id, mode, candidate_margin=None
        )

    def score_pair_with_margin(
        self,
        source_tracklet_id: int,
        target_tracklet_id: int,
        mode: LinkMode,
        *,
        candidate_margin: float | None,
    ) -> LinkResult:
        if mode not in {"short", "long"}:
            raise ContractError("S03 score_pair mode must be short or long")
        built = self._features.build_pair(
            int(source_tracklet_id), int(target_tracklet_id), mode
        )
        if built.feature_values is None:
            return LinkResult(
                probability=None,
                raw_score=None,
                decision="reject",
                feature_values=None,
                reason=built.reason or "features_unavailable",
                appearance_present=built.appearance_present,
                high_overlap=built.high_overlap,
                source_gallery=built.source_gallery,
                target_gallery=built.target_gallery,
            )
        scored = self._scorers[mode].score_features(
            built.feature_values,
            appearance_present=built.appearance_present,
            high_overlap=built.high_overlap,
            candidate_margin=candidate_margin,
        )
        return replace(
            scored,
            source_gallery=built.source_gallery,
            target_gallery=built.target_gallery,
        )


__all__ = ["CalibratedLinkScorer", "LinkMode", "PairFeatureProvider"]
