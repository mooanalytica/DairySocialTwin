from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cowtrack.config import ContractError
from cowtrack.linking.config import load_link_calibration_config
from cowtrack.linking.features import LONG_FEATURE_SCHEMA, SHORT_FEATURE_SCHEMA
from cowtrack.linking.model import (
    LinkScorer,
    clopper_pearson_upper,
    disabled_link_model,
    fit_link_model,
    save_link_model,
)
from cowtrack.linking.pseudo_pairs import TrackletPairFeatures
from cowtrack.linking.scorer import CalibratedLinkScorer


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "s03_calibration.yaml"


def _model(mode: str):
    values = np.r_[np.linspace(-2.0, -0.2, 40), np.linspace(0.2, 2.0, 40)][
        :, None
    ]
    labels = np.r_[np.zeros(40, dtype=np.int8), np.ones(40, dtype=np.int8)]
    return fit_link_model(
        values,
        labels,
        ("similarity",),
        mode=mode,
        random_seed=7,
        logistic_c=1.0,
        max_iter=1000,
        minimum_per_class=20,
    )


class Provider:
    def build_pair(self, source_tracklet_id, target_tracklet_id, mode):
        if source_tracklet_id == 99:
            return TrackletPairFeatures(None, False, False, "appearance_missing")
        return TrackletPairFeatures(
            {"similarity": 2.0}, True, target_tracklet_id == 88, None
        )


def _thresholds(mode: str, margin: float | None) -> dict[str, object]:
    certification_negatives = 30_000
    return {
        "mode": mode,
        "model_enabled": True,
        "disabled_reason": None,
        "confirmed_enabled": True,
        "confirmed_threshold": 0.5,
        "provisional_threshold": 0.4,
        "appearance_margin_threshold": margin,
        "confirmed_far_target": 1e-4,
        "confidence_level": 0.95,
        "confirmed_false_accepts": 0,
        "certification_hard_negative_count": certification_negatives,
        "confirmed_false_accept_upper": clopper_pearson_upper(
            0, certification_negatives, 0.95
        ),
        "confirmed_certified_independently": True,
    }


def _unified() -> CalibratedLinkScorer:
    short = LinkScorer(
        _model("short"),
        _thresholds("short", None),
    )
    long = LinkScorer(
        _model("long"),
        _thresholds("long", 0.2),
    )
    return CalibratedLinkScorer(short, long, Provider())


def test_three_argument_score_pair_and_long_margin_contract() -> None:
    scorer = _unified()
    assert scorer.score_pair(1, 2, "short").decision == "confirmed"
    assert scorer.score_pair(1, 2, "long").decision == "provisional"
    assert (
        scorer.score_pair_with_margin(1, 2, "long", candidate_margin=0.3).decision
        == "confirmed"
    )


def test_unified_scorer_rejects_missing_and_caps_overlap() -> None:
    scorer = _unified()
    missing = scorer.score_pair(99, 2, "short")
    assert missing.decision == "reject"
    assert missing.probability is None
    assert missing.feature_values is None
    assert missing.appearance_present is False
    overlap = scorer.score_pair(1, 88, "short")
    assert overlap.decision == "provisional"


def test_from_directory_verifies_common_success_fingerprints(tmp_path) -> None:
    short_reason = "short_disabled"
    long_reason = "long_disabled"
    save_link_model(
        tmp_path / "link_model_short.joblib",
        disabled_link_model(
            SHORT_FEATURE_SCHEMA, mode="short", random_seed=7, reason=short_reason
        ),
    )
    save_link_model(
        tmp_path / "link_model_long.joblib",
        disabled_link_model(
            LONG_FEATURE_SCHEMA, mode="long", random_seed=7, reason=long_reason
        ),
    )

    def disabled_threshold(mode: str, reason: str) -> dict[str, object]:
        return {
            "mode": mode,
            "model_enabled": False,
            "disabled_reason": reason,
            "confirmed_enabled": False,
            "confirmed_threshold": None,
            "provisional_threshold": None,
            "appearance_margin_threshold": None,
        }

    (tmp_path / "thresholds.json").write_text(
        json.dumps(
            {
                "short": disabled_threshold("short", short_reason),
                "long": disabled_threshold("long", long_reason),
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "pair_feature_schema.json").write_text(
        json.dumps(
            {
                "short": {"ordered_features": list(SHORT_FEATURE_SCHEMA)},
                "long": {"ordered_features": list(LONG_FEATURE_SCHEMA)},
            }
        ),
        encoding="utf-8",
    )
    _, effective_config, config_hash = load_link_calibration_config(CONFIG)
    (tmp_path / "effective_config.json").write_text(
        json.dumps(effective_config, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    runtime_names = (
        "link_model_short.joblib",
        "link_model_long.joblib",
        "thresholds.json",
        "pair_feature_schema.json",
        "effective_config.json",
    )
    fingerprints = []
    for name in runtime_names:
        payload = (tmp_path / name).read_bytes()
        fingerprints.append(
            {
                "path": name,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    (tmp_path / "_SUCCESS.json").write_text(
        json.dumps(
            {
                "stage": "S03",
                "config_hash": config_hash,
                "output_fingerprints": fingerprints,
            }
        ),
        encoding="utf-8",
    )
    scorer = CalibratedLinkScorer.from_directory(tmp_path, Provider())
    assert scorer.score_pair(1, 2, "short").decision == "reject"

    with (tmp_path / "thresholds.json").open("a", encoding="utf-8") as handle:
        handle.write(" ")
    with pytest.raises(ContractError, match="runtime artifact changed"):
        CalibratedLinkScorer.from_directory(tmp_path, Provider())
