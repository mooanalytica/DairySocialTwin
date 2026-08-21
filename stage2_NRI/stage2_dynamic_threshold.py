from __future__ import annotations

import math
from collections import Counter
from typing import Iterable


DYNAMIC_THRESHOLD_SCHEMA_VERSION = "stage2_gate_valence_threshold_v1"
CASCADE_LABELS = ("no_interaction", "friendly", "unfriendly")


def _finite_float(value, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def fixed_valence_threshold_config(threshold: float = 0.60, gate_threshold: float = 0.50) -> dict:
    threshold = _clamp(_finite_float(threshold, 0.60), 0.0, 1.0)
    return {
        "schema": DYNAMIC_THRESHOLD_SCHEMA_VERSION,
        "enabled": False,
        "method": "fixed_valence_conf",
        "gate_threshold": _clamp(_finite_float(gate_threshold, 0.50), 0.0, 1.0),
        "base_valence_conf": threshold,
        "min_valence_conf": threshold,
        "max_valence_conf": threshold,
        "gate_exponent": 1.0,
        "fixed_baseline_valence_conf": threshold,
    }


def coerce_dynamic_threshold_config(raw, fallback_fixed: float = 0.60, gate_threshold: float = 0.50) -> dict:
    if not isinstance(raw, dict):
        return fixed_valence_threshold_config(fallback_fixed, gate_threshold)

    cfg = dict(raw)
    cfg.setdefault("schema", DYNAMIC_THRESHOLD_SCHEMA_VERSION)
    cfg.setdefault("enabled", True)
    cfg["gate_threshold"] = _clamp(_finite_float(cfg.get("gate_threshold"), gate_threshold), 0.0, 1.0)
    cfg["base_valence_conf"] = _clamp(_finite_float(cfg.get("base_valence_conf"), fallback_fixed), 0.0, 1.0)
    cfg["min_valence_conf"] = _clamp(_finite_float(cfg.get("min_valence_conf"), cfg["base_valence_conf"]), 0.0, 1.0)
    cfg["max_valence_conf"] = _clamp(_finite_float(cfg.get("max_valence_conf"), cfg["base_valence_conf"]), 0.0, 1.0)
    cfg["gate_exponent"] = max(1e-6, _finite_float(cfg.get("gate_exponent"), 1.0))
    cfg["fixed_baseline_valence_conf"] = _clamp(
        _finite_float(cfg.get("fixed_baseline_valence_conf"), fallback_fixed),
        0.0,
        1.0,
    )
    if cfg["min_valence_conf"] > cfg["base_valence_conf"]:
        cfg["min_valence_conf"] = cfg["base_valence_conf"]
    if cfg["max_valence_conf"] < cfg["base_valence_conf"]:
        cfg["max_valence_conf"] = cfg["base_valence_conf"]
    return cfg


def dynamic_valence_conf_threshold(gate_prob: float, config: dict | None, fallback_fixed: float = 0.60) -> float:
    cfg = coerce_dynamic_threshold_config(config, fallback_fixed)
    if not bool(cfg.get("enabled", False)):
        return float(cfg["base_valence_conf"])

    gate = _clamp(_finite_float(gate_prob, cfg["gate_threshold"]), 0.0, 1.0)
    gate_floor = _clamp(_finite_float(cfg.get("gate_threshold"), 0.50), 0.0, 1.0)
    denom = max(1e-6, 1.0 - gate_floor)
    gate_strength = _clamp((gate - gate_floor) / denom, 0.0, 1.0)
    gate_strength = gate_strength ** max(1e-6, float(cfg["gate_exponent"]))

    base = float(cfg["base_valence_conf"])
    min_thr = float(cfg["min_valence_conf"])
    threshold = base - (base - min_thr) * gate_strength
    return _clamp(threshold, min_thr, float(cfg["max_valence_conf"]))


def macro_f1_for_labels(y_true: Iterable[str], y_pred: Iterable[str], labels: Iterable[str] = CASCADE_LABELS) -> float:
    y_true = [str(v) for v in y_true]
    y_pred = [str(v) for v in y_pred]
    f1s: list[float] = []
    for label in labels:
        tp = sum(1 for y, p in zip(y_true, y_pred) if y == label and p == label)
        fp = sum(1 for y, p in zip(y_true, y_pred) if y != label and p == label)
        fn = sum(1 for y, p in zip(y_true, y_pred) if y == label and p != label)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1s.append(0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall))
    return float(sum(f1s) / max(1, len(f1s)))


def predict_with_dynamic_threshold(row: dict, config: dict | None, gate_threshold: float, fixed_threshold: float = 0.60) -> str:
    gate_prob = _finite_float(row.get("gate_prob"), 0.0)
    if gate_prob < float(gate_threshold):
        return "no_interaction"

    valence_label = str(row.get("valence_label") or "")
    if valence_label not in {"friendly", "unfriendly"}:
        return "no_interaction"

    valence_conf = _finite_float(row.get("valence_conf"), 0.0)
    threshold = dynamic_valence_conf_threshold(gate_prob, config, fixed_threshold)
    return valence_label if valence_conf >= threshold else "no_interaction"


def _score_config(rows: list[dict], config: dict, gate_threshold: float, fixed_threshold: float) -> tuple[float, list[str]]:
    y_true = [str(row.get("true") or "") for row in rows]
    y_pred = [predict_with_dynamic_threshold(row, config, gate_threshold, fixed_threshold) for row in rows]
    return macro_f1_for_labels(y_true, y_pred), y_pred


def learn_dynamic_valence_threshold(
    rows: list[dict],
    gate_threshold: float,
    fixed_threshold: float = 0.60,
) -> dict:
    if not rows:
        cfg = fixed_valence_threshold_config(fixed_threshold, gate_threshold)
        cfg["reason"] = "no_validation_rows"
        return cfg

    fixed_cfg = fixed_valence_threshold_config(fixed_threshold, gate_threshold)
    baseline_f1, baseline_pred = _score_config(rows, fixed_cfg, gate_threshold, fixed_threshold)
    best_cfg = dict(fixed_cfg)
    best_cfg["enabled"] = True
    best_cfg["method"] = "monotonic_gate_prob_grid_search"
    best_f1 = baseline_f1
    best_pred = baseline_pred

    base_values = sorted({round(x / 100.0, 2) for x in range(50, 86, 2)} | {round(float(fixed_threshold), 2)})
    min_values = sorted({round(x / 100.0, 2) for x in range(30, 72, 2)} | {round(float(fixed_threshold), 2)})
    exponents = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

    for base in base_values:
        for min_thr in min_values:
            if min_thr > base:
                continue
            for exponent in exponents:
                cfg = {
                    "schema": DYNAMIC_THRESHOLD_SCHEMA_VERSION,
                    "enabled": True,
                    "method": "monotonic_gate_prob_grid_search",
                    "gate_threshold": float(gate_threshold),
                    "base_valence_conf": float(base),
                    "min_valence_conf": float(min_thr),
                    "max_valence_conf": float(base),
                    "gate_exponent": float(exponent),
                    "fixed_baseline_valence_conf": float(fixed_threshold),
                }
                score, pred = _score_config(rows, cfg, gate_threshold, fixed_threshold)
                current_key = (
                    score,
                    -abs(base - fixed_threshold),
                    -(base - min_thr),
                    -abs(exponent - 1.0),
                )
                best_key = (
                    best_f1,
                    -abs(float(best_cfg["base_valence_conf"]) - fixed_threshold),
                    -(float(best_cfg["base_valence_conf"]) - float(best_cfg["min_valence_conf"])),
                    -abs(float(best_cfg["gate_exponent"]) - 1.0),
                )
                if current_key > best_key:
                    best_f1 = score
                    best_cfg = cfg
                    best_pred = pred

    truth = [str(row.get("true") or "") for row in rows]
    best_cfg["baseline_val_macro_f1"] = float(baseline_f1)
    best_cfg["best_val_macro_f1"] = float(best_f1)
    best_cfg["val_rows"] = int(len(rows))
    best_cfg["prediction_counts"] = dict(Counter(best_pred))
    best_cfg["true_counts"] = dict(Counter(truth))
    best_cfg["optimization_target"] = "macro_f1"
    return coerce_dynamic_threshold_config(best_cfg, fixed_threshold, gate_threshold)
