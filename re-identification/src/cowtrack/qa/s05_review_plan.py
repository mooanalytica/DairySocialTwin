"""Pure deterministic sampling for independent S05 candidate review.

The input contract matches proposal-only S05 candidate rows.  Every accepted
row must be ``decision == 'provisional'`` and ``proposed_for_review == True``;
selection is review evidence and never a confirmed link or merge.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
import random
import re
from typing import Any

import numpy as np

from cowtrack.config import ContractError


PLAN_SCHEMA_VERSION = "cowtrack.s05-review-plan.v1"

# This is a strict, intentionally small projection of LONG_CANDIDATE_EDGES_SCHEMA.
CANDIDATE_COLUMNS = (
    "candidate_id",
    "source_stable_id",
    "target_stable_id",
    "temporally_nonoverlapping",
    "appearance_present",
    "high_overlap",
    "appearance_rank_out",
    "appearance_rank_in",
    "best_margin_out",
    "best_margin_in",
    "model_probability",
    "candidate_margin",
    "provisional_threshold",
    "selected_probability_threshold",
    "selected_margin_threshold",
    "passes_provisional_threshold",
    "passes_selected_probability_gate",
    "passes_selected_margin_gate",
    "passes_selected_gate",
    "decision",
    "selected_by_solver",
    "confirmed",
    "merge_applied",
    "proposed_for_review",
)

PURPLE_STABLE_POLICY = (
    "For one selected candidate, collect stable IDs from other provisional review "
    "candidates that share its source or target; remove the focal source and target. "
    "Draw those IDs purple only when an actual S04 stable-state detection exists in "
    "the raw frame. Every other real detection remains unrelated gray. Never "
    "interpolate, predict, synthesize, relabel, or alter a box."
)


@dataclass(frozen=True)
class S05EvidenceRecord:
    candidate_id: str
    source_stable_id: int
    target_stable_id: int
    probability: float
    candidate_margin: float | None
    provisional_threshold: float
    selected_probability_threshold: float
    selected_margin_threshold: float
    appearance_rank_out: int | None
    appearance_rank_in: int | None
    high_overlap: bool
    passes_selected_probability_gate: bool
    passes_selected_margin_gate: bool
    passes_selected_gate: bool


@dataclass(frozen=True)
class S05ReviewCase:
    candidate: S05EvidenceRecord
    selection_group: str
    selection_order: int

    @property
    def suggested_filename(self) -> str:
        item = self.candidate
        margin = "missing" if item.candidate_margin is None else f"{item.candidate_margin:.4f}"
        return (
            f"{self.selection_order:03d}_{self.selection_group}_"
            f"{safe_filename_token(item.candidate_id)}_"
            f"p{item.probability:.4f}_m{margin}.mp4"
        )

    def to_manifest_record(self) -> dict[str, Any]:
        item = self.candidate
        return {
            "candidate_id": item.candidate_id,
            "source_stable_id": item.source_stable_id,
            "target_stable_id": item.target_stable_id,
            "probability": item.probability,
            "provisional_threshold": item.provisional_threshold,
            "distance_to_provisional_threshold": abs(
                item.probability - item.provisional_threshold
            ),
            "candidate_margin": item.candidate_margin,
            "selected_probability_threshold": item.selected_probability_threshold,
            "selected_margin_threshold": item.selected_margin_threshold,
            "appearance_rank_out": item.appearance_rank_out,
            "appearance_rank_in": item.appearance_rank_in,
            "high_overlap": item.high_overlap,
            "passes_selected_probability_gate": (
                item.passes_selected_probability_gate
            ),
            "passes_selected_margin_gate": item.passes_selected_margin_gate,
            "passes_selected_gate": item.passes_selected_gate,
            "candidate_semantics": "provisional_review_evidence",
            "selection_group": self.selection_group,
            "selection_order": self.selection_order,
            "suggested_filename": self.suggested_filename,
        }


@dataclass(frozen=True)
class S05ReviewPlan:
    candidates: tuple[S05EvidenceRecord, ...]
    cases: tuple[S05ReviewCase, ...]
    provisional_threshold: float
    random_seed: int
    maximum_cases: int
    high_score_count: int
    threshold_near_count: int
    ambiguous_count: int
    random_count: int
    threshold_probability_band: float
    ambiguity_margin_upper: float
    ambiguous_mutual_rank_above: int

    def competing_stable_ids(self, case: S05ReviewCase) -> tuple[int, ...]:
        """Return real stable IDs from candidates competing with the focal edge."""

        focal = case.candidate
        result: set[int] = set()
        for candidate in self.candidates:
            if candidate.candidate_id == focal.candidate_id:
                continue
            if (
                candidate.source_stable_id in {
                    focal.source_stable_id,
                    focal.target_stable_id,
                }
                or candidate.target_stable_id
                in {focal.source_stable_id, focal.target_stable_id}
            ):
                result.add(candidate.source_stable_id)
                result.add(candidate.target_stable_id)
        result.discard(focal.source_stable_id)
        result.discard(focal.target_stable_id)
        return tuple(sorted(result))

    def to_manifest_payload(self) -> dict[str, Any]:
        return {
            "schema_version": PLAN_SCHEMA_VERSION,
            "candidate_semantics": "provisional_review_evidence_only",
            "selection_is_read_only": True,
            "automatic_merge_allowed": False,
            "solver_used": False,
            "frozen_provisional_threshold": self.provisional_threshold,
            "selection_policy": {
                "maximum_cases": self.maximum_cases,
                "if_available_at_or_below_maximum": (
                    "select_all_probability_desc_then_candidate_id"
                ),
                "deduplication_priority": [
                    "high_score",
                    "threshold_near",
                    "ambiguous",
                    "random",
                ],
                "groups": [
                    {
                        "group": "high_score",
                        "maximum_count": self.high_score_count,
                        "order": "probability_desc_then_candidate_id",
                    },
                    {
                        "group": "threshold_near",
                        "maximum_count": self.threshold_near_count,
                        "absolute_probability_band": self.threshold_probability_band,
                        "order": "absolute_distance_then_candidate_id",
                    },
                    {
                        "group": "ambiguous",
                        "maximum_count": self.ambiguous_count,
                        "eligible_when": {
                            "candidate_margin_missing_or_at_most": (
                                self.ambiguity_margin_upper
                            ),
                            "or_mutual_rank_missing_or_above": (
                                self.ambiguous_mutual_rank_above
                            ),
                            "or_high_overlap": True,
                        },
                        "order": (
                            "missing_margin_then_high_overlap_then_margin_asc_"
                            "then_worst_rank_desc_then_candidate_id"
                        ),
                    },
                    {
                        "group": "random",
                        "maximum_count": self.random_count,
                        "population": "remaining_candidate_id_ascending",
                        "algorithm": (
                            "python_random.Random(seed).sample_without_replacement"
                        ),
                        "seed": self.random_seed,
                    },
                ],
                "duplicates_allowed": False,
            },
            "render_window": {
                "source_tail_sec": 2.0,
                "target_head_sec": 2.0,
                "coordinates": "original_raw_frame_before_whole_frame_resize",
                "autorotation": False,
            },
            "visual_contract": {
                "text": False,
                "ids": False,
                "scores": False,
                "legend": False,
                "keypoints": False,
                "trajectory": False,
                "synthetic_boxes": False,
                "box_geometry_modified": False,
                "colors": {
                    "unrelated": "gray",
                    "source": "yellow",
                    "target": "green",
                    "intermediate_or_competing_stable_state": "purple",
                },
                "purple_stable_policy": PURPLE_STABLE_POLICY,
            },
            "summary": {
                "num_available_candidates": len(self.candidates),
                "num_selected_cases": len(self.cases),
                "num_high_score": sum(
                    case.selection_group == "high_score" for case in self.cases
                ),
                "num_threshold_near": sum(
                    case.selection_group == "threshold_near" for case in self.cases
                ),
                "num_ambiguous": sum(
                    case.selection_group == "ambiguous" for case in self.cases
                ),
                "num_random": sum(
                    case.selection_group == "random" for case in self.cases
                ),
                "num_all_available": sum(
                    case.selection_group == "all_available" for case in self.cases
                ),
            },
            "cases": [case.to_manifest_record() for case in self.cases],
        }


def safe_filename_token(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", value):
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"candidate-{digest}"


def _column(
    columns: Mapping[str, Sequence[Any] | np.ndarray], name: str, count: int
) -> np.ndarray:
    if name not in columns:
        raise ContractError(f"S05 review candidates missing column: {name}")
    result = np.asarray(columns[name], dtype=object)
    if result.ndim != 1 or len(result) != count:
        raise ContractError(
            f"S05 review candidates.{name} must be one-dimensional with {count} rows"
        )
    return result


def _string(value: Any, name: str, row: int) -> str:
    if isinstance(value, np.str_):
        value = str(value)
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ContractError(f"S05 review {name} is invalid at row {row}")
    return value


def _integer_or_none(
    value: Any, name: str, row: int, *, minimum: int
) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ContractError(f"S05 review {name} must be an integer or null at row {row}")
    result = int(value)
    if result < minimum:
        raise ContractError(
            f"S05 review {name} must be >= {minimum} at row {row}"
        )
    return result


def _integer(value: Any, name: str, row: int, *, minimum: int) -> int:
    result = _integer_or_none(value, name, row, minimum=minimum)
    if result is None:
        raise ContractError(f"S05 review {name} cannot be null at row {row}")
    return result


def _number_or_none(value: Any, name: str, row: int) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ContractError(f"S05 review {name} must be numeric or null at row {row}")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"S05 review {name} must be finite at row {row}")
    return result


def _boolean(value: Any, name: str, row: int) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ContractError(f"S05 review {name} must be boolean at row {row}")
    return bool(value)


def parse_s05_evidence_records(
    columns: Mapping[str, Sequence[Any] | np.ndarray],
) -> tuple[S05EvidenceRecord, ...]:
    """Parse only proposal-only, provisional rows selected for review."""

    if not isinstance(columns, Mapping) or "candidate_id" not in columns:
        raise ContractError("S05 review candidates missing column: candidate_id")
    ids = np.asarray(columns["candidate_id"], dtype=object)
    if ids.ndim != 1:
        raise ContractError(
            "S05 review candidates.candidate_id must be one-dimensional"
        )
    arrays = {name: _column(columns, name, len(ids)) for name in CANDIDATE_COLUMNS}
    records: list[S05EvidenceRecord] = []
    for row in range(len(ids)):
        candidate_id = _string(arrays["candidate_id"][row], "candidate_id", row)
        source = _integer(
            arrays["source_stable_id"][row], "source_stable_id", row, minimum=0
        )
        target = _integer(
            arrays["target_stable_id"][row], "target_stable_id", row, minimum=0
        )
        if source == target:
            raise ContractError(f"S05 review candidate self-link at row {row}")
        if not _boolean(
            arrays["temporally_nonoverlapping"][row],
            "temporally_nonoverlapping",
            row,
        ):
            raise ContractError(f"S05 review candidate temporally overlaps at row {row}")
        if not _boolean(arrays["appearance_present"][row], "appearance_present", row):
            raise ContractError(f"S05 review candidate lacks appearance at row {row}")
        high_overlap = _boolean(arrays["high_overlap"][row], "high_overlap", row)
        rank_out = _integer_or_none(
            arrays["appearance_rank_out"][row],
            "appearance_rank_out",
            row,
            minimum=1,
        )
        rank_in = _integer_or_none(
            arrays["appearance_rank_in"][row],
            "appearance_rank_in",
            row,
            minimum=1,
        )
        margin_out = _number_or_none(
            arrays["best_margin_out"][row], "best_margin_out", row
        )
        margin_in = _number_or_none(
            arrays["best_margin_in"][row], "best_margin_in", row
        )
        margin = (
            min(margin_out, margin_in)
            if margin_out is not None and margin_in is not None
            else None
        )
        recorded_margin = _number_or_none(
            arrays["candidate_margin"][row], "candidate_margin", row
        )
        if (recorded_margin is None) != (margin is None) or (
            recorded_margin is not None
            and margin is not None
            and not math.isclose(
                recorded_margin, margin, rel_tol=0.0, abs_tol=1e-12
            )
        ):
            raise ContractError(
                f"S05 review candidate_margin differs from directional margins at row {row}"
            )
        probability = _number_or_none(
            arrays["model_probability"][row], "model_probability", row
        )
        if probability is None or not 0.0 <= probability <= 1.0:
            raise ContractError(
                f"S05 review model_probability must be finite in [0, 1] at row {row}"
            )
        provisional_threshold = _number_or_none(
            arrays["provisional_threshold"][row], "provisional_threshold", row
        )
        selected_probability_threshold = _number_or_none(
            arrays["selected_probability_threshold"][row],
            "selected_probability_threshold",
            row,
        )
        selected_margin_threshold = _number_or_none(
            arrays["selected_margin_threshold"][row],
            "selected_margin_threshold",
            row,
        )
        if (
            provisional_threshold is None
            or selected_probability_threshold is None
            or selected_margin_threshold is None
            or not 0.0 <= provisional_threshold <= 1.0
            or not 0.0 <= selected_probability_threshold <= 1.0
        ):
            raise ContractError(f"S05 review frozen thresholds are invalid at row {row}")
        passes_provisional = _boolean(
            arrays["passes_provisional_threshold"][row],
            "passes_provisional_threshold",
            row,
        )
        probability_gate = _boolean(
            arrays["passes_selected_probability_gate"][row],
            "passes_selected_probability_gate",
            row,
        )
        margin_gate = _boolean(
            arrays["passes_selected_margin_gate"][row],
            "passes_selected_margin_gate",
            row,
        )
        selected_gate = _boolean(
            arrays["passes_selected_gate"][row], "passes_selected_gate", row
        )
        if passes_provisional != (probability >= provisional_threshold):
            raise ContractError(
                f"S05 review provisional-threshold evidence is inconsistent at row {row}"
            )
        if probability_gate != (probability >= selected_probability_threshold):
            raise ContractError(
                f"S05 review selected probability evidence is inconsistent at row {row}"
            )
        if margin_gate != (
            recorded_margin is not None
            and recorded_margin >= selected_margin_threshold
        ):
            raise ContractError(
                f"S05 review selected margin evidence is inconsistent at row {row}"
            )
        if selected_gate != (probability_gate and margin_gate and not high_overlap):
            raise ContractError(
                f"S05 review selected-gate evidence is inconsistent at row {row}"
            )
        decision = _string(arrays["decision"][row], "decision", row)
        proposed = _boolean(
            arrays["proposed_for_review"][row], "proposed_for_review", row
        )
        if decision != "provisional" or not proposed or not passes_provisional:
            raise ContractError(
                "S05 review accepts only decision='provisional', "
                "proposed_for_review=true rows that pass the provisional threshold"
            )
        for field in ("selected_by_solver", "confirmed", "merge_applied"):
            if _boolean(arrays[field][row], field, row):
                raise ContractError(
                    f"S05 proposal-only review requires {field}=false at row {row}"
                )
        records.append(
            S05EvidenceRecord(
                candidate_id=candidate_id,
                source_stable_id=source,
                target_stable_id=target,
                probability=probability,
                candidate_margin=recorded_margin,
                provisional_threshold=provisional_threshold,
                selected_probability_threshold=selected_probability_threshold,
                selected_margin_threshold=selected_margin_threshold,
                appearance_rank_out=rank_out,
                appearance_rank_in=rank_in,
                high_overlap=high_overlap,
                passes_selected_probability_gate=probability_gate,
                passes_selected_margin_gate=margin_gate,
                passes_selected_gate=selected_gate,
            )
        )
    if len({item.candidate_id for item in records}) != len(records):
        raise ContractError("S05 review candidate_id values must be unique")
    directed = [(item.source_stable_id, item.target_stable_id) for item in records]
    if len(set(directed)) != len(records):
        raise ContractError("S05 review directed stable candidate pairs must be unique")
    for name, values in (
        ("provisional", {item.provisional_threshold.hex() for item in records}),
        (
            "selected probability",
            {item.selected_probability_threshold.hex() for item in records},
        ),
        (
            "selected margin",
            {item.selected_margin_threshold.hex() for item in records},
        ),
    ):
        if len(values) > 1:
            raise ContractError(f"S05 review candidates use different {name} thresholds")
    return tuple(records)


def _take(
    candidates: Sequence[S05EvidenceRecord], count: int, chosen: set[str]
) -> list[S05EvidenceRecord]:
    output: list[S05EvidenceRecord] = []
    for candidate in candidates:
        if candidate.candidate_id in chosen:
            continue
        output.append(candidate)
        chosen.add(candidate.candidate_id)
        if len(output) == count:
            break
    return output


def _is_ambiguous(
    item: S05EvidenceRecord, *, margin_upper: float, rank_above: int
) -> bool:
    return bool(
        item.candidate_margin is None
        or item.candidate_margin <= margin_upper
        or item.appearance_rank_out is None
        or item.appearance_rank_in is None
        or item.appearance_rank_out > rank_above
        or item.appearance_rank_in > rank_above
        or item.high_overlap
    )


def _ambiguity_key(item: S05EvidenceRecord) -> tuple[Any, ...]:
    margin = -math.inf if item.candidate_margin is None else item.candidate_margin
    ranks = [
        rank
        for rank in (item.appearance_rank_out, item.appearance_rank_in)
        if rank is not None
    ]
    worst_rank = max(ranks) if ranks else math.inf
    return (
        item.candidate_margin is not None,
        not item.high_overlap,
        margin,
        -worst_rank,
        item.candidate_id,
    )


def build_s05_review_plan(
    candidates: Mapping[str, Sequence[Any] | np.ndarray],
    *,
    provisional_threshold: float,
    random_seed: int,
    maximum_cases: int = 50,
    high_score_count: int = 15,
    threshold_near_count: int = 15,
    ambiguous_count: int = 10,
    random_count: int = 10,
    threshold_probability_band: float = 0.03,
    ambiguity_margin_upper: float = 0.08,
    ambiguous_mutual_rank_above: int = 1,
) -> S05ReviewPlan:
    """Select bounded high/near-threshold/ambiguous/random evidence strata."""

    integers = {
        "random_seed": (random_seed, None),
        "maximum_cases": (maximum_cases, 1),
        "high_score_count": (high_score_count, 0),
        "threshold_near_count": (threshold_near_count, 0),
        "ambiguous_count": (ambiguous_count, 0),
        "random_count": (random_count, 0),
        "ambiguous_mutual_rank_above": (ambiguous_mutual_rank_above, 1),
    }
    for name, (value, minimum) in integers.items():
        if isinstance(value, bool) or not isinstance(value, int) or (
            minimum is not None and value < minimum
        ):
            bound = "an integer" if minimum is None else f"an integer >= {minimum}"
            raise ContractError(f"S05 review {name} must be {bound}")
    if sum(
        (high_score_count, threshold_near_count, ambiguous_count, random_count)
    ) != maximum_cases:
        raise ContractError("S05 review selection quotas must sum to maximum_cases")
    for name, value in (
        ("provisional_threshold", provisional_threshold),
        ("threshold_probability_band", threshold_probability_band),
        ("ambiguity_margin_upper", ambiguity_margin_upper),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(
            float(value)
        ):
            raise ContractError(f"S05 review {name} must be finite")
    if not 0.0 <= float(provisional_threshold) <= 1.0:
        raise ContractError("S05 review provisional_threshold must be in [0, 1]")
    if not 0.0 < float(threshold_probability_band) <= 1.0:
        raise ContractError("S05 review threshold_probability_band must be in (0, 1]")

    records = parse_s05_evidence_records(candidates)
    for item in records:
        if not math.isclose(
            item.provisional_threshold,
            float(provisional_threshold),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ContractError(
                "S05 review candidate threshold differs from the supplied frozen threshold"
            )
    grouped: list[tuple[S05EvidenceRecord, str]]
    if len(records) <= maximum_cases:
        grouped = [
            (candidate, "all_available")
            for candidate in sorted(
                records, key=lambda item: (-item.probability, item.candidate_id)
            )
        ]
    else:
        chosen: set[str] = set()
        high = _take(
            sorted(records, key=lambda item: (-item.probability, item.candidate_id)),
            high_score_count,
            chosen,
        )
        near = _take(
            sorted(
                (
                    item
                    for item in records
                    if abs(item.probability - float(provisional_threshold))
                    <= float(threshold_probability_band)
                ),
                key=lambda item: (
                    abs(item.probability - float(provisional_threshold)),
                    item.candidate_id,
                ),
            ),
            threshold_near_count,
            chosen,
        )
        ambiguous = _take(
            sorted(
                (
                    item
                    for item in records
                    if _is_ambiguous(
                        item,
                        margin_upper=float(ambiguity_margin_upper),
                        rank_above=ambiguous_mutual_rank_above,
                    )
                ),
                key=_ambiguity_key,
            ),
            ambiguous_count,
            chosen,
        )
        remainder = sorted(
            (item for item in records if item.candidate_id not in chosen),
            key=lambda item: item.candidate_id,
        )
        random_rows = random.Random(random_seed).sample(
            remainder, min(random_count, len(remainder))
        )
        grouped = (
            [(item, "high_score") for item in high]
            + [(item, "threshold_near") for item in near]
            + [(item, "ambiguous") for item in ambiguous]
            + [(item, "random") for item in random_rows]
        )
    cases = tuple(
        S05ReviewCase(candidate, group, order)
        for order, (candidate, group) in enumerate(grouped, start=1)
    )
    selected_ids = [case.candidate.candidate_id for case in cases]
    if len(selected_ids) != len(set(selected_ids)):
        raise ContractError("internal error: duplicate S05 review candidate selected")
    if len(cases) > maximum_cases:
        raise ContractError("internal error: S05 review exceeded its fixed maximum")
    if len(records) <= maximum_cases and len(cases) != len(records):
        raise ContractError("internal error: S05 review did not select all small input")
    return S05ReviewPlan(
        candidates=records,
        cases=cases,
        provisional_threshold=float(provisional_threshold),
        random_seed=random_seed,
        maximum_cases=maximum_cases,
        high_score_count=high_score_count,
        threshold_near_count=threshold_near_count,
        ambiguous_count=ambiguous_count,
        random_count=random_count,
        threshold_probability_band=float(threshold_probability_band),
        ambiguity_margin_upper=float(ambiguity_margin_upper),
        ambiguous_mutual_rank_above=ambiguous_mutual_rank_above,
    )


__all__ = [
    "CANDIDATE_COLUMNS",
    "PLAN_SCHEMA_VERSION",
    "PURPLE_STABLE_POLICY",
    "S05EvidenceRecord",
    "S05ReviewCase",
    "S05ReviewPlan",
    "build_s05_review_plan",
    "parse_s05_evidence_records",
    "safe_filename_token",
]
