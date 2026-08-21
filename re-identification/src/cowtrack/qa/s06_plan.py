"""Pure deterministic QA planning helpers for forced-appearance S06.

This module intentionally reads no artifacts and decodes no video.  It works
only from already validated rows supplied by the S06 stage.  In particular,
the directional ranks and margins below describe the *persisted* forced S05
candidate graph; they do not claim to rank every temporally compatible pair.
Appearance cosine and cosine margin are never probabilities.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from cowtrack.config import ContractError


PLAN_SCHEMA_VERSION = "cowtrack.s06-qa-plan.v1"
FORCED_ID_STATUS = "forced_provisional"
FORCED_AUTHORIZATION_BASIS = "operator_forced_appearance_exact_62"
CONTACT_SHEET_ROLES = (
    "start",
    "middle",
    "end",
    "weakest_link_source",
    "weakest_link_target",
    "longest_gap_source",
    "longest_gap_target",
)

_CANDIDATE_REQUIRED_COLUMNS = (
    "candidate_id",
    "source_stable_id",
    "target_stable_id",
    "source_end_clip_id",
    "target_start_clip_id",
    "source_end_global_frame",
    "target_start_global_frame",
    "source_end_time_sec",
    "target_start_time_sec",
    "temporal_gap_sec",
    "strictly_nonoverlapping",
    "appearance_cosine",
    "source_evidence_grade",
    "target_evidence_grade",
    "selected_by_source_topk",
    "selected_by_target_topk",
    "temporal_backbone",
    "prior_global_link",
    "selected_by_solver",
    "global_link_id",
    "authorization_basis",
    "id_status",
)


@dataclass(frozen=True)
class CandidateMetric:
    """Directional evidence for one edge in the persisted S05 graph."""

    candidate_id: str
    source_stable_id: int
    target_stable_id: int
    source_end_clip_id: str
    target_start_clip_id: str
    source_end_global_frame: int
    target_start_global_frame: int
    source_end_time_sec: float
    target_start_time_sec: float
    temporal_gap_sec: float
    appearance_cosine: float
    source_evidence_grade: str
    target_evidence_grade: str
    selected_by_source_topk: bool
    selected_by_target_topk: bool
    temporal_backbone: bool
    prior_global_link: bool
    selected_by_solver: bool
    global_link_id: str | None
    authorization_basis: str
    id_status: str
    outgoing_rank: int
    incoming_rank: int
    outgoing_margin: float | None
    incoming_margin: float | None
    conservative_margin: float | None

    def to_record(self) -> dict[str, Any]:
        """Return names suitable for the S06 CSV/HTML audit records."""

        return {
            "candidate_id": self.candidate_id,
            "source_stable_id": self.source_stable_id,
            "target_stable_id": self.target_stable_id,
            "source_end_clip_id": self.source_end_clip_id,
            "target_start_clip_id": self.target_start_clip_id,
            "source_end_global_frame": self.source_end_global_frame,
            "target_start_global_frame": self.target_start_global_frame,
            "source_end_time_sec": self.source_end_time_sec,
            "target_start_time_sec": self.target_start_time_sec,
            "temporal_gap_sec": self.temporal_gap_sec,
            "appearance_cosine": self.appearance_cosine,
            "source_evidence_grade": self.source_evidence_grade,
            "target_evidence_grade": self.target_evidence_grade,
            "selected_by_source_topk": self.selected_by_source_topk,
            "selected_by_target_topk": self.selected_by_target_topk,
            "temporal_backbone": self.temporal_backbone,
            "prior_global_link": self.prior_global_link,
            "selected_by_solver": self.selected_by_solver,
            "global_link_id": self.global_link_id,
            "authorization_basis": self.authorization_basis,
            "id_status": self.id_status,
            "persisted_graph_outgoing_rank": self.outgoing_rank,
            "persisted_graph_incoming_rank": self.incoming_rank,
            "outgoing_cosine_margin_to_best_alternative": self.outgoing_margin,
            "incoming_cosine_margin_to_best_alternative": self.incoming_margin,
            "conservative_cosine_margin": self.conservative_margin,
            "rank_scope": "persisted_candidate_graph",
            "score_semantics": "cosine_not_probability",
        }

    @property
    def source_candidate_rank(self) -> int:
        return self.outgoing_rank

    @property
    def target_candidate_rank(self) -> int:
        return self.incoming_rank

    @property
    def source_second_best_cosine_margin(self) -> float | None:
        return self.outgoing_margin

    @property
    def target_second_best_cosine_margin(self) -> float | None:
        return self.incoming_margin


@dataclass(frozen=True)
class CropSource:
    """One real S00 detection crop, in raw 3840x2160 coordinates."""

    det_id: int
    clip_id: str
    local_frame: int
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class DetectionCrop:
    """Validated identity/time metadata for a real, valid detection."""

    det_id: int
    clip_id: str
    local_frame: int
    global_frame: int
    global_time_sec: float
    stable_id: int
    global_track_id: int
    display_global_id: str
    id_status: str
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def source(self) -> CropSource:
        return CropSource(
            det_id=self.det_id,
            clip_id=self.clip_id,
            local_frame=self.local_frame,
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
        )


@dataclass(frozen=True, order=True)
class CropUse:
    """One artifact/role that consumes a unique decoded crop."""

    consumer_id: str
    role: str
    label: str


@dataclass(frozen=True)
class CropRequest:
    source: CropSource
    use: CropUse


@dataclass(frozen=True)
class IndexedCropRequest:
    source: CropSource
    uses: tuple[CropUse, ...]


@dataclass(frozen=True)
class DecodeFrameRequest:
    clip_id: str
    local_frame: int
    crops: tuple[IndexedCropRequest, ...]


@dataclass(frozen=True)
class CropRequestIndex:
    """Canonical single-pass video decode and crop-deduplication index."""

    frames: tuple[DecodeFrameRequest, ...]

    @property
    def num_decode_frames(self) -> int:
        return len(self.frames)

    @property
    def num_unique_crops(self) -> int:
        return sum(len(frame.crops) for frame in self.frames)

    @property
    def num_uses(self) -> int:
        return sum(
            len(crop.uses) for frame in self.frames for crop in frame.crops
        )

    def uses_for_det_id(self, det_id: int) -> tuple[CropUse, ...]:
        for frame in self.frames:
            for crop in frame.crops:
                if crop.source.det_id == det_id:
                    return crop.uses
        raise KeyError(det_id)


@dataclass(frozen=True)
class ContactSheetSlot:
    """One fixed overview slot; ``source=None`` is an intentional blank."""

    role: str
    source: CropSource | None
    global_time_sec: float | None
    annotation: str
    deduplicated_to_role: str | None = None


@dataclass(frozen=True)
class ContactSheetPlan:
    global_track_id: int
    display_global_id: str
    id_status: str
    slots: tuple[ContactSheetSlot, ...]
    weakest_link: CandidateMetric | None
    longest_gap_link: CandidateMetric | None

    @property
    def items(self) -> tuple[ContactSheetSlot, ...]:
        """Return only real, unique crops; blank slots remain in ``slots``."""

        return tuple(slot for slot in self.slots if slot.source is not None)

    def crop_requests(self) -> tuple[CropRequest, ...]:
        consumer = f"contact_sheet:{self.display_global_id}"
        return tuple(
            CropRequest(
                source=slot.source,
                use=CropUse(
                    consumer_id=consumer,
                    role=slot.role,
                    label=_slot_label(slot),
                ),
            )
            for slot in self.slots
            if slot.source is not None
        )


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if not _is_int(value):
        raise ContractError(f"{label} must be an integer")
    result = int(value)
    if result < minimum:
        raise ContractError(f"{label} must be >= {minimum}")
    return result


def _signed_int64(value: Any, *, label: str) -> int:
    if not _is_int(value):
        raise ContractError(f"{label} must be an integer")
    result = int(value)
    if not -(1 << 63) <= result < (1 << 63):
        raise ContractError(f"{label} must fit signed int64")
    return result


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ContractError(f"{label} must be finite numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{label} must be finite numeric")
    return result


def _string(value: Any, *, label: str) -> str:
    if isinstance(value, np.str_):
        value = str(value)
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ContractError(f"{label} must be a nonblank string")
    return value


def _boolean(value: Any, *, label: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ContractError(f"{label} must be boolean")
    return bool(value)


def _row_records(
    values: Mapping[str, Sequence[Any] | np.ndarray] | Sequence[Mapping[str, Any]],
    *,
    required: Sequence[str],
    label: str,
) -> tuple[dict[str, Any], ...]:
    """Normalize columnar or row-oriented data without accepting partial rows."""

    if isinstance(values, Mapping):
        missing = sorted(set(required) - set(values))
        if missing:
            raise ContractError(f"{label} missing columns: {', '.join(missing)}")
        columns: dict[str, np.ndarray] = {}
        count: int | None = None
        for name in required:
            array = np.asarray(values[name], dtype=object)
            if array.ndim != 1:
                raise ContractError(f"{label}.{name} must be one-dimensional")
            if count is None:
                count = len(array)
            elif len(array) != count:
                raise ContractError(f"{label} columns have different row counts")
            columns[name] = array
        return tuple(
            {name: columns[name][row] for name in required}
            for row in range(0 if count is None else count)
        )
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ContractError(f"{label} must be columnar or a sequence of rows")
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(values):
        if not isinstance(source, Mapping):
            raise ContractError(f"{label}[{index}] must be an object")
        missing = sorted(set(required) - set(source))
        if missing:
            raise ContractError(
                f"{label}[{index}] missing columns: {', '.join(missing)}"
            )
        rows.append({name: source[name] for name in required})
    return tuple(rows)


def _parse_candidate_rows(
    candidates: Mapping[str, Sequence[Any] | np.ndarray]
    | Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    rows = _row_records(
        candidates,
        required=_CANDIDATE_REQUIRED_COLUMNS,
        label="S06 persisted candidates",
    )
    parsed: list[dict[str, Any]] = []
    for index, source in enumerate(rows):
        prefix = f"S06 persisted candidates[{index}]"
        candidate_id = _string(source["candidate_id"], label=f"{prefix}.candidate_id")
        source_id = _integer(
            source["source_stable_id"], label=f"{prefix}.source_stable_id"
        )
        target_id = _integer(
            source["target_stable_id"], label=f"{prefix}.target_stable_id"
        )
        if source_id == target_id:
            raise ContractError(f"{prefix} is a stable self-link")
        source_clip = _string(
            source["source_end_clip_id"], label=f"{prefix}.source_end_clip_id"
        )
        target_clip = _string(
            source["target_start_clip_id"], label=f"{prefix}.target_start_clip_id"
        )
        source_frame = _integer(
            source["source_end_global_frame"],
            label=f"{prefix}.source_end_global_frame",
        )
        target_frame = _integer(
            source["target_start_global_frame"],
            label=f"{prefix}.target_start_global_frame",
        )
        source_time = _finite(
            source["source_end_time_sec"], label=f"{prefix}.source_end_time_sec"
        )
        target_time = _finite(
            source["target_start_time_sec"], label=f"{prefix}.target_start_time_sec"
        )
        gap = _finite(source["temporal_gap_sec"], label=f"{prefix}.temporal_gap_sec")
        if not _boolean(
            source["strictly_nonoverlapping"],
            label=f"{prefix}.strictly_nonoverlapping",
        ):
            raise ContractError(f"{prefix} must be strictly nonoverlapping")
        if source_frame >= target_frame or source_time >= target_time or gap <= 0.0:
            raise ContractError(f"{prefix} temporal endpoints are not strictly ordered")
        if not math.isclose(
            gap, target_time - source_time, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ContractError(f"{prefix} temporal_gap_sec differs from endpoints")
        cosine = _finite(
            source["appearance_cosine"], label=f"{prefix}.appearance_cosine"
        )
        if not -1.0 <= cosine <= 1.0:
            raise ContractError(f"{prefix}.appearance_cosine must be in [-1, 1]")
        selected = _boolean(
            source["selected_by_solver"], label=f"{prefix}.selected_by_solver"
        )
        link_value = source["global_link_id"]
        if link_value is None:
            link_id = None
        else:
            link_id = _string(link_value, label=f"{prefix}.global_link_id")
        status = _string(source["id_status"], label=f"{prefix}.id_status")
        authorization = _string(
            source["authorization_basis"], label=f"{prefix}.authorization_basis"
        )
        if authorization != FORCED_AUTHORIZATION_BASIS:
            raise ContractError(
                f"{prefix}.authorization_basis must be {FORCED_AUTHORIZATION_BASIS}"
            )
        if selected != (link_id is not None):
            raise ContractError(f"{prefix} selected/global_link_id state differs")
        if selected and status != FORCED_ID_STATUS:
            raise ContractError(
                f"{prefix} selected edge must remain {FORCED_ID_STATUS}"
            )
        if not selected and status != "candidate_only":
            raise ContractError(f"{prefix} unselected edge must be candidate_only")
        parsed.append(
            {
                "candidate_id": candidate_id,
                "source_stable_id": source_id,
                "target_stable_id": target_id,
                "source_end_clip_id": source_clip,
                "target_start_clip_id": target_clip,
                "source_end_global_frame": source_frame,
                "target_start_global_frame": target_frame,
                "source_end_time_sec": source_time,
                "target_start_time_sec": target_time,
                "temporal_gap_sec": gap,
                "appearance_cosine": cosine,
                "source_evidence_grade": _string(
                    source["source_evidence_grade"],
                    label=f"{prefix}.source_evidence_grade",
                ),
                "target_evidence_grade": _string(
                    source["target_evidence_grade"],
                    label=f"{prefix}.target_evidence_grade",
                ),
                "selected_by_source_topk": _boolean(
                    source["selected_by_source_topk"],
                    label=f"{prefix}.selected_by_source_topk",
                ),
                "selected_by_target_topk": _boolean(
                    source["selected_by_target_topk"],
                    label=f"{prefix}.selected_by_target_topk",
                ),
                "temporal_backbone": _boolean(
                    source["temporal_backbone"],
                    label=f"{prefix}.temporal_backbone",
                ),
                "prior_global_link": _boolean(
                    source["prior_global_link"],
                    label=f"{prefix}.prior_global_link",
                ),
                "selected_by_solver": selected,
                "global_link_id": link_id,
                "authorization_basis": authorization,
                "id_status": status,
            }
        )
    ids = [str(row["candidate_id"]) for row in parsed]
    pairs = [
        (int(row["source_stable_id"]), int(row["target_stable_id"]))
        for row in parsed
    ]
    if len(ids) != len(set(ids)):
        raise ContractError("S06 persisted candidate_id values must be unique")
    if len(pairs) != len(set(pairs)):
        raise ContractError("S06 persisted directed candidate pairs must be unique")
    return tuple(parsed)


def _directional_values(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_field: str,
    tie_field: str,
) -> tuple[dict[str, int], dict[str, float | None]]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row[group_field])].append(row)
    ranks: dict[str, int] = {}
    margins: dict[str, float | None] = {}
    for group in grouped.values():
        ordered = sorted(
            group,
            key=lambda row: (
                -float(row["appearance_cosine"]),
                int(row[tie_field]),
                str(row["candidate_id"]),
            ),
        )
        for rank, row in enumerate(ordered, start=1):
            candidate_id = str(row["candidate_id"])
            ranks[candidate_id] = rank
            if len(ordered) == 1:
                margins[candidate_id] = None
            else:
                best_alternative = ordered[1] if rank == 1 else ordered[0]
                margins[candidate_id] = float(row["appearance_cosine"]) - float(
                    best_alternative["appearance_cosine"]
                )
    return ranks, margins


def compute_directional_candidate_metrics(
    candidates: Mapping[str, Sequence[Any] | np.ndarray]
    | Sequence[Mapping[str, Any]],
) -> tuple[CandidateMetric, ...]:
    """Rank every persisted candidate in both directions, deterministically.

    Outgoing candidates share ``source_stable_id`` and use target stable ID as
    the tie-break. Incoming candidates share ``target_stable_id`` and use
    source stable ID as the tie-break. A margin is this candidate's cosine
    minus its best alternative's cosine, so a non-best selected edge can have
    a negative margin. A singleton direction has a null margin.
    """

    rows = _parse_candidate_rows(candidates)
    outgoing_ranks, outgoing_margins = _directional_values(
        rows,
        group_field="source_stable_id",
        tie_field="target_stable_id",
    )
    incoming_ranks, incoming_margins = _directional_values(
        rows,
        group_field="target_stable_id",
        tie_field="source_stable_id",
    )
    output: list[CandidateMetric] = []
    for row in rows:
        candidate_id = str(row["candidate_id"])
        out_margin = outgoing_margins[candidate_id]
        in_margin = incoming_margins[candidate_id]
        conservative = (
            min(out_margin, in_margin)
            if out_margin is not None and in_margin is not None
            else None
        )
        output.append(
            CandidateMetric(
                **row,
                outgoing_rank=outgoing_ranks[candidate_id],
                incoming_rank=incoming_ranks[candidate_id],
                outgoing_margin=out_margin,
                incoming_margin=in_margin,
                conservative_margin=conservative,
            )
        )
    return tuple(
        sorted(
            output,
            key=lambda item: (
                item.source_stable_id,
                item.target_stable_id,
                item.candidate_id,
            ),
        )
    )


def select_low_confidence_links(
    metrics: Sequence[CandidateMetric], *, threshold: float = 0.5
) -> tuple[CandidateMetric, ...]:
    """Return selected links below a cosine threshold in canonical QA order."""

    threshold_value = _finite(threshold, label="S06 low-confidence threshold")
    if not -1.0 <= threshold_value <= 1.0:
        raise ContractError("S06 low-confidence threshold must be in [-1, 1]")
    if not isinstance(metrics, Sequence):
        raise ContractError("S06 candidate metrics must be a sequence")
    seen: set[str] = set()
    selected: list[CandidateMetric] = []
    for index, metric in enumerate(metrics):
        if not isinstance(metric, CandidateMetric):
            raise ContractError(f"S06 candidate metrics[{index}] has invalid type")
        if metric.candidate_id in seen:
            raise ContractError("S06 candidate metrics contain duplicate candidate_id")
        seen.add(metric.candidate_id)
        if metric.selected_by_solver and metric.appearance_cosine < threshold_value:
            selected.append(metric)
    return tuple(
        sorted(
            selected,
            key=lambda item: (
                item.appearance_cosine,
                item.candidate_id,
                item.source_stable_id,
                item.target_stable_id,
            ),
        )
    )


def _validate_detection(item: DetectionCrop, *, index: int) -> DetectionCrop:
    if not isinstance(item, DetectionCrop):
        raise ContractError(f"S06 detections[{index}] has invalid type")
    prefix = f"S06 detections[{index}]"
    _signed_int64(item.det_id, label=f"{prefix}.det_id")
    _string(item.clip_id, label=f"{prefix}.clip_id")
    _integer(item.local_frame, label=f"{prefix}.local_frame")
    _integer(item.global_frame, label=f"{prefix}.global_frame")
    _finite(item.global_time_sec, label=f"{prefix}.global_time_sec")
    _integer(item.stable_id, label=f"{prefix}.stable_id")
    global_id = _integer(item.global_track_id, label=f"{prefix}.global_track_id")
    expected_display = f"G{global_id + 1:04d}"
    if item.display_global_id != expected_display:
        raise ContractError(
            f"{prefix}.display_global_id must be {expected_display!r}"
        )
    if item.id_status != FORCED_ID_STATUS:
        raise ContractError(f"{prefix}.id_status must remain {FORCED_ID_STATUS}")
    coordinates = tuple(
        _finite(value, label=f"{prefix}.{name}")
        for name, value in zip(
            ("x1", "y1", "x2", "y2"),
            (item.x1, item.y1, item.x2, item.y2),
            strict=True,
        )
    )
    x1, y1, x2, y2 = coordinates
    if not (0.0 <= x1 < x2 <= 3840.0 and 0.0 <= y1 < y2 <= 2160.0):
        raise ContractError(f"{prefix} bbox is outside raw 3840x2160 geometry")
    return item


def _detection_order(item: DetectionCrop) -> tuple[Any, ...]:
    return (
        item.global_time_sec,
        item.global_frame,
        item.clip_id,
        item.local_frame,
        item.det_id,
    )


def _endpoint_detection(
    detections_by_stable: Mapping[int, Sequence[DetectionCrop]],
    *,
    stable_id: int,
    first: bool,
    expected_clip_id: str,
    expected_global_frame: int,
    candidate_id: str,
) -> DetectionCrop:
    matches = tuple(detections_by_stable.get(stable_id, ()))
    if not matches:
        raise ContractError(
            f"S06 selected link {candidate_id} endpoint stable {stable_id} has no detection"
        )
    chosen = matches[0] if first else matches[-1]
    if (
        chosen.clip_id != expected_clip_id
        or chosen.global_frame != expected_global_frame
    ):
        side = "target start" if first else "source end"
        raise ContractError(
            f"S06 selected link {candidate_id} {side} detection differs from candidate"
        )
    return chosen


def _slot_label(slot: ContactSheetSlot) -> str:
    if slot.global_time_sec is None:
        return f"{slot.role}: {slot.annotation}"
    return f"{slot.role} t={slot.global_time_sec:.3f}s: {slot.annotation}"


def build_contact_sheet_plan(
    display_global_id: str,
    detections: Sequence[DetectionCrop],
    selected_links: Sequence[CandidateMetric],
) -> ContactSheetPlan:
    """Plan seven fixed overview slots without duplicating a detection crop."""

    if not isinstance(detections, Sequence) or not detections:
        raise ContractError("S06 contact sheet requires at least one detection")
    validated = tuple(
        _validate_detection(item, index=index)
        for index, item in enumerate(detections)
    )
    global_ids = {item.global_track_id for item in validated}
    display_ids = {item.display_global_id for item in validated}
    statuses = {item.id_status for item in validated}
    det_ids = [item.det_id for item in validated]
    if len(global_ids) != 1 or display_ids != {display_global_id}:
        raise ContractError("S06 contact sheet detections must belong to one display ID")
    if statuses != {FORCED_ID_STATUS}:
        raise ContractError("S06 contact sheet status must remain forced_provisional")
    if len(det_ids) != len(set(det_ids)):
        raise ContractError("S06 contact sheet detection IDs must be unique")
    ordered = tuple(sorted(validated, key=_detection_order))
    global_id = next(iter(global_ids))
    expected_display = f"G{global_id + 1:04d}"
    if display_global_id != expected_display:
        raise ContractError(
            f"S06 contact sheet display ID must be {expected_display!r}"
        )

    midpoint = (ordered[0].global_time_sec + ordered[-1].global_time_sec) / 2.0
    middle = min(
        ordered,
        key=lambda item: (
            abs(item.global_time_sec - midpoint),
            *_detection_order(item),
        ),
    )
    role_detections: dict[str, DetectionCrop | None] = {
        "start": ordered[0],
        "middle": middle,
        "end": ordered[-1],
        "weakest_link_source": None,
        "weakest_link_target": None,
        "longest_gap_source": None,
        "longest_gap_target": None,
    }
    role_annotations: dict[str, str] = {
        "start": "track start",
        "middle": "time midpoint representative",
        "end": "track end",
        "weakest_link_source": "no selected link",
        "weakest_link_target": "no selected link",
        "longest_gap_source": "no selected link",
        "longest_gap_target": "no selected link",
    }

    stable_ids = {item.stable_id for item in ordered}
    path_links: list[CandidateMetric] = []
    seen_candidates: set[str] = set()
    for index, link in enumerate(selected_links):
        if not isinstance(link, CandidateMetric):
            raise ContractError(f"S06 selected_links[{index}] has invalid type")
        if link.candidate_id in seen_candidates:
            raise ContractError("S06 selected_links contain duplicate candidate_id")
        seen_candidates.add(link.candidate_id)
        if not link.selected_by_solver:
            raise ContractError("S06 contact sheet received an unselected candidate")
        source_inside = link.source_stable_id in stable_ids
        target_inside = link.target_stable_id in stable_ids
        if source_inside != target_inside:
            raise ContractError(
                f"S06 selected link {link.candidate_id} crosses global path membership"
            )
        if source_inside:
            path_links.append(link)

    weakest = (
        min(path_links, key=lambda link: (link.appearance_cosine, link.candidate_id))
        if path_links
        else None
    )
    longest = (
        min(path_links, key=lambda link: (-link.temporal_gap_sec, link.candidate_id))
        if path_links
        else None
    )
    by_stable: dict[int, list[DetectionCrop]] = defaultdict(list)
    for item in ordered:
        by_stable[item.stable_id].append(item)

    for prefix, link in (("weakest_link", weakest), ("longest_gap", longest)):
        if link is None:
            continue
        source = _endpoint_detection(
            by_stable,
            stable_id=link.source_stable_id,
            first=False,
            expected_clip_id=link.source_end_clip_id,
            expected_global_frame=link.source_end_global_frame,
            candidate_id=link.candidate_id,
        )
        target = _endpoint_detection(
            by_stable,
            stable_id=link.target_stable_id,
            first=True,
            expected_clip_id=link.target_start_clip_id,
            expected_global_frame=link.target_start_global_frame,
            candidate_id=link.candidate_id,
        )
        role_detections[f"{prefix}_source"] = source
        role_detections[f"{prefix}_target"] = target
        detail = (
            f"{link.candidate_id}; cosine={link.appearance_cosine:.6f}; "
            f"gap={link.temporal_gap_sec:.3f}s"
        )
        role_annotations[f"{prefix}_source"] = detail
        role_annotations[f"{prefix}_target"] = detail

    first_role_by_det: dict[int, str] = {}
    slots: list[ContactSheetSlot] = []
    for role in CONTACT_SHEET_ROLES:
        detection = role_detections[role]
        if detection is None:
            slots.append(
                ContactSheetSlot(
                    role=role,
                    source=None,
                    global_time_sec=None,
                    annotation=role_annotations[role],
                )
            )
            continue
        prior_role = first_role_by_det.get(detection.det_id)
        if prior_role is not None:
            slots.append(
                ContactSheetSlot(
                    role=role,
                    source=None,
                    global_time_sec=None,
                    annotation=(
                        f"same real detection as {prior_role}; intentionally blank"
                    ),
                    deduplicated_to_role=prior_role,
                )
            )
            continue
        first_role_by_det[detection.det_id] = role
        slots.append(
            ContactSheetSlot(
                role=role,
                source=detection.source,
                global_time_sec=detection.global_time_sec,
                annotation=role_annotations[role],
            )
        )
    if tuple(slot.role for slot in slots) != CONTACT_SHEET_ROLES:
        raise ContractError("internal error: S06 contact sheet role order differs")
    sources = [slot.source.det_id for slot in slots if slot.source is not None]
    if len(sources) != len(set(sources)):
        raise ContractError("internal error: S06 contact sheet duplicated a crop")
    return ContactSheetPlan(
        global_track_id=global_id,
        display_global_id=display_global_id,
        id_status=FORCED_ID_STATUS,
        slots=tuple(slots),
        weakest_link=weakest,
        longest_gap_link=longest,
    )


def build_low_confidence_crop_requests(
    links: Sequence[CandidateMetric],
    detections: Sequence[DetectionCrop],
    *,
    crops_per_endpoint: int = 6,
) -> tuple[CropRequest, ...]:
    """Plan source-tail/target-head crops for low-cosine HTML evidence."""

    count = _integer(
        crops_per_endpoint, label="S06 crops_per_endpoint", minimum=1
    )
    validated = tuple(
        _validate_detection(item, index=index)
        for index, item in enumerate(detections)
    )
    det_ids = [item.det_id for item in validated]
    if len(det_ids) != len(set(det_ids)):
        raise ContractError("S06 crop-plan detection IDs must be unique")
    by_stable: dict[int, list[DetectionCrop]] = defaultdict(list)
    for item in sorted(validated, key=_detection_order):
        by_stable[item.stable_id].append(item)
    requests: list[CropRequest] = []
    seen_candidates: set[str] = set()
    for index, link in enumerate(links):
        if not isinstance(link, CandidateMetric) or not link.selected_by_solver:
            raise ContractError(
                f"S06 low-confidence links[{index}] must be a selected CandidateMetric"
            )
        if link.candidate_id in seen_candidates:
            raise ContractError("S06 low-confidence links contain duplicate candidate_id")
        seen_candidates.add(link.candidate_id)
        source_all = by_stable.get(link.source_stable_id, [])
        target_all = by_stable.get(link.target_stable_id, [])
        if not source_all or not target_all:
            raise ContractError(
                f"S06 low-confidence link {link.candidate_id} lacks endpoint crops"
            )
        source_rows = source_all[-count:]
        target_rows = target_all[:count]
        consumer = f"low_confidence_link:{link.candidate_id}"
        for order, item in enumerate(source_rows, start=1):
            requests.append(
                CropRequest(
                    source=item.source,
                    use=CropUse(
                        consumer_id=consumer,
                        role=f"source_tail_{order:02d}",
                        label=f"source t={item.global_time_sec:.3f}s",
                    ),
                )
            )
        for order, item in enumerate(target_rows, start=1):
            requests.append(
                CropRequest(
                    source=item.source,
                    use=CropUse(
                        consumer_id=consumer,
                        role=f"target_head_{order:02d}",
                        label=f"target t={item.global_time_sec:.3f}s",
                    ),
                )
            )
    return tuple(requests)


def _validate_crop_source(source: CropSource, *, label: str) -> CropSource:
    if not isinstance(source, CropSource):
        raise ContractError(f"{label} source has invalid type")
    _signed_int64(source.det_id, label=f"{label}.det_id")
    _string(source.clip_id, label=f"{label}.clip_id")
    _integer(source.local_frame, label=f"{label}.local_frame")
    x1, y1, x2, y2 = (
        _finite(source.x1, label=f"{label}.x1"),
        _finite(source.y1, label=f"{label}.y1"),
        _finite(source.x2, label=f"{label}.x2"),
        _finite(source.y2, label=f"{label}.y2"),
    )
    if not (0.0 <= x1 < x2 <= 3840.0 and 0.0 <= y1 < y2 <= 2160.0):
        raise ContractError(f"{label} bbox is outside raw 3840x2160 geometry")
    return source


def build_crop_request_index(requests: Iterable[CropRequest]) -> CropRequestIndex:
    """Deduplicate real crops and group frames for one sequential decode per clip."""

    if isinstance(requests, (str, bytes)):
        raise ContractError("S06 crop requests must be an iterable of CropRequest")
    source_by_det: dict[int, CropSource] = {}
    uses_by_det: dict[int, set[CropUse]] = defaultdict(set)
    try:
        iterator = iter(requests)
    except TypeError as exc:
        raise ContractError(
            "S06 crop requests must be an iterable of CropRequest"
        ) from exc
    for index, request in enumerate(iterator):
        if not isinstance(request, CropRequest):
            raise ContractError(f"S06 crop requests[{index}] has invalid type")
        source = _validate_crop_source(request.source, label=f"S06 crop requests[{index}]")
        use = request.use
        if not isinstance(use, CropUse):
            raise ContractError(f"S06 crop requests[{index}] use has invalid type")
        _string(use.consumer_id, label=f"S06 crop requests[{index}].consumer_id")
        _string(use.role, label=f"S06 crop requests[{index}].role")
        _string(use.label, label=f"S06 crop requests[{index}].label")
        previous = source_by_det.get(source.det_id)
        if previous is not None and previous != source:
            raise ContractError(
                f"S06 det_id {source.det_id} has conflicting crop provenance"
            )
        source_by_det[source.det_id] = source
        uses_by_det[source.det_id].add(use)
    indexed = [
        IndexedCropRequest(source=source, uses=tuple(sorted(uses_by_det[det_id])))
        for det_id, source in source_by_det.items()
    ]
    indexed.sort(
        key=lambda item: (
            item.source.clip_id,
            item.source.local_frame,
            item.source.det_id,
        )
    )
    grouped: dict[tuple[str, int], list[IndexedCropRequest]] = defaultdict(list)
    for item in indexed:
        grouped[(item.source.clip_id, item.source.local_frame)].append(item)
    frames = tuple(
        DecodeFrameRequest(
            clip_id=clip_id,
            local_frame=local_frame,
            crops=tuple(crops),
        )
        for (clip_id, local_frame), crops in sorted(grouped.items())
    )
    return CropRequestIndex(frames=frames)


__all__ = [
    "CONTACT_SHEET_ROLES",
    "FORCED_ID_STATUS",
    "FORCED_AUTHORIZATION_BASIS",
    "PLAN_SCHEMA_VERSION",
    "CandidateMetric",
    "ContactSheetPlan",
    "ContactSheetSlot",
    "CropRequest",
    "CropRequestIndex",
    "CropSource",
    "CropUse",
    "DecodeFrameRequest",
    "DetectionCrop",
    "IndexedCropRequest",
    "build_contact_sheet_plan",
    "build_crop_request_index",
    "build_low_confidence_crop_requests",
    "compute_directional_candidate_metrics",
    "select_low_confidence_links",
]
