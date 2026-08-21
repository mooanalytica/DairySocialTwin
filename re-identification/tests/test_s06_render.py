from __future__ import annotations

import numpy as np
import pytest

from cowtrack.qa import s06_render
from cowtrack.qa.s06_plan import (
    CONTACT_SHEET_ROLES,
    ContactSheetPlan,
    ContactSheetSlot,
    CropSource,
)
from cowtrack.qa.s06_render import (
    OVERLAY_HEIGHT,
    OVERLAY_WIDTH,
    OverlayDetection,
    extract_detection_crop,
    global_id_color,
    render_contact_sheet,
    render_s06_overlay_frame,
)


def _raw(value: tuple[int, int, int] = (17, 18, 19)) -> np.ndarray:
    return np.full((2160, 3840, 3), value, dtype=np.uint8)


def _valid(
    det_id: int,
    global_id: int,
    box: tuple[float, float, float, float],
) -> OverlayDetection:
    return OverlayDetection(
        det_id=det_id,
        valid=True,
        x1=box[0],
        y1=box[1],
        x2=box[2],
        y2=box[3],
        global_track_id=global_id,
        display_global_id=f"G{global_id + 1:04d}",
    )


def test_global_id_colors_are_deterministic_varied_and_bright() -> None:
    first = [global_id_color(index) for index in range(62)]
    second = [global_id_color(index) for index in range(62)]
    assert first == second
    assert len(set(first)) == 62
    assert all(max(color) >= 230 and min(color) <= 100 for color in first)
    with pytest.raises(ValueError, match=r"\[0, 61\]"):
        global_id_color(62)


def test_overlay_draws_valid_boxes_at_half_geometry_without_mutating_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _raw()
    before = raw.copy()
    labels: list[str] = []
    real_put_text = s06_render.cv2.putText

    def capture_put_text(*args, **kwargs):  # type: ignore[no-untyped-def]
        labels.append(str(args[1]))
        return real_put_text(*args, **kwargs)

    monkeypatch.setattr(s06_render.cv2, "putText", capture_put_text)
    detections = [
        _valid(2, 1, (600.0, 200.0, 800.0, 400.0)),
        OverlayDetection(
            det_id=3,
            valid=False,
            x1=float("nan"),
            y1=None,
            x2=None,
            y2=None,
            global_track_id=None,
            display_global_id=None,
        ),
        _valid(1, 0, (200.0, 200.0, 400.0, 400.0)),
    ]

    rendered = render_s06_overlay_frame(raw, detections)

    np.testing.assert_array_equal(raw, before)
    assert rendered.shape == (OVERLAY_HEIGHT, OVERLAY_WIDTH, 3)
    assert tuple(rendered[100, 100]) == global_id_color(0)
    assert tuple(rendered[100, 300]) == global_id_color(1)
    assert tuple(rendered[500, 500]) == (17, 18, 19)
    # Each visible label is drawn once for a black outline and once in ID color.
    assert labels == ["G0001", "G0001", "G0002", "G0002"]


def test_overlay_is_shuffle_invariant_and_rejects_identity_on_invalid_row() -> None:
    raw = _raw((0, 0, 0))
    rows = [
        _valid(2, 1, (600.0, 200.0, 800.0, 400.0)),
        _valid(1, 0, (200.0, 200.0, 400.0, 400.0)),
    ]
    np.testing.assert_array_equal(
        render_s06_overlay_frame(raw, rows),
        render_s06_overlay_frame(raw, rows[::-1]),
    )

    invalid = OverlayDetection(3, False, None, None, None, None, 0, "G0001")
    with pytest.raises(ValueError, match="invalid row must have null global identity"):
        render_s06_overlay_frame(raw, [invalid])


def test_overlay_accepts_signed_int64_detection_ids() -> None:
    rendered = render_s06_overlay_frame(
        _raw(), [_valid(-(1 << 62), 0, (100.0, 100.0, 300.0, 300.0))]
    )
    assert rendered.shape == (OVERLAY_HEIGHT, OVERLAY_WIDTH, 3)


def test_overlay_requires_raw_unrotated_frame_and_exact_output() -> None:
    with pytest.raises(ValueError, match="raw unrotated 3840x2160"):
        render_s06_overlay_frame(
            np.zeros((3840, 2160, 3), dtype=np.uint8), []
        )
    with pytest.raises(ValueError, match="must be 1920x1080"):
        render_s06_overlay_frame(
            _raw(), [], output_width=1280, output_height=720
        )
    with pytest.raises(ValueError, match="bbox must satisfy"):
        render_s06_overlay_frame(
            _raw(), [_valid(1, 0, (-1.0, 1.0, 20.0, 20.0))]
        )


def test_unobserved_frame_is_preserved_without_identity_overlay() -> None:
    raw = _raw((23, 45, 67))
    before = raw.copy()

    rendered = render_s06_overlay_frame(raw, [])

    np.testing.assert_array_equal(raw, before)
    np.testing.assert_array_equal(
        rendered,
        s06_render.cv2.resize(
            raw,
            (OVERLAY_WIDTH, OVERLAY_HEIGHT),
            interpolation=s06_render.cv2.INTER_AREA,
        ),
    )


def test_extract_detection_crop_uses_raw_bbox_without_mutating_frame() -> None:
    raw = np.zeros((2160, 3840, 3), dtype=np.uint8)
    raw[20:40, 10:30] = (1, 2, 3)
    before = raw.copy()
    source = CropSource(1, "GX040006", 0, 10.0, 20.0, 30.0, 40.0)

    crop = extract_detection_crop(raw, source, padding_fraction=0.0)

    assert crop.shape == (20, 20, 3)
    assert np.all(crop == (1, 2, 3))
    np.testing.assert_array_equal(raw, before)


def _contact_plan() -> ContactSheetPlan:
    slots = []
    for index, role in enumerate(CONTACT_SHEET_ROLES):
        source = (
            CropSource(
                det_id=index + 1,
                clip_id="GX040006",
                local_frame=index,
                x1=10.0,
                y1=20.0,
                x2=30.0,
                y2=40.0,
            )
            if index in (0, 2)
            else None
        )
        slots.append(
            ContactSheetSlot(
                role=role,
                source=source,
                global_time_sec=float(index) if source is not None else None,
                annotation="real" if source is not None else "intentionally blank",
                deduplicated_to_role="start" if index == 1 else None,
            )
        )
    return ContactSheetPlan(
        global_track_id=0,
        display_global_id="G0001",
        id_status="forced_provisional",
        slots=tuple(slots),
        weakest_link=None,
        longest_gap_link=None,
    )


def test_contact_sheet_keeps_blank_slots_and_never_reuses_missing_crop() -> None:
    plan = _contact_plan()
    red = np.full((40, 60, 3), (0, 0, 255), dtype=np.uint8)
    # det_id 3 is deliberately absent: its slot must remain a missing marker,
    # never a copy of det_id 1.
    rendered = render_contact_sheet(
        plan,
        {1: red},
        tile_width=240,
        tile_height=180,
        columns=4,
        header_height=92,
    )

    assert rendered.shape == (92 + 2 * 180, 4 * 240, 3)
    assert tuple(rendered[92 + 20, 120]) == (0, 0, 255)
    # Deduplicated middle slot and absent decoded crop are visibly not red.
    assert tuple(rendered[92 + 20, 240 + 120]) != (0, 0, 255)
    assert tuple(rendered[92 + 20, 2 * 240 + 120]) != (0, 0, 255)


def test_contact_sheet_is_deterministic() -> None:
    plan = _contact_plan()
    crops = {1: np.full((20, 30, 3), 80, dtype=np.uint8)}
    first = render_contact_sheet(plan, crops, tile_width=200, tile_height=160)
    second = render_contact_sheet(plan, crops, tile_width=200, tile_height=160)
    np.testing.assert_array_equal(first, second)
