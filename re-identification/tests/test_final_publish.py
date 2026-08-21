from __future__ import annotations

import csv
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cowtrack.config import ContractError
from cowtrack.final_publish import IDENTITY_COLUMNS, publish_final_deliverables
from cowtrack.schemas.detections import DETECTIONS_SCHEMA
from cowtrack.schemas.s05_finalize import DET_TO_GLOBAL_SCHEMA
from cowtrack.schemas.s06 import DETECTIONS_WITH_GLOBAL_ID_SCHEMA


SOURCE_HEADER = [
    "video",
    "frame",
    "track_id",
    "x",
    "y",
    "w",
    "h",
    "score",
    "identity",
    "id_conf",
]
SOURCE_ROWS = {
    "A": [
        ["A.mp4", "000", " 1 ", "1.000", "2", "3", "4", "0.90", "a,b", ""],
        ["A.mp4", "1", "2", "-0.0", "2.50", "3", "4", "", 'say "hi"', "0"],
    ],
    "B": [
        ["B.mp4", "0", "7", "9", "8", "7", "6", "1.0", "unknown", "0.000"],
    ],
}


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _derived_row(
    clip_id: str,
    clip_order: int,
    csv_row_index: int,
    *,
    valid: bool,
    global_id: int | None,
) -> list[str]:
    values = {name: "" for name in DETECTIONS_WITH_GLOBAL_ID_SCHEMA.names}
    values.update(
        {
            "sequence_id": "seq",
            "clip_id": clip_id,
            "clip_order": str(clip_order),
            "det_id": str(100 + clip_order * 10 + csv_row_index),
            "csv_row_index": str(csv_row_index),
            "local_frame": str(csv_row_index),
            "global_frame": str(clip_order * 100 + csv_row_index),
            "global_time_sec": str(float(csv_row_index)),
            "x1": "0.0",
            "y1": "0.0",
            "x2": "1.0",
            "y2": "1.0",
            "valid": str(valid),
            "qa_flags": "0" if valid else "8",
            "invalid_reason": "" if valid else "EMPTY_AFTER_CLAMP",
        }
    )
    if global_id is not None:
        values.update(
            {
                "global_track_id": str(global_id),
                "global_track_uuid": f"uuid-{global_id}",
                "display_global_id": f"G{global_id + 1:04d}",
                "id_status": "forced_provisional",
                "identity_basis": "operator_forced_appearance_exact_62",
            }
        )
    return [values[name] for name in DETECTIONS_WITH_GLOBAL_ID_SCHEMA.names]


def _fixture(tmp_path: Path) -> dict[str, Path]:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    manifest_rows: list[list[str]] = []
    for order, clip in enumerate(("A", "B")):
        video = inputs / f"{clip}.mp4"
        video.write_bytes(f"source-{clip}".encode())
        tracking = inputs / f"{clip}.csv"
        _write_csv(tracking, SOURCE_HEADER, SOURCE_ROWS[clip])
        manifest_rows.append(
            ["seq", str(order), clip, str(video), str(tracking), "0", "xywh"]
        )
    manifest = tmp_path / "manifest.csv"
    _write_csv(
        manifest,
        [
            "sequence_id",
            "clip_order",
            "clip_id",
            "video_path",
            "bbox_csv_path",
            "frame_index_base",
            "bbox_format",
        ],
        manifest_rows,
    )
    s06 = tmp_path / "s06"
    videos = s06 / "qa" / "videos"
    videos.mkdir(parents=True)
    for clip in ("A", "B"):
        (videos / f"{clip}_tracked.mp4").write_bytes(f"overlay-{clip}".encode())
    derived = s06 / "detections_with_global_id.csv"
    _write_csv(
        derived,
        DETECTIONS_WITH_GLOBAL_ID_SCHEMA.names,
        [
            _derived_row("A", 0, 0, valid=True, global_id=0),
            _derived_row("A", 0, 1, valid=False, global_id=None),
            _derived_row("B", 1, 0, valid=True, global_id=1),
        ],
    )
    return {
        "manifest": manifest,
        "s06": s06,
        "derived": derived,
        "final": tmp_path / "final",
    }


def _publish(paths: dict[str, Path]):
    return publish_final_deliverables(
        paths["manifest"],
        paths["s06"],
        paths["final"],
        expected_invalid_count=1,
        expected_global_track_count=2,
    )


def test_publishes_source_cells_exactly_and_strictly_resumes(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    result = _publish(paths)

    assert not result.resumed
    assert (result.row_count, result.valid_count, result.invalid_count) == (3, 2, 1)
    assert {path.name for path in paths["final"].iterdir()} == {
        "detections_with_global_id.csv",
        "A_tracked.mp4",
        "B_tracked.mp4",
    }
    for clip in ("A", "B"):
        published = paths["final"] / f"{clip}_tracked.mp4"
        source = paths["s06"] / "qa" / "videos" / f"{clip}_tracked.mp4"
        assert published.is_file()
        assert not published.is_symlink()
        assert os.path.samefile(published, source)

    with result.detections_csv.open("r", encoding="utf-8", newline="") as handle:
        exported = list(csv.reader(handle))
    assert exported[0] == [*SOURCE_HEADER, *IDENTITY_COLUMNS]
    assert [row[:10] for row in exported[1:]] == [
        *SOURCE_ROWS["A"],
        *SOURCE_ROWS["B"],
    ]
    assert exported[1][-5:] == [
        "0",
        "uuid-0",
        "G0001",
        "forced_provisional",
        "operator_forced_appearance_exact_62",
    ]
    assert exported[2][-5:] == ["", "", "", "", ""]

    resumed = _publish(paths)
    assert resumed.resumed
    assert resumed.marker_payload()["row_count"] == 3


def test_publication_supports_more_than_two_manifest_clips(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    inputs = tmp_path / "inputs"
    video_c = inputs / "C.mp4"
    tracking_c = inputs / "C.csv"
    video_c.write_bytes(b"source-C")
    row_c = ["C.mp4", "0", "8", "1", "2", "3", "4", "0.8", "", ""]
    _write_csv(tracking_c, SOURCE_HEADER, [row_c])
    _write_csv(
        paths["manifest"],
        [
            "sequence_id",
            "clip_order",
            "clip_id",
            "video_path",
            "bbox_csv_path",
            "frame_index_base",
            "bbox_format",
        ],
        [
            ["seq", "0", "A", str(inputs / "A.mp4"), str(inputs / "A.csv"), "0", "xywh"],
            ["seq", "1", "B", str(inputs / "B.mp4"), str(inputs / "B.csv"), "0", "xywh"],
            ["seq", "2", "C", str(video_c), str(tracking_c), "0", "xywh"],
        ],
    )
    (paths["s06"] / "qa" / "videos" / "C_tracked.mp4").write_bytes(b"overlay-C")
    _write_csv(
        paths["derived"],
        DETECTIONS_WITH_GLOBAL_ID_SCHEMA.names,
        [
            _derived_row("A", 0, 0, valid=True, global_id=0),
            _derived_row("A", 0, 1, valid=False, global_id=None),
            _derived_row("B", 1, 0, valid=True, global_id=1),
            _derived_row("C", 2, 0, valid=True, global_id=2),
        ],
    )

    result = publish_final_deliverables(
        paths["manifest"],
        paths["s06"],
        paths["final"],
        expected_invalid_count=1,
        expected_global_track_count=3,
    )

    assert result.row_count == 4
    assert set(result.videos_by_clip) == {"A", "B", "C"}
    assert (paths["final"] / "C_tracked.mp4").is_file()


def test_missing_identity_key_fails_without_publishing_directory(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _write_csv(
        paths["derived"],
        DETECTIONS_WITH_GLOBAL_ID_SCHEMA.names,
        [
            _derived_row("A", 0, 0, valid=True, global_id=0),
            _derived_row("B", 1, 0, valid=True, global_id=1),
        ],
    )
    with pytest.raises(ContractError, match="bijection failed"):
        _publish(paths)
    assert not paths["final"].exists()
    assert not list(tmp_path.glob(".final.staging-*"))


def test_duplicate_identity_key_is_rejected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    duplicate = _derived_row("A", 0, 0, valid=False, global_id=None)
    _write_csv(
        paths["derived"],
        DETECTIONS_WITH_GLOBAL_ID_SCHEMA.names,
        [
            _derived_row("A", 0, 0, valid=True, global_id=0),
            duplicate,
            _derived_row("B", 1, 0, valid=True, global_id=1),
        ],
    )
    with pytest.raises(ContractError, match="duplicated or not in manifest order"):
        _publish(paths)


def test_invalid_derived_row_must_have_five_blank_identity_fields(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    rows = [
        _derived_row("A", 0, 0, valid=True, global_id=0),
        _derived_row("A", 0, 1, valid=False, global_id=None),
        _derived_row("B", 1, 0, valid=True, global_id=1),
    ]
    identity_basis = DETECTIONS_WITH_GLOBAL_ID_SCHEMA.names.index("identity_basis")
    rows[1][identity_basis] = "fabricated"
    _write_csv(paths["derived"], DETECTIONS_WITH_GLOBAL_ID_SCHEMA.names, rows)
    with pytest.raises(ContractError, match="invalid row has identity"):
        _publish(paths)


def test_identity_metadata_must_be_consistent_per_global_id(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    rows = [
        _derived_row("A", 0, 0, valid=True, global_id=0),
        _derived_row("A", 0, 1, valid=True, global_id=0),
        _derived_row("B", 1, 0, valid=True, global_id=1),
    ]
    uuid_index = DETECTIONS_WITH_GLOBAL_ID_SCHEMA.names.index("global_track_uuid")
    rows[1][uuid_index] = "different-uuid-for-the-same-id"
    _write_csv(paths["derived"], DETECTIONS_WITH_GLOBAL_ID_SCHEMA.names, rows)

    with pytest.raises(ContractError, match="identity metadata is inconsistent"):
        publish_final_deliverables(
            paths["manifest"],
            paths["s06"],
            paths["final"],
            expected_invalid_count=0,
            expected_global_track_count=2,
        )


def test_existing_delivery_is_revalidated_and_never_repaired(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    result = _publish(paths)
    with result.detections_csv.open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")
    before = result.detections_csv.read_bytes()
    with pytest.raises(ContractError, match="extra data rows"):
        _publish(paths)
    assert result.detections_csv.read_bytes() == before


def _default_value(field: pa.Field) -> object:
    if pa.types.is_string(field.type):
        return "x"
    if pa.types.is_boolean(field.type):
        return True
    if pa.types.is_integer(field.type):
        return 0
    if pa.types.is_floating(field.type):
        return 0.0
    raise AssertionError(field)


def _arrow_rows(schema: pa.Schema, overrides: list[dict[str, object]]) -> pa.Table:
    rows: list[dict[str, object]] = []
    for override in overrides:
        row = {field.name: _default_value(field) for field in schema}
        row.update(override)
        rows.append(row)
    return pa.Table.from_pylist(rows, schema=schema)


def test_can_join_s00_detections_to_forced_mapping(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    s00_path = tmp_path / "detections.parquet"
    mapping_path = tmp_path / "det_to_global.parquet"
    detections = _arrow_rows(
        DETECTIONS_SCHEMA,
        [
            {"det_id": 10, "sequence_id": "seq", "clip_id": "A", "csv_row_index": 0},
            {
                "det_id": 11,
                "sequence_id": "seq",
                "clip_id": "A",
                "csv_row_index": 1,
                "valid": False,
            },
            {"det_id": 12, "sequence_id": "seq", "clip_id": "B", "csv_row_index": 0},
        ],
    )
    mapping = _arrow_rows(
        DET_TO_GLOBAL_SCHEMA,
        [
            {
                "det_id": 12,
                "sequence_id": "seq",
                "clip_id": "B",
                "clip_order": 1,
                "global_track_id": 1,
                "global_track_uuid": "uuid-1",
                "display_global_id": "G0002",
                "id_status": "forced_provisional",
                "identity_basis": "operator_forced_appearance_exact_62",
            },
            {
                "det_id": 10,
                "sequence_id": "seq",
                "clip_id": "A",
                "clip_order": 0,
                "global_track_id": 0,
                "global_track_uuid": "uuid-0",
                "display_global_id": "G0001",
                "id_status": "forced_provisional",
                "identity_basis": "operator_forced_appearance_exact_62",
            },
        ],
    )
    pq.write_table(detections, s00_path)
    pq.write_table(mapping, mapping_path)

    result = publish_final_deliverables(
        paths["manifest"],
        paths["s06"],
        paths["final"],
        detections_path=s00_path,
        forced_mapping_path=mapping_path,
        expected_invalid_count=1,
        expected_global_track_count=2,
    )
    assert (result.row_count, result.valid_count, result.invalid_count) == (3, 2, 1)
