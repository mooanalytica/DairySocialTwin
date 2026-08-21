from __future__ import annotations

from pathlib import Path

import pytest

from cowtrack.config import ContractError, load_manifest


def test_manifest_rejects_duplicate_clip_order(tmp_path: Path) -> None:
    video_a = tmp_path / "a.mp4"
    video_b = tmp_path / "b.mp4"
    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    for path in (video_a, video_b, csv_a, csv_b):
        path.write_bytes(b"x")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "sequence_id,clip_order,clip_id,video_path,bbox_csv_path,frame_index_base,bbox_format\n"
        f"s,0,a,{video_a},{csv_a},0,xywh\n"
        f"s,0,b,{video_b},{csv_b},0,xywh\n",
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="clip_order"):
        load_manifest(manifest)

