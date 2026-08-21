"""Closed dataset contract shared by the 11-clip linking stages.

The repository intentionally targets the one GoPro sequence named here.  Keep
the ordered clip list in one module so S03--S06 cannot silently disagree about
which boundaries are legal linking transitions.
"""

from __future__ import annotations


EXPECTED_SEQUENCE_ID = "dairy_farm_1_gopro1_20250505"
EXPECTED_CLIP_ORDER = tuple(f"GX{index:02d}0006" for index in range(1, 12))
EXPECTED_FRAME_COUNTS = (
    88_320,
    84_480,
    78_720,
    84_480,
    78_720,
    78_720,
    76_800,
    71_040,
    72_960,
    65_280,
    46_972,
)
EXPECTED_FRAME_COUNT = sum(EXPECTED_FRAME_COUNTS)
MAX_GLOBAL_TRACK_COUNT = 62


__all__ = [
    "EXPECTED_CLIP_ORDER",
    "EXPECTED_FRAME_COUNT",
    "EXPECTED_FRAME_COUNTS",
    "EXPECTED_SEQUENCE_ID",
    "MAX_GLOBAL_TRACK_COUNT",
]
