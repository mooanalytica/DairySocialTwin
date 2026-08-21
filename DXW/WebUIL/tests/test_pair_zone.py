from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dairy_social.aggregation import pair_zone


class PairZoneTest(unittest.TestCase):
    def test_same_or_cross_treats_path_as_the_other_zone(self) -> None:
        cases = (
            ("path", "path", "path"),
            ("path", "rest", "rest"),
            ("rest", "path", "rest"),
            ("food", "food", "food"),
            ("food", "rest", "cross_zone"),
        )
        for zone_i, zone_j, expected in cases:
            with self.subTest(zone_i=zone_i, zone_j=zone_j):
                self.assertEqual(
                    pair_zone(zone_i, zone_j, directed=False, mode="same_or_cross"),
                    expected,
                )

    def test_same_or_cross_is_symmetric(self) -> None:
        for zone_i, zone_j in (
            ("path", "rest"),
            ("food", "path"),
            ("food", "rest"),
            ("path", "path"),
        ):
            with self.subTest(zone_i=zone_i, zone_j=zone_j):
                forward = pair_zone(zone_i, zone_j, directed=True, mode="same_or_cross")
                reverse = pair_zone(zone_j, zone_i, directed=True, mode="same_or_cross")
                self.assertEqual(forward, reverse)

    def test_same_only_keeps_existing_behavior(self) -> None:
        self.assertEqual(pair_zone("rest", "rest", directed=False, mode="same_only"), "rest")
        self.assertIsNone(pair_zone("path", "rest", directed=False, mode="same_only"))

    def test_zone_pair_keeps_directed_and_undirected_behavior(self) -> None:
        self.assertEqual(
            pair_zone("rest", "food", directed=True, mode="zone_pair"),
            "rest__to__food",
        )
        self.assertEqual(
            pair_zone("rest", "food", directed=False, mode="zone_pair"),
            "food__rest",
        )
        self.assertEqual(
            pair_zone("food", "rest", directed=False, mode="zone_pair"),
            "food__rest",
        )

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported pair_zone_mode"):
            pair_zone("path", "rest", directed=False, mode="unknown")


if __name__ == "__main__":
    unittest.main()
