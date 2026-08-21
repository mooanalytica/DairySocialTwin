from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FrontendContractTests(unittest.TestCase):
    def test_fixed_review_controls_and_video_contract(self) -> None:
        html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        self.assertIn('id="occurrenceList"', html)
        self.assertIn('id="reviewVideo"', html)
        self.assertIn('id="videoProgress"', html)
        self.assertIn('id="acceptButton"', html)
        self.assertIn('id="invalidButton"', html)
        self.assertIn('id="updateButton"', html)
        self.assertIn('min="1"', html)
        self.assertIn('max="62"', html)
        self.assertIn("autoplay", html)
        self.assertIn("loop", html)
        self.assertNotIn("<canvas", html.lower())

    def test_api_paths_and_no_external_assets(self) -> None:
        html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        javascript = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
        self.assertIn('requestJson("/api/state")', javascript)
        self.assertIn("/api/reviews/${encodeURIComponent(submittedId)}", javascript)
        self.assertIn('submitReview("accept")', javascript)
        self.assertIn('submitReview("invalid_multiple_cows")', javascript)
        self.assertIn('submitReview("update_id")', javascript)
        self.assertIn("item.playback_start_sec", javascript)
        self.assertIn("elements.video.currentTime = target", javascript)
        self.assertNotIn("https://", html)
        self.assertNotIn("http://", html)


if __name__ == "__main__":
    unittest.main()
