from __future__ import annotations

import unittest

import torch

from stage2_pti_head import PTIHead


class PTIHeadTest(unittest.TestCase):
    def test_normal_shape(self) -> None:
        model = PTIHead(num_joints=17, max_persons=6, max_frames=64)
        keypoints = torch.rand(2, 16, 4, 17, 3)
        person_mask = torch.ones(2, 16, 4, dtype=torch.bool)

        out = model(keypoints, person_mask)

        self.assertEqual(tuple(out["prob"].shape), (2,))
        self.assertEqual(tuple(out["pair_prob"].shape), (2, 6))
        self.assertEqual(tuple(out["pairs"].shape), (6, 2))

    def test_single_person_returns_low_probability(self) -> None:
        model = PTIHead(num_joints=17, max_persons=6, max_frames=64)
        out = model(torch.rand(2, 16, 1, 17, 3))

        self.assertEqual(tuple(out["prob"].shape), (2,))
        self.assertEqual(tuple(out["pair_prob"].shape), (2, 0))
        self.assertTrue(torch.all(out["prob"] < 1e-6))

    def test_missing_person_pairs_are_zero(self) -> None:
        model = PTIHead(num_joints=17, max_persons=6, max_frames=64)
        keypoints = torch.rand(1, 16, 3, 17, 3)
        person_mask = torch.ones(1, 16, 3, dtype=torch.bool)
        person_mask[:, :, 1] = False
        person_mask[:, :, 2] = False

        out = model(keypoints, person_mask)

        self.assertFalse(torch.isnan(out["prob"]).any())
        self.assertTrue(torch.all(out["pair_prob"] == 0.0))

    def test_backward_pass(self) -> None:
        model = PTIHead(num_joints=17, max_persons=6, max_frames=64)
        out = model(torch.rand(2, 16, 4, 17, 3))
        loss = out["logit"].mean()
        loss.backward()

        grads = [p.grad for p in model.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None and torch.isfinite(g).all() for g in grads))

    def test_c2_without_confidence(self) -> None:
        model = PTIHead(num_joints=17, max_persons=6, max_frames=64)
        out = model(torch.rand(2, 16, 4, 17, 2))

        self.assertEqual(tuple(out["prob"].shape), (2,))
        self.assertFalse(torch.isnan(out["prob"]).any())


if __name__ == "__main__":
    unittest.main()
