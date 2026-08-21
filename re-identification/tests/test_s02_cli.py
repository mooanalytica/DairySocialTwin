from __future__ import annotations

from pathlib import Path

import cowtrack.cli


def test_s02_cli_dispatches_complete_contract(monkeypatch) -> None:
    captured: list[tuple[object, ...]] = []

    def fake_run(*args: object) -> None:
        captured.append(args)

    monkeypatch.setattr(cowtrack.cli, "run_s02", fake_run)
    result = cowtrack.cli.main(
        [
            "run",
            "s02",
            "--manifest",
            "data/manifest.csv",
            "--ingest",
            "work/00_ingest",
            "--microtrack",
            "work/01_microtrack",
            "--review-manifest",
            "work/01_review/review_manifest.json",
            "--config",
            "configs/s02_appearance.yaml",
            "--device",
            "cuda:0",
            "--output",
            "work/02_appearance",
        ]
    )
    assert result == 0
    assert captured == [
        (
            Path("data/manifest.csv"),
            Path("work/00_ingest"),
            Path("work/01_microtrack"),
            Path("work/01_review/review_manifest.json"),
            Path("configs/s02_appearance.yaml"),
            "cuda:0",
            Path("work/02_appearance"),
        )
    ]
