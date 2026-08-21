from __future__ import annotations

from pathlib import Path

import cowtrack.cli


def test_s01_cli_dispatches_all_three_paths(monkeypatch) -> None:
    captured: list[tuple[Path, Path, Path]] = []

    def fake_run(input_dir: Path, config: Path, output: Path) -> None:
        captured.append((input_dir, config, output))

    monkeypatch.setattr(cowtrack.cli, "run_s01", fake_run)
    result = cowtrack.cli.main(
        [
            "run",
            "s01",
            "--input",
            "input",
            "--config",
            "config.yaml",
            "--output",
            "output",
        ]
    )
    assert result == 0
    assert captured == [(Path("input"), Path("config.yaml"), Path("output"))]


def test_s01_review_cli_dispatches_all_four_paths(monkeypatch) -> None:
    captured: list[tuple[Path, Path, Path, Path]] = []

    def fake_run(
        ingest: Path,
        microtrack: Path,
        config: Path,
        output: Path,
    ) -> None:
        captured.append((ingest, microtrack, config, output))

    monkeypatch.setattr(cowtrack.cli, "run_s01_review", fake_run)
    result = cowtrack.cli.main(
        [
            "qa",
            "s01-review",
            "--ingest",
            "00_ingest",
            "--microtrack",
            "01_microtrack",
            "--config",
            "review.yaml",
            "--output",
            "01_review",
        ]
    )

    assert result == 0
    assert captured == [
        (
            Path("00_ingest"),
            Path("01_microtrack"),
            Path("review.yaml"),
            Path("01_review"),
        )
    ]
