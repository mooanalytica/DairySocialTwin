from __future__ import annotations

from pathlib import Path

import cowtrack.cli


def test_s04_propose_cli_dispatches_fixed_contract(monkeypatch) -> None:
    captured: list[tuple[object, ...]] = []

    monkeypatch.setattr(
        cowtrack.cli, "run_s04_propose", lambda *args: captured.append(args)
    )
    result = cowtrack.cli.main(
        [
            "run",
            "s04-propose",
            "--ingest",
            "work/00_ingest",
            "--microtrack",
            "work/01_microtrack",
            "--appearance",
            "work/02_appearance",
            "--calibration",
            "work/03_calibration",
            "--config",
            "configs/s04_proposals.yaml",
            "--output",
            "work/04_short_proposals",
        ]
    )
    assert result == 0
    assert captured == [
        (
            Path("work/00_ingest"),
            Path("work/01_microtrack"),
            Path("work/02_appearance"),
            Path("work/03_calibration"),
            Path("configs/s04_proposals.yaml"),
            Path("work/04_short_proposals"),
        )
    ]


def test_s04_review_cli_dispatches_fixed_contract(monkeypatch) -> None:
    captured: list[dict[str, object]] = []

    monkeypatch.setattr(
        cowtrack.cli, "run_s04_review", lambda **kwargs: captured.append(kwargs)
    )
    result = cowtrack.cli.main(
        [
            "run",
            "s04-review",
            "--proposals",
            "work/04_short_proposals",
            "--ingest",
            "work/00_ingest",
            "--microtrack",
            "work/01_microtrack",
            "--config",
            "configs/s04_review.yaml",
            "--output",
            "work/04_short_review",
        ]
    )
    assert result == 0
    assert captured == [
        {
            "proposals_dir": Path("work/04_short_proposals"),
            "ingest_dir": Path("work/00_ingest"),
            "microtrack_dir": Path("work/01_microtrack"),
            "config_path": Path("configs/s04_review.yaml"),
            "output_dir": Path("work/04_short_review"),
        }
    ]


def test_s04_conflict_review_cli_dispatches_fixed_contract(monkeypatch) -> None:
    captured: list[dict[str, object]] = []

    monkeypatch.setattr(
        cowtrack.cli,
        "run_s04_conflict_review",
        lambda **kwargs: captured.append(kwargs),
    )
    result = cowtrack.cli.main(
        [
            "run",
            "s04-conflict-review",
            "--proposals",
            "work/04_short_proposals",
            "--ingest",
            "work/00_ingest",
            "--microtrack",
            "work/01_microtrack",
            "--config",
            "configs/s04_conflict_review.yaml",
            "--output",
            "work/04_conflict_review",
        ]
    )
    assert result == 0
    assert captured == [
        {
            "proposals_dir": Path("work/04_short_proposals"),
            "ingest_dir": Path("work/00_ingest"),
            "microtrack_dir": Path("work/01_microtrack"),
            "config_path": Path("configs/s04_conflict_review.yaml"),
            "output_dir": Path("work/04_conflict_review"),
        }
    ]


def test_s04_finalize_cli_dispatches_operator_approved_contract(monkeypatch) -> None:
    captured: list[tuple[object, ...]] = []

    monkeypatch.setattr(
        cowtrack.cli, "run_s04_finalize", lambda *args: captured.append(args)
    )
    result = cowtrack.cli.main(
        [
            "run",
            "s04-finalize",
            "--ingest",
            "work/00_ingest",
            "--microtrack",
            "work/01_microtrack",
            "--appearance",
            "work/02_appearance",
            "--calibration",
            "work/03_calibration",
            "--proposals",
            "work/04_short_proposals",
            "--config",
            "configs/s04_finalize.yaml",
            "--output",
            "work/04_short_stable",
        ]
    )
    assert result == 0
    assert captured == [
        (
            Path("work/00_ingest"),
            Path("work/01_microtrack"),
            Path("work/02_appearance"),
            Path("work/03_calibration"),
            Path("work/04_short_proposals"),
            Path("configs/s04_finalize.yaml"),
            Path("work/04_short_stable"),
        )
    ]
