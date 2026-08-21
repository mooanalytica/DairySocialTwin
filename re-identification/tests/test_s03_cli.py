from __future__ import annotations

from pathlib import Path

import cowtrack.cli


def test_s03_cli_dispatches_approved_contract(monkeypatch) -> None:
    captured: list[tuple[object, ...]] = []

    def fake_run(*args: object) -> None:
        captured.append(args)

    monkeypatch.setattr(cowtrack.cli, "run_s03", fake_run)
    result = cowtrack.cli.main(
        [
            "run",
            "s03",
            "--ingest",
            "work/00_ingest",
            "--microtrack",
            "work/01_microtrack",
            "--appearance",
            "work/02_appearance",
            "--config",
            "configs/s03_calibration.yaml",
            "--output",
            "work/03_calibration",
        ]
    )
    assert result == 0
    assert captured == [
        (
            Path("work/00_ingest"),
            Path("work/01_microtrack"),
            Path("work/02_appearance"),
            Path("configs/s03_calibration.yaml"),
            Path("work/03_calibration"),
        )
    ]

