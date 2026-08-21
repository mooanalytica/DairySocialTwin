from __future__ import annotations

from pathlib import Path

import cowtrack.cli
from cowtrack.config import ContractError


def _args() -> list[str]:
    return [
        "run",
        "s06",
        "--manifest",
        "data/manifest.csv",
        "--ingest",
        "work/00_ingest",
        "--microtrack",
        "work/01_microtrack",
        "--stable",
        "work/04_short_stable",
        "--global",
        "work/05_forced_appearance",
        "--config",
        "configs/s06_export.yaml",
        "--output",
        "work/06_export",
    ]


def test_s06_cli_dispatches_all_seven_paths(monkeypatch) -> None:
    captured: list[tuple[object, ...]] = []
    monkeypatch.setattr(cowtrack.cli, "run_s06", lambda *args: captured.append(args))

    assert cowtrack.cli.main(_args()) == 0
    assert captured == [
        (
            Path("data/manifest.csv"),
            Path("work/00_ingest"),
            Path("work/01_microtrack"),
            Path("work/04_short_stable"),
            Path("work/05_forced_appearance"),
            Path("configs/s06_export.yaml"),
            Path("work/06_export"),
        )
    ]


def test_s06_cli_reports_contract_error_as_exit_two(monkeypatch, capsys) -> None:
    def fail(*_args: object) -> None:
        raise ContractError("synthetic S06 contract violation")

    monkeypatch.setattr(cowtrack.cli, "run_s06", fail)

    assert cowtrack.cli.main(_args()) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "[fatal] synthetic S06 contract violation\n"
