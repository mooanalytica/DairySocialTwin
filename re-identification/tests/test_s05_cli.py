from __future__ import annotations

from pathlib import Path

import cowtrack.cli
import pytest
from cowtrack.config import ContractError


def _approved_args() -> list[str]:
    return [
        "run",
        "s05-calibrate-long",
        "--ingest",
        "work/00_ingest",
        "--microtrack",
        "work/01_microtrack",
        "--appearance",
        "work/02_appearance",
        "--calibration",
        "work/03_calibration",
        "--stable",
        "work/04_short_stable",
        "--config",
        "configs/s05_long_calibration.yaml",
        "--output",
        "work/05_long_calibration",
    ]


def _proposal_args() -> list[str]:
    return [
        "run",
        "s05-propose",
        "--ingest",
        "work/00_ingest",
        "--microtrack",
        "work/01_microtrack",
        "--appearance",
        "work/02_appearance",
        "--stable",
        "work/04_short_stable",
        "--long-calibration",
        "work/05_long_calibration",
        "--config",
        "configs/s05_proposals.yaml",
        "--output",
        "work/05_long_proposals",
    ]


def _review_args() -> list[str]:
    return [
        "run",
        "s05-review",
        "--proposals",
        "work/05_long_proposals",
        "--ingest",
        "work/00_ingest",
        "--stable",
        "work/04_short_stable",
        "--config",
        "configs/s05_review.yaml",
        "--output",
        "work/05_long_review",
    ]


def _finalize_args() -> list[str]:
    return [
        "run",
        "s05-finalize",
        "--ingest",
        "work/00_ingest",
        "--microtrack",
        "work/01_microtrack",
        "--appearance",
        "work/02_appearance",
        "--stable",
        "work/04_short_stable",
        "--long-calibration",
        "work/05_long_calibration",
        "--proposals",
        "work/05_long_proposals",
        "--config",
        "configs/s05_finalize.yaml",
        "--output",
        "work/05_global_link",
    ]


def _force_appearance_prepare_args() -> list[str]:
    return [
        "run",
        "s05-force-appearance-prepare",
        "--manifest",
        "data/manifest.csv",
        "--ingest",
        "work/00_ingest",
        "--microtrack",
        "work/01_microtrack",
        "--appearance",
        "work/02_appearance",
        "--stable",
        "work/04_short_stable",
        "--config",
        "configs/s05_force_appearance.yaml",
        "--device",
        "cuda:0",
        "--output",
        "work/05_forced_appearance_prepare",
    ]


def _force_appearance_repackage_args() -> list[str]:
    return [
        "run",
        "s05-force-appearance-repackage-prepared",
        "--source-prepared",
        "work/05_forced_appearance_prepare",
        "--config",
        "configs/s05_force_appearance.yaml",
        "--output",
        "work/05_forced_appearance_prepare_repackaged",
    ]


def _force_appearance_args() -> list[str]:
    return [
        "run",
        "s05-force-appearance",
        "--manifest",
        "data/manifest.csv",
        "--ingest",
        "work/00_ingest",
        "--microtrack",
        "work/01_microtrack",
        "--appearance",
        "work/02_appearance",
        "--stable",
        "work/04_short_stable",
        "--prior-global",
        "work/05_global_link",
        "--config",
        "configs/s05_force_appearance.yaml",
        "--prepared",
        "work/05_forced_appearance_prepare",
        "--output",
        "work/05_forced_appearance",
    ]


def test_s05_calibrate_long_cli_dispatches_fixed_contract(monkeypatch) -> None:
    captured: list[tuple[object, ...]] = []

    monkeypatch.setattr(
        cowtrack.cli,
        "run_s05_calibrate_long",
        lambda *args: captured.append(args),
    )

    assert cowtrack.cli.main(_approved_args()) == 0
    assert captured == [
        (
            Path("work/00_ingest"),
            Path("work/01_microtrack"),
            Path("work/02_appearance"),
            Path("work/03_calibration"),
            Path("work/04_short_stable"),
            Path("configs/s05_long_calibration.yaml"),
            Path("work/05_long_calibration"),
        )
    ]


def test_s05_calibrate_long_cli_reports_contract_error(
    monkeypatch, capsys
) -> None:
    def fail(*_args: object) -> None:
        raise ContractError("synthetic S05 contract violation")

    monkeypatch.setattr(cowtrack.cli, "run_s05_calibrate_long", fail)

    assert cowtrack.cli.main(_approved_args()) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "[fatal] synthetic S05 contract violation\n"


def test_s05_propose_cli_dispatches_proposal_only_contract(monkeypatch) -> None:
    captured: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        cowtrack.cli,
        "run_s05_propose",
        lambda *args: captured.append(args),
    )

    assert cowtrack.cli.main(_proposal_args()) == 0
    assert captured == [
        (
            Path("work/00_ingest"),
            Path("work/01_microtrack"),
            Path("work/02_appearance"),
            Path("work/04_short_stable"),
            Path("work/05_long_calibration"),
            Path("configs/s05_proposals.yaml"),
            Path("work/05_long_proposals"),
        )
    ]


def test_s05_propose_cli_reports_contract_error(monkeypatch, capsys) -> None:
    def fail(*_args: object) -> None:
        raise ContractError("synthetic S05 proposal violation")

    monkeypatch.setattr(cowtrack.cli, "run_s05_propose", fail)
    assert cowtrack.cli.main(_proposal_args()) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "[fatal] synthetic S05 proposal violation\n"


def test_s05_review_cli_dispatches_independent_review(monkeypatch) -> None:
    captured: list[dict[str, object]] = []
    monkeypatch.setattr(
        cowtrack.cli,
        "run_s05_review",
        lambda **kwargs: captured.append(kwargs),
    )

    assert cowtrack.cli.main(_review_args()) == 0
    assert captured == [
        {
            "proposals_dir": Path("work/05_long_proposals"),
            "ingest_dir": Path("work/00_ingest"),
            "stable_dir": Path("work/04_short_stable"),
            "config_path": Path("configs/s05_review.yaml"),
            "output_dir": Path("work/05_long_review"),
        }
    ]


def test_s05_review_cli_reports_contract_error(monkeypatch, capsys) -> None:
    def fail(**_kwargs: object) -> None:
        raise ContractError("synthetic S05 review violation")

    monkeypatch.setattr(cowtrack.cli, "run_s05_review", fail)
    assert cowtrack.cli.main(_review_args()) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "[fatal] synthetic S05 review violation\n"


def test_s05_finalize_cli_dispatches_complete_two_clip_graph(monkeypatch) -> None:
    captured: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        cowtrack.cli,
        "run_s05_finalize",
        lambda *args: captured.append(args),
    )

    assert cowtrack.cli.main(_finalize_args()) == 0
    assert captured == [
        (
            Path("work/00_ingest"),
            Path("work/01_microtrack"),
            Path("work/02_appearance"),
            Path("work/04_short_stable"),
            Path("work/05_long_calibration"),
            Path("work/05_long_proposals"),
            Path("configs/s05_finalize.yaml"),
            Path("work/05_global_link"),
        )
    ]


def test_s05_finalize_cli_reports_contract_error(monkeypatch, capsys) -> None:
    def fail(*_args: object) -> None:
        raise ContractError("synthetic S05 finalize violation")

    monkeypatch.setattr(cowtrack.cli, "run_s05_finalize", fail)
    assert cowtrack.cli.main(_finalize_args()) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "[fatal] synthetic S05 finalize violation\n"


def test_s05_force_appearance_prepare_cli_dispatches_gpu_prepare_contract(
    monkeypatch,
) -> None:
    captured: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        cowtrack.cli,
        "run_s05_force_appearance_prepare",
        lambda *args: captured.append(args),
    )

    assert cowtrack.cli.main(_force_appearance_prepare_args()) == 0
    assert captured == [
        (
            Path("data/manifest.csv"),
            Path("work/00_ingest"),
            Path("work/01_microtrack"),
            Path("work/02_appearance"),
            Path("work/04_short_stable"),
            Path("configs/s05_force_appearance.yaml"),
            "cuda:0",
            Path("work/05_forced_appearance_prepare"),
        )
    ]


def test_s05_force_appearance_prepare_cli_reports_contract_error(
    monkeypatch, capsys
) -> None:
    def fail(*_args: object) -> None:
        raise ContractError("synthetic forced appearance prepare violation")

    monkeypatch.setattr(cowtrack.cli, "run_s05_force_appearance_prepare", fail)
    assert cowtrack.cli.main(_force_appearance_prepare_args()) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "[fatal] synthetic forced appearance prepare violation\n"
    )


def test_s05_force_appearance_repackage_cli_dispatches_exact_contract(
    monkeypatch,
) -> None:
    captured: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        cowtrack.cli,
        "run_s05_force_appearance_repackage_prepared",
        lambda *args: captured.append(args),
    )

    assert cowtrack.cli.main(_force_appearance_repackage_args()) == 0
    assert captured == [
        (
            Path("work/05_forced_appearance_prepare"),
            Path("configs/s05_force_appearance.yaml"),
            Path("work/05_forced_appearance_prepare_repackaged"),
        )
    ]


def test_s05_force_appearance_repackage_cli_reports_contract_error(
    monkeypatch, capsys
) -> None:
    def fail(*_args: object) -> None:
        raise ContractError("synthetic prepared repackage violation")

    monkeypatch.setattr(
        cowtrack.cli,
        "run_s05_force_appearance_repackage_prepared",
        fail,
    )
    assert cowtrack.cli.main(_force_appearance_repackage_args()) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "[fatal] synthetic prepared repackage violation\n"


def test_s05_force_appearance_cli_dispatches_prepared_exact_62_contract(
    monkeypatch,
) -> None:
    captured: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        cowtrack.cli,
        "run_s05_force_appearance",
        lambda *args: captured.append(args),
    )

    assert cowtrack.cli.main(_force_appearance_args()) == 0
    assert captured == [
        (
            Path("data/manifest.csv"),
            Path("work/00_ingest"),
            Path("work/01_microtrack"),
            Path("work/02_appearance"),
            Path("work/04_short_stable"),
            Path("work/05_global_link"),
            Path("configs/s05_force_appearance.yaml"),
            Path("work/05_forced_appearance_prepare"),
            Path("work/05_forced_appearance"),
        )
    ]


def test_s05_force_appearance_cli_requires_explicit_prepared_bundle() -> None:
    args = _force_appearance_args()
    prepared_index = args.index("--prepared")
    del args[prepared_index : prepared_index + 2]

    with pytest.raises(SystemExit) as raised:
        cowtrack.cli.main(args)

    assert raised.value.code == 2


def test_s05_force_appearance_cli_reports_contract_error(monkeypatch, capsys) -> None:
    def fail(*_args: object) -> None:
        raise ContractError("synthetic forced appearance violation")

    monkeypatch.setattr(cowtrack.cli, "run_s05_force_appearance", fail)
    assert cowtrack.cli.main(_force_appearance_args()) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "[fatal] synthetic forced appearance violation\n"
