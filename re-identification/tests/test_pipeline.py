from __future__ import annotations

import json
from pathlib import Path

import pytest

from cowtrack.config import ContractError
import cowtrack.pipeline as pipeline
from cowtrack.pipeline import (
    EXPECTED_BBOX_ROOT,
    EXPECTED_CLIPS,
    EXPECTED_VIDEO_ROOT,
    WORKSPACE_MARKER,
    PipelineLayout,
    PipelineRunners,
    execute_stages,
    prepare_fresh_workspace,
    realtime_pipeline_log,
)


def test_fixed_layout_resumes_existing_all11_outputs() -> None:
    layout = PipelineLayout.fixed()

    assert len(EXPECTED_CLIPS) == 11
    assert EXPECTED_CLIPS[0] == (0, "GX010006")
    assert EXPECTED_CLIPS[-1] == (10, "GX110006")
    assert layout.work_root.name == "dairy_farm_1_gopro1_20250505_all11"
    assert layout.reference_root.name == (
        "dairy_farm_1_gopro1_20250505_all11_inspection_backup_20260715"
    )
    assert layout.deliverables_root.name == (
        "dairy_farm_1_gopro1_20250505_all11_y20260716"
    )
    assert layout.log_path.name == "run_pipeline_all11_y20260716.log"
    assert EXPECTED_VIDEO_ROOT == Path(
        "/home/hyw/UPAN_HYW/May 5 2025 Dairy Farm 1 Videos/"
        "Gopro1/100GOPRO"
    )
    assert EXPECTED_BBOX_ROOT == Path(
        "/home/hyw/UPAN_HYW/stage1_output/1/Gopro1"
    )


def _layout(tmp_path: Path) -> PipelineLayout:
    root = tmp_path / "project"
    return PipelineLayout(
        project_root=root,
        venv_root=tmp_path / "venv",
        manifest=root / "data" / "manifest.csv",
        work_root=root / "work" / "sequence",
        reference_root=root / "work" / "sequence_inspection",
        deliverables_root=root / "final" / "sequence",
        log_path=root / "logs" / "pipeline.log",
    )


def test_prepare_workspace_renames_inspection_tree_once(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    layout.work_root.mkdir(parents=True)
    (layout.work_root / "old.txt").write_text("inspection only", encoding="utf-8")

    prepare_fresh_workspace(layout)

    assert (layout.reference_root / "old.txt").read_text(encoding="utf-8") == (
        "inspection only"
    )
    marker = layout.work_root / WORKSPACE_MARKER
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["historical_output_reused"] is False
    assert payload["inspection_backup"] == str(layout.reference_root)

    prepare_fresh_workspace(layout)
    assert marker.is_file()


def test_prepare_workspace_rejects_ambiguous_unmanaged_trees(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    layout.work_root.mkdir(parents=True)
    layout.reference_root.mkdir(parents=True)

    with pytest.raises(ContractError, match="both exist"):
        prepare_fresh_workspace(layout)


def test_realtime_pipeline_log_flushes_immediately(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "run.log"
    with realtime_pipeline_log(path):
        print("visible-now", flush=True)
        assert "visible-now" in path.read_text(encoding="utf-8")
    text = path.read_text(encoding="utf-8")
    assert "realtime log probe" in text
    assert "realtime log verified" in text


def test_execute_stages_uses_fixed_unattended_order(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    calls: list[tuple[str, tuple[object, ...]]] = []

    def record(name: str):
        def runner(*args: object) -> dict[str, str]:
            calls.append((name, args))
            return {"stage": name}

        return runner

    runners = PipelineRunners(
        s00=record("s00"),
        s01=record("s01"),
        s01_plan=record("s01_plan"),
        s02=record("s02"),
        s03=record("s03"),
        s04_propose=record("s04_propose"),
        s04_finalize=record("s04_finalize"),
        s05_calibrate_long=record("s05_calibrate_long"),
        s05_propose=record("s05_propose"),
        s05_finalize=record("s05_finalize"),
        s05_force_prepare=record("s05_force_prepare"),
        s05_force=record("s05_force"),
        s06=record("s06"),
        publish=record("publish"),
    )

    execute_stages(layout, runners)

    assert [name for name, _ in calls] == [
        "s00",
        "s01",
        "s01_plan",
        "s02",
        "s03",
        "s04_propose",
        "s04_finalize",
        "s05_calibrate_long",
        "s05_propose",
        "s05_finalize",
        "s05_force_prepare",
        "s05_force",
        "s06",
        "publish",
    ]
    by_name = dict(calls)
    assert by_name["s02"][3] == (
        layout.stage("01_review_plan") / "review_manifest.json"
    )
    assert by_name["s02"][5] == "cuda:0"
    assert by_name["s05_force_prepare"] == (
        layout.manifest,
        layout.stage("00_ingest"),
        layout.stage("01_microtrack"),
        layout.stage("02_appearance"),
        layout.stage("04_short_stable"),
        layout.config("s05_force_appearance.yaml"),
        "cuda:0",
        layout.stage("05_forced_appearance_prepare"),
    )
    assert by_name["s05_force"] == (
        layout.manifest,
        layout.stage("00_ingest"),
        layout.stage("01_microtrack"),
        layout.stage("02_appearance"),
        layout.stage("04_short_stable"),
        layout.stage("05_global_link"),
        layout.config("s05_force_appearance.yaml"),
        layout.stage("05_forced_appearance_prepare"),
        layout.stage("05_forced_appearance"),
    )
    assert by_name["s06"][4] == layout.stage("05_forced_appearance")
    assert by_name["publish"] == (
        layout.manifest,
        layout.stage("06_export"),
        layout.deliverables_root,
    )


def test_pipeline_preflights_and_imports_before_preparing_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _layout(tmp_path)
    events: list[str] = []
    sentinel = object()
    monkeypatch.setattr(
        pipeline, "validate_fixed_layout", lambda selected: events.append("validate")
    )
    monkeypatch.setattr(
        pipeline,
        "load_default_runners",
        lambda: events.append("import-runners") or sentinel,
    )
    monkeypatch.setattr(
        pipeline,
        "prepare_fresh_workspace",
        lambda selected: events.append("prepare-workspace"),
    )
    monkeypatch.setattr(
        pipeline,
        "execute_stages",
        lambda selected, runners: events.append("execute")
        if runners is sentinel
        else pytest.fail("unexpected runners"),
    )

    pipeline.run_pipeline(
        layout=layout,
        preflight=lambda selected: events.append("preflight"),
    )

    assert events == [
        "validate",
        "preflight",
        "import-runners",
        "prepare-workspace",
        "execute",
    ]


def test_preflight_failure_leaves_workspace_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _layout(tmp_path)
    events: list[str] = []
    monkeypatch.setattr(
        pipeline, "validate_fixed_layout", lambda selected: events.append("validate")
    )
    monkeypatch.setattr(
        pipeline,
        "prepare_fresh_workspace",
        lambda selected: events.append("prepare-workspace"),
    )

    def fail_preflight(selected: PipelineLayout) -> None:
        events.append("preflight")
        raise ContractError("preflight failed")

    with pytest.raises(ContractError, match="preflight failed"):
        pipeline.run_pipeline(layout=layout, preflight=fail_preflight)

    assert events == ["validate", "preflight"]
    assert not layout.work_root.exists()


def test_main_writes_contract_failure_to_realtime_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _layout(tmp_path)
    monkeypatch.setattr(
        pipeline.PipelineLayout,
        "fixed",
        classmethod(lambda cls: layout),
    )

    def fail_body(*args: object, **kwargs: object) -> None:
        raise ContractError("deliberate logged failure")

    monkeypatch.setattr(pipeline, "_run_pipeline_body", fail_body)

    assert pipeline.main() == 2
    assert "[fatal] deliberate logged failure" in layout.log_path.read_text(
        encoding="utf-8"
    )
