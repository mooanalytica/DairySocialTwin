from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cowtrack.config import ContractError
from cowtrack.qa.s01_review import run_s01_review
from cowtrack.qa.s04_conflict_review import run_s04_conflict_review
from cowtrack.qa.s04_review import run_s04_review
from cowtrack.qa.s05_review import run_s05_review
from cowtrack.stages.s00_ingest import run_s00
from cowtrack.stages.s01_microtrack import run_s01
from cowtrack.stages.s02_appearance import run_s02
from cowtrack.stages.s03_calibrate import run_s03
from cowtrack.stages.s04_finalize import run_s04_finalize
from cowtrack.stages.s04_propose import run_s04_propose
from cowtrack.stages.s05_calibrate_long import run_s05_calibrate_long
from cowtrack.stages.s05_finalize import run_s05_finalize
from cowtrack.stages.s05_force_appearance import (
    run_s05_force_appearance,
    run_s05_force_appearance_prepare,
    run_s05_force_appearance_repackage_prepared,
)
from cowtrack.stages.s05_propose import run_s05_propose
from cowtrack.stages.s06_export import run_s06


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cowtrack")
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="run one independent pipeline stage")
    stages = run_parser.add_subparsers(dest="stage", required=True)
    s00 = stages.add_parser("s00", help="normalize videos and bbox CSV files")
    s00.add_argument("--manifest", type=Path, required=True)
    s00.add_argument("--config", type=Path, required=True)
    s00.add_argument("--output", type=Path, required=True)
    s01 = stages.add_parser("s01", help="build conservative bbox-only micro-tracklets")
    s01.add_argument("--input", type=Path, required=True)
    s01.add_argument("--config", type=Path, required=True)
    s01.add_argument("--output", type=Path, required=True)
    s02 = stages.add_parser(
        "s02", help="select representative crops and build appearance embeddings"
    )
    s02.add_argument("--manifest", type=Path, required=True)
    s02.add_argument("--ingest", type=Path, required=True)
    s02.add_argument("--microtrack", type=Path, required=True)
    s02.add_argument("--review-manifest", type=Path, required=True)
    s02.add_argument("--config", type=Path, required=True)
    s02.add_argument("--device", required=True)
    s02.add_argument("--output", type=Path, required=True)
    s03 = stages.add_parser(
        "s03", help="calibrate leakage-safe short/long link scorers"
    )
    s03.add_argument("--ingest", type=Path, required=True)
    s03.add_argument("--microtrack", type=Path, required=True)
    s03.add_argument("--appearance", type=Path, required=True)
    s03.add_argument("--config", type=Path, required=True)
    s03.add_argument("--output", type=Path, required=True)
    s04_propose = stages.add_parser(
        "s04-propose", help="generate proposal-only calibrated short-link candidates"
    )
    s04_propose.add_argument("--ingest", type=Path, required=True)
    s04_propose.add_argument("--microtrack", type=Path, required=True)
    s04_propose.add_argument("--appearance", type=Path, required=True)
    s04_propose.add_argument("--calibration", type=Path, required=True)
    s04_propose.add_argument("--config", type=Path, required=True)
    s04_propose.add_argument("--output", type=Path, required=True)
    s04_review = stages.add_parser(
        "s04-review", help="render a fixed aggregate-quality sample of S04 proposals"
    )
    s04_review.add_argument("--proposals", type=Path, required=True)
    s04_review.add_argument("--ingest", type=Path, required=True)
    s04_review.add_argument("--microtrack", type=Path, required=True)
    s04_review.add_argument("--config", type=Path, required=True)
    s04_review.add_argument("--output", type=Path, required=True)
    s04_conflict_review = stages.add_parser(
        "s04-conflict-review",
        help="render representative provisional conflict groups",
    )
    s04_conflict_review.add_argument("--proposals", type=Path, required=True)
    s04_conflict_review.add_argument("--ingest", type=Path, required=True)
    s04_conflict_review.add_argument("--microtrack", type=Path, required=True)
    s04_conflict_review.add_argument("--config", type=Path, required=True)
    s04_conflict_review.add_argument("--output", type=Path, required=True)
    s04_finalize = stages.add_parser(
        "s04-finalize",
        help="apply the operator-approved component union and build stable tracks",
    )
    s04_finalize.add_argument("--ingest", type=Path, required=True)
    s04_finalize.add_argument("--microtrack", type=Path, required=True)
    s04_finalize.add_argument("--appearance", type=Path, required=True)
    s04_finalize.add_argument("--calibration", type=Path, required=True)
    s04_finalize.add_argument("--proposals", type=Path, required=True)
    s04_finalize.add_argument("--config", type=Path, required=True)
    s04_finalize.add_argument("--output", type=Path, required=True)
    s05_calibrate_long = stages.add_parser(
        "s05-calibrate-long",
        help="calibrate the stable-track long-gap link scorer",
    )
    s05_calibrate_long.add_argument("--ingest", type=Path, required=True)
    s05_calibrate_long.add_argument("--microtrack", type=Path, required=True)
    s05_calibrate_long.add_argument("--appearance", type=Path, required=True)
    s05_calibrate_long.add_argument("--calibration", type=Path, required=True)
    s05_calibrate_long.add_argument("--stable", type=Path, required=True)
    s05_calibrate_long.add_argument("--config", type=Path, required=True)
    s05_calibrate_long.add_argument("--output", type=Path, required=True)
    s05_propose = stages.add_parser(
        "s05-propose",
        help="generate exact proposal-only long-gap stable-path candidates",
    )
    s05_propose.add_argument("--ingest", type=Path, required=True)
    s05_propose.add_argument("--microtrack", type=Path, required=True)
    s05_propose.add_argument("--appearance", type=Path, required=True)
    s05_propose.add_argument("--stable", type=Path, required=True)
    s05_propose.add_argument("--long-calibration", type=Path, required=True)
    s05_propose.add_argument("--config", type=Path, required=True)
    s05_propose.add_argument("--output", type=Path, required=True)
    s05_review = stages.add_parser(
        "s05-review",
        help="render an independent bounded sample of provisional long candidates",
    )
    s05_review.add_argument("--proposals", type=Path, required=True)
    s05_review.add_argument("--ingest", type=Path, required=True)
    s05_review.add_argument("--stable", type=Path, required=True)
    s05_review.add_argument("--config", type=Path, required=True)
    s05_review.add_argument("--output", type=Path, required=True)
    s05_finalize = stages.add_parser(
        "s05-finalize",
        help="apply the operator-approved full-graph global path cover",
    )
    s05_finalize.add_argument("--ingest", type=Path, required=True)
    s05_finalize.add_argument("--microtrack", type=Path, required=True)
    s05_finalize.add_argument("--appearance", type=Path, required=True)
    s05_finalize.add_argument("--stable", type=Path, required=True)
    s05_finalize.add_argument("--long-calibration", type=Path, required=True)
    s05_finalize.add_argument("--proposals", type=Path, required=True)
    s05_finalize.add_argument("--config", type=Path, required=True)
    s05_finalize.add_argument("--output", type=Path, required=True)
    s05_force_prepare = stages.add_parser(
        "s05-force-appearance-prepare",
        help="prepare and persist graded appearance descriptors for forced linking",
    )
    s05_force_prepare.add_argument("--manifest", type=Path, required=True)
    s05_force_prepare.add_argument("--ingest", type=Path, required=True)
    s05_force_prepare.add_argument("--microtrack", type=Path, required=True)
    s05_force_prepare.add_argument("--appearance", type=Path, required=True)
    s05_force_prepare.add_argument("--stable", type=Path, required=True)
    s05_force_prepare.add_argument("--config", type=Path, required=True)
    s05_force_prepare.add_argument("--device", required=True)
    s05_force_prepare.add_argument("--output", type=Path, required=True)
    s05_force_repackage = stages.add_parser(
        "s05-force-appearance-repackage-prepared",
        help="re-key validated prepared descriptors after a solve-only config change",
    )
    s05_force_repackage.add_argument(
        "--source-prepared", type=Path, required=True
    )
    s05_force_repackage.add_argument("--config", type=Path, required=True)
    s05_force_repackage.add_argument("--output", type=Path, required=True)
    s05_force = stages.add_parser(
        "s05-force-appearance",
        help="force all stable fragments into exactly 62 provisional appearance IDs",
    )
    s05_force.add_argument("--manifest", type=Path, required=True)
    s05_force.add_argument("--ingest", type=Path, required=True)
    s05_force.add_argument("--microtrack", type=Path, required=True)
    s05_force.add_argument("--appearance", type=Path, required=True)
    s05_force.add_argument("--stable", type=Path, required=True)
    s05_force.add_argument("--prior-global", type=Path, required=True)
    s05_force.add_argument("--prepared", type=Path, required=True)
    s05_force.add_argument("--config", type=Path, required=True)
    s05_force.add_argument("--output", type=Path, required=True)
    s06 = stages.add_parser(
        "s06",
        help="export forced-provisional IDs, QA reports, contact sheets, and overlays",
    )
    s06.add_argument("--manifest", type=Path, required=True)
    s06.add_argument("--ingest", type=Path, required=True)
    s06.add_argument("--microtrack", type=Path, required=True)
    s06.add_argument("--stable", type=Path, required=True)
    s06.add_argument("--global", dest="global_dir", type=Path, required=True)
    s06.add_argument("--config", type=Path, required=True)
    s06.add_argument("--output", type=Path, required=True)
    qa_parser = commands.add_parser("qa", help="build independent review artifacts")
    qa_tasks = qa_parser.add_subparsers(dest="qa_task", required=True)
    s01_review = qa_tasks.add_parser(
        "s01-review", help="render S01 risk cases and quality references"
    )
    s01_review.add_argument("--ingest", type=Path, required=True)
    s01_review.add_argument("--microtrack", type=Path, required=True)
    s01_review.add_argument("--config", type=Path, required=True)
    s01_review.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run" and args.stage == "s00":
            run_s00(args.manifest, args.config, args.output)
            return 0
        if args.command == "run" and args.stage == "s01":
            run_s01(args.input, args.config, args.output)
            return 0
        if args.command == "run" and args.stage == "s02":
            run_s02(
                args.manifest,
                args.ingest,
                args.microtrack,
                args.review_manifest,
                args.config,
                args.device,
                args.output,
            )
            return 0
        if args.command == "run" and args.stage == "s03":
            run_s03(
                args.ingest,
                args.microtrack,
                args.appearance,
                args.config,
                args.output,
            )
            return 0
        if args.command == "run" and args.stage == "s04-propose":
            run_s04_propose(
                args.ingest,
                args.microtrack,
                args.appearance,
                args.calibration,
                args.config,
                args.output,
            )
            return 0
        if args.command == "run" and args.stage == "s04-review":
            run_s04_review(
                proposals_dir=args.proposals,
                ingest_dir=args.ingest,
                microtrack_dir=args.microtrack,
                config_path=args.config,
                output_dir=args.output,
            )
            return 0
        if args.command == "run" and args.stage == "s04-conflict-review":
            run_s04_conflict_review(
                proposals_dir=args.proposals,
                ingest_dir=args.ingest,
                microtrack_dir=args.microtrack,
                config_path=args.config,
                output_dir=args.output,
            )
            return 0
        if args.command == "run" and args.stage == "s04-finalize":
            run_s04_finalize(
                args.ingest,
                args.microtrack,
                args.appearance,
                args.calibration,
                args.proposals,
                args.config,
                args.output,
            )
            return 0
        if args.command == "run" and args.stage == "s05-calibrate-long":
            run_s05_calibrate_long(
                args.ingest,
                args.microtrack,
                args.appearance,
                args.calibration,
                args.stable,
                args.config,
                args.output,
            )
            return 0
        if args.command == "run" and args.stage == "s05-propose":
            run_s05_propose(
                args.ingest,
                args.microtrack,
                args.appearance,
                args.stable,
                args.long_calibration,
                args.config,
                args.output,
            )
            return 0
        if args.command == "run" and args.stage == "s05-review":
            run_s05_review(
                proposals_dir=args.proposals,
                ingest_dir=args.ingest,
                stable_dir=args.stable,
                config_path=args.config,
                output_dir=args.output,
            )
            return 0
        if args.command == "run" and args.stage == "s05-finalize":
            run_s05_finalize(
                args.ingest,
                args.microtrack,
                args.appearance,
                args.stable,
                args.long_calibration,
                args.proposals,
                args.config,
                args.output,
            )
            return 0
        if args.command == "run" and args.stage == "s05-force-appearance-prepare":
            run_s05_force_appearance_prepare(
                args.manifest,
                args.ingest,
                args.microtrack,
                args.appearance,
                args.stable,
                args.config,
                args.device,
                args.output,
            )
            return 0
        if (
            args.command == "run"
            and args.stage == "s05-force-appearance-repackage-prepared"
        ):
            run_s05_force_appearance_repackage_prepared(
                args.source_prepared,
                args.config,
                args.output,
            )
            return 0
        if args.command == "run" and args.stage == "s05-force-appearance":
            run_s05_force_appearance(
                args.manifest,
                args.ingest,
                args.microtrack,
                args.appearance,
                args.stable,
                args.prior_global,
                args.config,
                args.prepared,
                args.output,
            )
            return 0
        if args.command == "run" and args.stage == "s06":
            run_s06(
                args.manifest,
                args.ingest,
                args.microtrack,
                args.stable,
                args.global_dir,
                args.config,
                args.output,
            )
            return 0
        if args.command == "qa" and args.qa_task == "s01-review":
            run_s01_review(
                args.ingest,
                args.microtrack,
                args.config,
                args.output,
            )
            return 0
        raise ContractError(f"unsupported command: {args.command}")
    except ContractError as exc:
        print(f"[fatal] {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
