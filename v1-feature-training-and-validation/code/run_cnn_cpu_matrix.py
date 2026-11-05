#!/usr/bin/env python3
"""Run the MAC-only CNN-PN-RF matrix concurrently on a CPU server."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import experiment_common as common
import run_training_matrix as matrix


DEFAULT_WINDOWS = [4]
DEFAULT_MAX_CONCURRENT_EXPERIMENTS = 6
DEFAULT_RF_JOBS = 2
STANDARD_BUILD_COMMAND = matrix.build_command


def build_cpu_command(
    job: matrix.ExperimentJob,
    settings: matrix.RunnerSettings,
    rf_jobs: int,
    lifecycle: str | None = None,
) -> list[str]:
    """Build the standard CNN command with a bounded Random Forest worker count."""
    command = STANDARD_BUILD_COMMAND(job, settings, lifecycle)
    rf_arguments = ["--rf-jobs", str(rf_jobs)]
    if lifecycle is None:
        command.extend(rf_arguments)
    else:
        command[-1:-1] = rf_arguments
    return command


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    default_repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_repo)
    parser.add_argument(
        "--features-root", type=Path, default=Path("built_features_mac")
    )
    parser.add_argument("--labels-root", type=Path, default=Path("labels"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output-performance-cnn-server-cpu-mac"),
    )
    parser.add_argument("--windows", type=int, nargs="+", default=DEFAULT_WINDOWS)
    parser.add_argument(
        "--classification-modes",
        nargs="+",
        choices=matrix.CLASSIFICATION_MODES,
        default=matrix.CLASSIFICATION_MODES,
    )
    parser.add_argument(
        "--split-kinds",
        nargs="+",
        choices=matrix.SPLIT_KINDS,
        default=matrix.SPLIT_KINDS,
    )
    parser.add_argument("--seed", type=int, default=matrix.SEED)
    parser.add_argument("--removal-runs", type=int, default=matrix.REMOVAL_RUNS)
    parser.add_argument(
        "--cnn-validation-fraction",
        type=float,
        default=matrix.CNN_VALIDATION_FRACTION,
    )
    parser.add_argument(
        "--cnn-test-trials", type=int, default=matrix.CNN_TEST_TRIALS
    )
    parser.add_argument(
        "--max-concurrent-experiments",
        type=int,
        default=DEFAULT_MAX_CONCURRENT_EXPERIMENTS,
        help="simultaneous CNN trainer processes (maximum: 6)",
    )
    parser.add_argument(
        "--cnn-rf-jobs",
        type=int,
        default=DEFAULT_RF_JOBS,
        help="Random Forest workers within each CNN experiment",
    )
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    lifecycle = parser.add_mutually_exclusive_group()
    lifecycle.add_argument("--resume", action="store_true")
    lifecycle.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.max_concurrent_experiments <= 6:
        parser.error("--max-concurrent-experiments must be between 1 and 6")
    if args.cnn_rf_jobs <= 0:
        parser.error("--cnn-rf-jobs must be positive")
    if args.removal_runs <= 0:
        parser.error("--removal-runs must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()

    def resolve_from_repo(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (repo_root / path).resolve()

    stop_file = resolve_from_repo(args.stop_file) if args.stop_file else None
    if common.stop_requested(stop_file):
        print(f"Stop requested before startup; no output was changed: {stop_file}")
        return 0
    settings = matrix.RunnerSettings(
        repo_root=repo_root,
        features_root=resolve_from_repo(args.features_root),
        labels_root=resolve_from_repo(args.labels_root),
        output_root=resolve_from_repo(args.output_root),
        stop_file=stop_file,
        seed=args.seed,
        removal_runs=args.removal_runs,
        cnn_validation_fraction=args.cnn_validation_fraction,
        cnn_test_trials=args.cnn_test_trials,
        cnn_device="cpu",
        max_concurrent_experiments=args.max_concurrent_experiments,
        workers_per_experiment=1,
        smoke=False,
    )
    jobs = matrix.generate_jobs(
        ["cnn-pn-rf"],
        args.classification_modes,
        args.windows,
        args.split_kinds,
    )

    original_build_command = matrix.build_command

    def cpu_command(
        job: matrix.ExperimentJob,
        current_settings: matrix.RunnerSettings,
        lifecycle: str | None = None,
    ) -> list[str]:
        return build_cpu_command(
            job,
            current_settings,
            args.cnn_rf_jobs,
            lifecycle,
        )

    matrix.build_command = cpu_command
    try:
        print(
            "CPU CNN policy: "
            f"concurrent experiments={args.max_concurrent_experiments}; "
            f"CNN model jobs=1; RF jobs per experiment={args.cnn_rf_jobs}; "
            "CPU affinity and memory limits are unrestricted",
            flush=True,
        )
        return matrix.execute_matrix(
            jobs,
            settings,
            resume=args.resume,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            continue_on_error=args.continue_on_error,
        )
    except (matrix.MatrixError, common.PipelineError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        matrix.build_command = original_build_command


if __name__ == "__main__":
    raise SystemExit(main())
