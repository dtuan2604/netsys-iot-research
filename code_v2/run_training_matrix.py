#!/usr/bin/env python3
"""Run the complete MAC-only cross-dataset classification experiment matrix."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import itertools
import math
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import pyarrow.parquet as pq

import experiment_common as common


# These are intentionally plain Python lists/dictionaries so the default matrix is
# easy to edit before starting a long unattended server run.
WINDOW_SECONDS = [5, 10, 30]
MODELS = ["cnn-pn-rf", "decision-tree"]
CLASSIFICATION_MODES = ["binary", "multiclass"]
SPLIT_KINDS = ["1-train-1-test", "2-train-1-test", "3-train-1-test"]
SPLIT_TRAIN_COUNTS = {
    "1-train-1-test": 1,
    "2-train-1-test": 2,
    "3-train-1-test": 3,
}
REMOVAL_COUNTS = {"NCSU": 4, "UNSW1": 6, "UNSW2": 4, "YourThing": 30}
REMOVAL_CLASS = 1
REMOVAL_RUNS = 15
SEED = 42
CNN_VALIDATION_FRACTION = 0.2
CNN_TEST_TRIALS = 100
CNN_DEVICE = "auto"
CNN_MAX_CONCURRENT_EXPERIMENTS = 1
DECISION_TREE_MAX_CONCURRENT_EXPERIMENTS = 4
DECISION_TREE_JOBS = 3

CLASSIFICATION_CONFIG = {
    "binary": {
        "feature_directory": "binary-class",
        "label_file": "mac_mapping_binary_labels.json",
        "output_directory": "binary-classification",
    },
    "multiclass": {
        "feature_directory": "multi-class",
        "label_file": "mac_mapping_multi_labels.json",
        "output_directory": "multiclass-classification",
    },
}

MODEL_SCRIPTS = {
    "cnn-pn-rf": "train_cnn_pn_rf.py",
    "decision-tree": "train_decision_tree.py",
}


class MatrixError(RuntimeError):
    """Invalid runner configuration or failed child experiment."""


@dataclass(frozen=True)
class ExperimentJob:
    model: str
    classification: str
    window_seconds: int
    train_datasets: tuple[str, ...]
    test_dataset: str

    @property
    def split_kind(self) -> str:
        return f"{len(self.train_datasets)}-train-1-test"

    @property
    def job_id(self) -> str:
        train = "+".join(self.train_datasets)
        return (
            f"{self.model}__{self.classification}__window-{self.window_seconds}s__"
            f"train-{train}__test-{self.test_dataset}"
        )


@dataclass(frozen=True)
class RunnerSettings:
    repo_root: Path
    features_root: Path
    labels_root: Path
    output_root: Path
    stop_file: Path | None
    seed: int
    removal_runs: int
    cnn_validation_fraction: float
    cnn_test_trials: int
    cnn_device: str
    max_concurrent_experiments: int
    workers_per_experiment: int
    smoke: bool
    cnn_rf_jobs: int = -1


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def directional_splits(split_kinds: Sequence[str]) -> list[tuple[tuple[str, ...], str]]:
    splits: list[tuple[tuple[str, ...], str]] = []
    for split_kind in dict.fromkeys(split_kinds):
        try:
            train_count = SPLIT_TRAIN_COUNTS[split_kind]
        except KeyError as exc:
            raise MatrixError(f"unknown split kind: {split_kind}") from exc
        for test_dataset in common.DATASETS:
            train_candidates = tuple(
                dataset for dataset in common.DATASETS if dataset != test_dataset
            )
            splits.extend(
                (train_datasets, test_dataset)
                for train_datasets in itertools.combinations(
                    train_candidates, train_count
                )
            )
    return splits


def generate_jobs(
    models: Sequence[str],
    classifications: Sequence[str],
    windows: Sequence[int],
    split_kinds: Sequence[str],
    *,
    smoke: bool = False,
) -> list[ExperimentJob]:
    if smoke:
        return [
            ExperimentJob(model, "binary", 30, ("NCSU",), "UNSW1") for model in models
        ]
    splits = directional_splits(split_kinds)
    return [
        ExperimentJob(model, classification, window, train, test)
        for model in models
        for classification in classifications
        for window in windows
        for train, test in splits
    ]


def feature_root(job: ExperimentJob, settings: RunnerSettings) -> Path:
    feature_directory = CLASSIFICATION_CONFIG[job.classification]["feature_directory"]
    return settings.features_root / feature_directory / f"window_{job.window_seconds}s"


def job_output_root(job: ExperimentJob, settings: RunnerSettings) -> Path:
    classification_directory = CLASSIFICATION_CONFIG[job.classification][
        "output_directory"
    ]
    return settings.output_root / job.model / classification_directory / job.split_kind


def removal_specs(job: ExperimentJob) -> list[dict[str, Any]]:
    return [
        {
            "dataset": dataset,
            "class": REMOVAL_CLASS,
            "count": REMOVAL_COUNTS[dataset],
        }
        for dataset in job.train_datasets
    ]


def expected_experiment_directory(job: ExperimentJob, settings: RunnerSettings) -> Path:
    return job_output_root(job, settings) / common.experiment_directory_name(
        job.train_datasets,
        [job.test_dataset],
        feature_root(job, settings),
        removal_specs(job),
        settings.seed,
    )


def scheduling_policy(
    models: Sequence[str],
    requested_concurrency: int | None,
    decision_tree_jobs: int,
) -> tuple[str, int, int]:
    selected = list(dict.fromkeys(models))
    if len(selected) != 1:
        raise MatrixError(
            "select exactly one model per matrix invocation; CNN-PN-RF and "
            "decision-tree matrices must run separately"
        )
    model = selected[0]
    if model == "cnn-pn-rf":
        maximum = CNN_MAX_CONCURRENT_EXPERIMENTS
        workers = 1
    elif model == "decision-tree":
        maximum = DECISION_TREE_MAX_CONCURRENT_EXPERIMENTS
        if decision_tree_jobs not in (2, 3):
            raise MatrixError("--decision-tree-jobs must be 2 or 3")
        workers = decision_tree_jobs
    else:
        raise MatrixError(f"unknown model: {model}")
    concurrency = maximum if requested_concurrency is None else requested_concurrency
    if concurrency <= 0 or concurrency > maximum:
        raise MatrixError(
            f"--max-concurrent-experiments for {model} must be between 1 and {maximum}"
        )
    return model, concurrency, workers


def build_command(
    job: ExperimentJob,
    settings: RunnerSettings,
    lifecycle: str | None = None,
) -> list[str]:
    classification = CLASSIFICATION_CONFIG[job.classification]
    command = [
        sys.executable,
        str(settings.repo_root / "code_v2" / MODEL_SCRIPTS[job.model]),
        "--train-datasets",
        *job.train_datasets,
        "--test-datasets",
        job.test_dataset,
        "--features-root",
        str(feature_root(job, settings)),
        "--labels-root",
        str(settings.labels_root),
        "--label-file",
        str(classification["label_file"]),
        "--output-root",
        str(job_output_root(job, settings)),
        "--seed",
        str(settings.seed),
    ]
    for spec in removal_specs(job):
        command.extend(
            [
                "--remove-devices",
                str(spec["dataset"]),
                str(spec["class"]),
                str(spec["count"]),
            ]
        )
    command.extend(["--removal-runs", str(settings.removal_runs)])
    if job.model == "cnn-pn-rf":
        command.extend(
            [
                "--validation-fraction",
                str(settings.cnn_validation_fraction),
                "--test-trials",
                str(settings.cnn_test_trials),
                "--device",
                settings.cnn_device,
                "--rf-jobs",
                str(settings.cnn_rf_jobs),
                "--jobs",
                "1",
            ]
        )
        if settings.smoke:
            command.extend(
                [
                    "--max-epochs",
                    "1",
                    "--patience",
                    "1",
                    "--rf-trees",
                    "5",
                    "--batch-size",
                    "4096",
                ]
            )
    else:
        command.extend(["--jobs", str(settings.workers_per_experiment)])
    if settings.stop_file is not None:
        command.extend(["--stop-file", str(settings.stop_file)])
    if lifecycle is not None:
        command.append(f"--{lifecycle}")
    return command


def validate_job_inputs(
    jobs: Sequence[ExperimentJob], settings: RunnerSettings
) -> None:
    checked_roots: set[tuple[str, int]] = set()
    for job in jobs:
        root_key = (job.classification, job.window_seconds)
        if root_key not in checked_roots:
            label_file = str(CLASSIFICATION_CONFIG[job.classification]["label_file"])
            common.validate_mac_feature_manifest(
                feature_root(job, settings),
                settings.labels_root,
                label_file,
                common.DATASETS,
            )
            checked_roots.add(root_key)

    checked_counts: set[tuple[str, int, str]] = set()
    for job in jobs:
        label_file = str(CLASSIFICATION_CONFIG[job.classification]["label_file"])
        for dataset in job.train_datasets:
            key = (job.classification, job.window_seconds, dataset)
            if key in checked_counts:
                continue
            labels = common.load_mac_label_map(
                settings.labels_root, dataset, label_file
            )
            evaluable = 0
            for device_id, label in labels.items():
                if label != REMOVAL_CLASS:
                    continue
                path = (
                    feature_root(job, settings)
                    / dataset
                    / device_id
                    / "features.parquet"
                )
                if not path.is_file():
                    raise MatrixError(f"missing feature file: {path}")
                if pq.ParquetFile(path).metadata.num_rows > 0:
                    evaluable += 1
            removal_count = REMOVAL_COUNTS[dataset]
            if evaluable - removal_count < 1:
                raise MatrixError(
                    f"{key}: removing {removal_count} class-{REMOVAL_CLASS} devices "
                    f"would leave fewer than one of {evaluable} evaluable devices"
                )
            if math.comb(evaluable, removal_count) < settings.removal_runs:
                raise MatrixError(
                    f"{key}: cannot produce {settings.removal_runs} distinct removals"
                )
            checked_counts.add(key)


def trainer_status(experiment_dir: Path) -> str | None:
    path = experiment_dir / "_run_manifest.json"
    if not path.is_file():
        return None
    try:
        manifest = common.read_json(path)
    except common.PipelineError:
        return "invalid"
    status = manifest.get("status")
    return str(status) if status is not None else "invalid"


def matrix_configuration(
    jobs: Sequence[ExperimentJob], settings: RunnerSettings
) -> dict[str, Any]:
    return {
        "jobs": [asdict(job) for job in jobs],
        "features_root": str(settings.features_root),
        "labels_root": str(settings.labels_root),
        "output_root": str(settings.output_root),
        "seed": settings.seed,
        "removal_counts": REMOVAL_COUNTS,
        "removal_class": REMOVAL_CLASS,
        "removal_runs": settings.removal_runs,
        "cnn_validation_fraction": settings.cnn_validation_fraction,
        "cnn_test_trials": settings.cnn_test_trials,
        "cnn_device": settings.cnn_device,
        "cnn_rf_jobs": settings.cnn_rf_jobs,
        "max_concurrent_experiments": settings.max_concurrent_experiments,
        "workers_per_experiment": settings.workers_per_experiment,
        "smoke": settings.smoke,
    }


def fingerprint_configuration(configuration: dict[str, Any]) -> dict[str, Any]:
    """Exclude scheduling controls, including keys from legacy manifests."""
    scheduling_keys = {
        "cpu_target_percent",
        "cpu_safety_margin_percent",
        "cpu_ids",
        "memory_target_percent",
        "memory_per_experiment_gb",
        "max_concurrent_experiments",
        "cnn_rf_jobs",
        "decision_tree_jobs",
        "workers_per_experiment",
    }
    return {
        key: value for key, value in configuration.items() if key not in scheduling_keys
    }


def prepare_matrix_manifest(
    jobs: Sequence[ExperimentJob],
    settings: RunnerSettings,
    *,
    resume: bool,
    overwrite: bool,
) -> tuple[Path, dict[str, Any]]:
    path = settings.output_root / "_matrix_manifest.json"
    configuration = matrix_configuration(jobs, settings)
    fingerprint = common.stable_fingerprint(fingerprint_configuration(configuration))
    previous: dict[str, Any] = {}
    if path.exists():
        previous = common.read_json(path)
        if not resume and not overwrite:
            raise MatrixError(
                f"matrix manifest already exists: {path}; pass --resume or --overwrite"
            )
        if resume and previous.get("configuration_fingerprint") != fingerprint:
            raise MatrixError(
                "existing matrix manifest is incompatible with this --resume configuration"
            )
    old_jobs = previous.get("jobs", {}) if resume else {}

    def reusable_job_state(job_id: str) -> dict[str, Any]:
        state = dict(old_jobs.get(job_id, {}))
        state.pop("cpu_ids", None)
        return state

    manifest = {
        "status": "running",
        "started_at": previous.get("started_at", utc_now()),
        "updated_at": utc_now(),
        "completed_at": None,
        "configuration": configuration,
        "configuration_fingerprint": fingerprint,
        "jobs": {
            job.job_id: {
                **reusable_job_state(job.job_id),
                "job": asdict(job),
                "expected_experiment_directory": str(
                    expected_experiment_directory(job, settings)
                ),
            }
            for job in jobs
        },
    }
    common.atomic_write_json(path, manifest)
    return path, manifest


def run_job(
    job: ExperimentJob,
    settings: RunnerSettings,
    *,
    overwrite: bool,
) -> tuple[str, int | None, list[str], Path]:
    experiment_dir = expected_experiment_directory(job, settings)
    existing_status = trainer_status(experiment_dir)
    if existing_status == "complete" and not overwrite:
        return "complete", None, [], experiment_dir
    lifecycle = "overwrite" if overwrite and experiment_dir.exists() else None
    if lifecycle is None and experiment_dir.exists():
        lifecycle = "resume"
    command = build_command(job, settings, lifecycle)
    log_dir = settings.output_root / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job.job_id}.log"
    mode = "w" if overwrite else "a"
    print(f"starting {job.job_id}", flush=True)
    print(f"  log: {log_path}", flush=True)
    print(f"  command: {shlex.join(command)}", flush=True)
    with log_path.open(mode, encoding="utf-8", buffering=1) as log:
        log.write(f"[{utc_now()}] job={job.job_id}\n")
        log.write(f"command={shlex.join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=settings.repo_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return_code = process.wait()
        log.write(f"[{utc_now()}] return_code={return_code}\n")
    final_status = trainer_status(experiment_dir) or "missing"
    if common.stop_requested(settings.stop_file) or final_status == "stopped":
        return "stopped", return_code, command, experiment_dir
    if return_code != 0 or final_status != "complete":
        return "failed", return_code, command, experiment_dir
    return "complete", return_code, command, experiment_dir


def execute_matrix(
    jobs: Sequence[ExperimentJob],
    settings: RunnerSettings,
    *,
    resume: bool,
    overwrite: bool,
    dry_run: bool,
    continue_on_error: bool,
) -> int:
    concurrency = min(len(jobs), settings.max_concurrent_experiments)
    validate_job_inputs(jobs, settings)
    print(
        f"validated {len(jobs)} experiment(s); concurrent experiments={concurrency}; "
        "CPU affinity and memory limits are unrestricted",
        flush=True,
    )
    if dry_run:
        for job in jobs:
            experiment_dir = expected_experiment_directory(job, settings)
            status = trainer_status(experiment_dir)
            lifecycle = None
            if overwrite and experiment_dir.exists():
                lifecycle = "overwrite"
            elif experiment_dir.exists() and status != "complete":
                lifecycle = "resume"
            print(f"[{job.job_id}] status={status or 'new'}")
            command = build_command(job, settings, lifecycle)
            print(shlex.join(command))
        return 0

    settings.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path, manifest = prepare_matrix_manifest(
        jobs,
        settings,
        resume=resume,
        overwrite=overwrite,
    )
    manifest["runtime_allocation"] = {
        "concurrent_experiments": concurrency,
        "workers_per_experiment": settings.workers_per_experiment,
        "cpu_affinity": "unrestricted",
        "memory_limit": "unrestricted",
    }
    common.atomic_write_json(manifest_path, manifest)
    failures = 0
    pending: list[tuple[int, ExperimentJob]] = []
    for index, job in enumerate(jobs, start=1):
        experiment_dir = expected_experiment_directory(job, settings)
        if trainer_status(experiment_dir) == "complete" and not overwrite:
            print(f"[{index}/{len(jobs)}] complete checkpoint: {job.job_id}")
            manifest["jobs"][job.job_id].update(
                {"status": "complete", "checkpoint_reused": True}
            )
            manifest["updated_at"] = utc_now()
            continue
        pending.append((index, job))
    common.atomic_write_json(manifest_path, manifest)

    pending_index = 0
    stopped = False
    abort_scheduling = False
    futures: dict[
        concurrent.futures.Future[tuple[str, int | None, list[str], Path]],
        tuple[int, ExperimentJob],
    ] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        while pending_index < len(pending) or futures:
            while (
                pending_index < len(pending)
                and len(futures) < concurrency
                and not stopped
                and not abort_scheduling
                and not common.stop_requested(settings.stop_file)
            ):
                index, job = pending[pending_index]
                pending_index += 1
                entry = manifest["jobs"][job.job_id]
                entry.update(
                    {
                        "status": "running",
                        "started_at": utc_now(),
                    }
                )
                print(
                    f"[{index}/{len(jobs)}] scheduling {job.job_id}",
                    flush=True,
                )
                future = executor.submit(
                    run_job,
                    job,
                    settings,
                    overwrite=overwrite,
                )
                futures[future] = (index, job)
            manifest["updated_at"] = utc_now()
            common.atomic_write_json(manifest_path, manifest)

            if common.stop_requested(settings.stop_file):
                stopped = True
            if not futures:
                break
            done, _ = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                index, job = futures.pop(future)
                try:
                    status, return_code, command, experiment_dir = future.result()
                except BaseException as exc:
                    status = "failed"
                    return_code = None
                    command = []
                    experiment_dir = expected_experiment_directory(job, settings)
                    manifest["jobs"][job.job_id]["error"] = str(exc)
                entry = manifest["jobs"][job.job_id]
                entry.update(
                    {
                        "status": status,
                        "completed_at": utc_now(),
                        "return_code": return_code,
                        "command": command,
                        "expected_experiment_directory": str(experiment_dir),
                    }
                )
                print(f"[{index}/{len(jobs)}] {status}: {job.job_id}", flush=True)
                if status == "stopped":
                    stopped = True
                elif status == "failed":
                    failures += 1
                    if not continue_on_error:
                        abort_scheduling = True
                        manifest["error"] = f"experiment failed: {job.job_id}"
                        print(f"failed: {job.job_id}; see its log", file=sys.stderr)
            manifest["updated_at"] = utc_now()
            common.atomic_write_json(manifest_path, manifest)

    if stopped or common.stop_requested(settings.stop_file):
        manifest["status"] = "stopped"
        manifest["stopped_at"] = utc_now()
    elif failures:
        manifest["status"] = "failed"
    elif pending_index == len(pending):
        manifest["status"] = "complete"
        manifest["completed_at"] = utc_now()
    else:
        manifest["status"] = "failed"
        manifest["error"] = "matrix scheduling ended before all experiments ran"
    manifest["updated_at"] = utc_now()
    common.atomic_write_json(manifest_path, manifest)
    return 0 if manifest["status"] == "stopped" else (1 if failures else 0)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    default_repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_repo)
    parser.add_argument(
        "--features-root", type=Path, default=Path("built_features_mac_v2")
    )
    parser.add_argument("--labels-root", type=Path, default=Path("labels"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("output-performance-folder-mac-v2")
    )
    parser.add_argument("--windows", type=int, nargs="+", default=WINDOW_SECONDS)
    parser.add_argument(
        "--models",
        nargs=1,
        choices=MODELS,
        required=True,
        help="model to run; invoke the matrix separately for each model",
    )
    parser.add_argument(
        "--classification-modes",
        nargs="+",
        choices=CLASSIFICATION_MODES,
        default=CLASSIFICATION_MODES,
    )
    parser.add_argument(
        "--split-kinds", nargs="+", choices=SPLIT_KINDS, default=SPLIT_KINDS
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--removal-runs", type=int, default=REMOVAL_RUNS)
    parser.add_argument(
        "--cnn-validation-fraction", type=float, default=CNN_VALIDATION_FRACTION
    )
    parser.add_argument("--cnn-test-trials", type=int, default=CNN_TEST_TRIALS)
    parser.add_argument(
        "--cnn-device",
        choices=("auto", "cpu", "cuda", "mps", "xpu"),
        default=CNN_DEVICE,
    )
    parser.add_argument(
        "--cnn-rf-jobs",
        type=int,
        default=-1,
        help="Random Forest workers within each CNN experiment; -1 uses all CPUs",
    )
    parser.add_argument(
        "--max-concurrent-experiments",
        type=int,
        help=(
            "simultaneous trainer processes; defaults to 1 for CNN-PN-RF and 4 "
            "for decision tree"
        ),
    )
    parser.add_argument(
        "--decision-tree-jobs",
        type=int,
        choices=(2, 3),
        default=DECISION_TREE_JOBS,
        help="parallel model runs within each decision-tree experiment",
    )
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run one reduced binary NCSU-to-UNSW1 experiment for the selected model",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    lifecycle = parser.add_mutually_exclusive_group()
    lifecycle.add_argument("--resume", action="store_true")
    lifecycle.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.cnn_rf_jobs == 0 or args.cnn_rf_jobs < -1:
        parser.error("--cnn-rf-jobs must be -1 or a positive integer")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()

    def resolve_from_repo(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (repo_root / path).resolve()

    try:
        _, concurrency, workers_per_experiment = scheduling_policy(
            args.models,
            args.max_concurrent_experiments,
            args.decision_tree_jobs,
        )
        stop_file = resolve_from_repo(args.stop_file) if args.stop_file else None
        settings = RunnerSettings(
            repo_root=repo_root,
            features_root=resolve_from_repo(args.features_root),
            labels_root=resolve_from_repo(args.labels_root),
            output_root=resolve_from_repo(args.output_root),
            stop_file=stop_file,
            seed=args.seed,
            removal_runs=1 if args.smoke else args.removal_runs,
            cnn_validation_fraction=args.cnn_validation_fraction,
            cnn_test_trials=2 if args.smoke else args.cnn_test_trials,
            cnn_device=args.cnn_device,
            cnn_rf_jobs=args.cnn_rf_jobs,
            max_concurrent_experiments=concurrency,
            workers_per_experiment=workers_per_experiment,
            smoke=args.smoke,
        )
        jobs = generate_jobs(
            args.models,
            args.classification_modes,
            args.windows,
            args.split_kinds,
            smoke=args.smoke,
        )
        if common.stop_requested(stop_file):
            print(f"Stop requested before startup; no output was changed: {stop_file}")
            return 0
        return execute_matrix(
            jobs,
            settings,
            resume=args.resume,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            continue_on_error=args.continue_on_error,
        )
    except (MatrixError, common.PipelineError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
