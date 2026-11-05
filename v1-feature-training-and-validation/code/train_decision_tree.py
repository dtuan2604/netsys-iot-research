#!/usr/bin/env python3
"""Train and evaluate the notebook-derived cross-dataset decision tree."""

from __future__ import annotations

import argparse
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import experiment_common as common


WINDOW_PREDICTION_BASE_COLUMNS = (
    "dataset",
    "device_id",
    "capture_id",
    "window",
    "true_label",
    "predicted_label",
)
DEVICE_PREDICTION_BASE_COLUMNS = (
    "dataset",
    "device_id",
    "true_label",
    "n_samples",
    "predicted_label",
    "tie_break_used",
)


def probability_columns(classes: Sequence[int]) -> list[str]:
    return [f"probability_class_{label}" for label in classes]


def window_prediction_columns(classes: Sequence[int]) -> list[str]:
    return [*WINDOW_PREDICTION_BASE_COLUMNS, *probability_columns(classes)]


def device_prediction_columns(classes: Sequence[int]) -> list[str]:
    return [
        *DEVICE_PREDICTION_BASE_COLUMNS[:4],
        *(f"votes_class_{label}" for label in classes),
        *(f"mean_probability_class_{label}" for label in classes),
        *DEVICE_PREDICTION_BASE_COLUMNS[4:],
    ]


def create_decision_tree(seed: int) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        class_weight="balanced",
        random_state=seed,
    )


def predict_windows(
    frame: pd.DataFrame,
    scaler: StandardScaler,
    classifier: DecisionTreeClassifier,
) -> pd.DataFrame:
    features = scaler.transform(common.raw_feature_matrix(frame))
    predictions = classifier.predict(features).astype(np.int32)
    classes = [int(label) for label in classifier.classes_]
    if classes != list(range(len(classes))):
        raise common.PipelineError(f"unexpected Decision Tree classes: {classes}")
    probabilities = classifier.predict_proba(features)
    output = frame.loc[:, common.IDENTITY_COLUMNS].copy()
    output["true_label"] = frame["label"].to_numpy(dtype=np.int32)
    output["predicted_label"] = predictions
    for index, label in enumerate(classes):
        output[f"probability_class_{label}"] = probabilities[:, index].astype(
            np.float64
        )
    return output.loc[:, window_prediction_columns(classes)]


def aggregate_device_predictions(
    window_predictions: pd.DataFrame, classes: Sequence[int]
) -> pd.DataFrame:
    rows = []
    for (dataset, device_id), group in window_predictions.groupby(
        ["dataset", "device_id"], sort=True
    ):
        labels = group["true_label"].unique()
        if len(labels) != 1:
            raise common.PipelineError(
                f"{dataset}/{device_id}: inconsistent true labels"
            )
        votes = {
            int(label): int((group["predicted_label"] == label).sum())
            for label in classes
        }
        max_votes = max(votes.values())
        candidates = [label for label in classes if votes[int(label)] == max_votes]
        mean_probabilities = {
            int(label): float(group[f"probability_class_{label}"].mean())
            for label in classes
        }
        tied = len(candidates) > 1
        prediction = min(
            candidates,
            key=lambda label: (-mean_probabilities[int(label)], int(label)),
        )
        row = {
            "dataset": dataset,
            "device_id": device_id,
            "true_label": int(labels[0]),
            "n_samples": int(len(group)),
            "predicted_label": int(prediction),
            "tie_break_used": tied,
        }
        row.update({f"votes_class_{label}": votes[int(label)] for label in classes})
        row.update(
            {
                f"mean_probability_class_{label}": mean_probabilities[int(label)]
                for label in classes
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=device_prediction_columns(classes))


def _execute_tree_model_run(
    run_entry: Mapping[str, Any],
    experiment_dir: Path,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    unevaluable_devices: Sequence[Mapping[str, Any]],
    train_datasets: Sequence[str],
    test_datasets: Sequence[str],
    features_root: Path,
    labels_root: Path,
    label_file: str,
    classes: Sequence[int],
    seed: int,
    stop_file: str | None,
) -> dict[str, Any]:
    run_number = int(run_entry["run"])
    run_dir, final_run_dir = common.create_model_run_staging_directory(
        experiment_dir, run_number
    )
    manifest: dict[str, Any] = {"status": "running", "run": run_number}
    common.atomic_write_json(run_dir / "_run_manifest.json", manifest)
    try:
        common.check_stop(stop_file)
        reduced_train = common.apply_removal(train_frame, run_entry["removed_devices"])
        print(
            f"model run {run_number}: fitting decision tree on "
            f"{len(reduced_train):,} rows",
            flush=True,
        )
        scaler = StandardScaler().fit(common.raw_feature_matrix(reduced_train))
        train_features = scaler.transform(common.raw_feature_matrix(reduced_train))
        train_labels = reduced_train["label"].to_numpy(dtype=np.int64)
        classifier = create_decision_tree(seed)
        classifier.fit(train_features, train_labels)
        common.check_stop(stop_file)

        window_predictions = predict_windows(test_frame, scaler, classifier)
        device_predictions = aggregate_device_predictions(window_predictions, classes)
        window_metrics = common.evaluate_predictions(
            window_predictions, test_datasets, classes
        )
        device_metrics = common.evaluate_predictions(
            device_predictions.rename(
                columns={
                    f"mean_probability_class_{label}": f"probability_class_{label}"
                    for label in classes
                }
            ),
            test_datasets,
            classes,
        )
        report = {
            "classes": list(classes),
            "window_level": window_metrics,
            "device_level": device_metrics,
            "unevaluable_devices": list(unevaluable_devices),
        }

        joblib.dump(scaler, run_dir / "scaler.joblib")
        joblib.dump(classifier, run_dir / "decision_tree.joblib")
        common.atomic_write_json(
            run_dir / "removed_devices.json", run_entry["removed_devices"]
        )
        common.atomic_write_json(
            run_dir / "config.json",
            {
                "seed": seed,
                "train_datasets": list(train_datasets),
                "test_datasets": list(test_datasets),
                "features_root": str(features_root),
                "labels_root": str(labels_root),
                "label_file": label_file,
                "feature_columns": list(common.FEATURE_COLUMNS),
                "classes": list(classes),
                "preprocessing": "StandardScaler fitted on reduced training rows",
                "classifier": "DecisionTreeClassifier(class_weight='balanced')",
                "device_vote_tie_break": (
                    "greatest mean probability among tied classes, then smallest class"
                ),
                "removed_devices": run_entry["removed_devices"],
                "training_counts_after_removal": common.removal_counts(reduced_train),
                "versions": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                    "scikit_learn": sklearn.__version__,
                },
            },
        )
        window_predictions.to_parquet(
            run_dir / "window_predictions.parquet", index=False, compression="zstd"
        )
        device_predictions.to_parquet(
            run_dir / "device_predictions.parquet", index=False, compression="zstd"
        )
        common.atomic_write_json(run_dir / "metrics.json", report)
        common.check_stop(stop_file)

        reloaded_scaler = joblib.load(run_dir / "scaler.joblib")
        reloaded_tree = joblib.load(run_dir / "decision_tree.joblib")
        reloaded = predict_windows(test_frame, reloaded_scaler, reloaded_tree)
        if not np.array_equal(
            window_predictions["predicted_label"].to_numpy(),
            reloaded["predicted_label"].to_numpy(),
        ) or not np.array_equal(
            window_predictions.loc[:, probability_columns(classes)].to_numpy(),
            reloaded.loc[:, probability_columns(classes)].to_numpy(),
        ):
            raise common.PipelineError(
                "saved Decision Tree artifacts do not reproduce predictions"
            )

        metric_rows = [
            common.flatten_metrics(
                "combined", run_number, None, device_metrics["combined"]
            )
        ]
        metric_rows.extend(
            common.flatten_metrics(
                dataset, run_number, None, device_metrics["by_dataset"][dataset]
            )
            for dataset in test_datasets
        )
        manifest.update(
            {
                "status": "complete",
                "training_counts_after_removal": common.removal_counts(reduced_train),
                "test_rows": int(len(test_frame)),
                "evaluable_test_devices": int(len(device_predictions)),
                "unevaluable_test_devices": len(unevaluable_devices),
                "artifact_reload_verified": True,
            }
        )
        common.atomic_write_json(run_dir / "_run_manifest.json", manifest)
        common.check_stop(stop_file)
        common.commit_model_run(run_dir, final_run_dir)
        return {
            "run": run_number,
            "run_dir": str(final_run_dir),
            "device_metrics": device_metrics,
            "metric_rows": metric_rows,
        }
    except common.StopRequested:
        common.discard_model_run(run_dir)
        raise
    except BaseException as exc:
        manifest.update({"status": "failed", "error": str(exc)})
        common.atomic_write_json(run_dir / "_run_manifest.json", manifest)
        common.discard_model_run(run_dir)
        raise


def _load_completed_tree_result(run_dir: Path) -> dict[str, Any]:
    manifest = common.read_json(run_dir / "_run_manifest.json")
    if manifest.get("status") != "complete" or not manifest.get(
        "artifact_reload_verified"
    ):
        raise common.PipelineError(f"incomplete model-run checkpoint: {run_dir}")
    for name in (
        "scaler.joblib",
        "decision_tree.joblib",
        "config.json",
        "window_predictions.parquet",
        "device_predictions.parquet",
    ):
        if not (run_dir / name).is_file():
            raise common.PipelineError(
                f"incomplete model-run checkpoint: {run_dir / name}"
            )
    report = common.read_json(run_dir / "metrics.json")
    run_number = int(run_dir.name.rsplit("_", 1)[1])
    rows = [
        common.flatten_metrics(
            "combined", run_number, None, report["device_level"]["combined"]
        )
    ]
    rows.extend(
        common.flatten_metrics(
            dataset,
            run_number,
            None,
            report["device_level"]["by_dataset"][dataset],
        )
        for dataset in report["device_level"]["by_dataset"]
    )
    return {
        "run": run_number,
        "run_dir": str(run_dir),
        "device_metrics": report["device_level"],
        "metric_rows": rows,
    }


def execute_experiment(args: argparse.Namespace) -> Path:
    train_datasets, test_datasets = common.validate_dataset_sets(
        args.train_datasets, args.test_datasets
    )
    jobs = int(getattr(args, "jobs", 1))
    if jobs <= 0:
        raise common.PipelineError("--jobs must be positive")
    features_root = args.features_root.resolve()
    labels_root = args.labels_root.resolve()
    label_file = common.validate_label_file_name(
        getattr(args, "label_file", common.DEFAULT_MAC_LABEL_FILE)
    )
    removal_specs = common.parse_removal_specs(args.remove_devices, train_datasets)
    resume = bool(getattr(args, "resume", False))
    overwrite = bool(getattr(args, "overwrite", False))
    stop_file_arg = getattr(args, "stop_file", None)
    stop_file = Path(stop_file_arg).resolve() if stop_file_arg is not None else None
    experiment_dir = args.output_root.resolve() / common.experiment_directory_name(
        train_datasets, test_datasets, features_root, removal_specs, args.seed
    )
    if common.stop_requested(stop_file):
        print(f"Stop requested before startup; output was not changed: {stop_file}")
        return experiment_dir

    provenance = common.validate_mac_feature_manifest(
        features_root,
        labels_root,
        label_file,
        [*train_datasets, *test_datasets],
    )
    if common.stop_requested(stop_file):
        print(f"Stop requested during setup; output was not changed: {stop_file}")
        return experiment_dir
    print(f"loading training datasets: {', '.join(train_datasets)}", flush=True)
    train_data = common.load_mac_datasets(
        train_datasets, features_root, labels_root, label_file
    )
    if common.stop_requested(stop_file):
        print(f"Stop requested during setup; output was not changed: {stop_file}")
        return experiment_dir
    print(f"loading test datasets: {', '.join(test_datasets)}", flush=True)
    test_data = common.load_mac_datasets(
        test_datasets, features_root, labels_root, label_file
    )
    if common.stop_requested(stop_file):
        print(f"Stop requested during setup; output was not changed: {stop_file}")
        return experiment_dir
    classes = common.validate_class_sets(train_data.frame, test_data.frame)
    removal_plan = common.generate_removal_plan(
        train_data.frame, removal_specs, args.removal_runs, args.seed
    )
    resume_configuration = {
        "pipeline": "decision_tree_mac_v1",
        "feature_provenance": provenance,
        "train_datasets": train_datasets,
        "test_datasets": test_datasets,
        "label_file": label_file,
        "classes": classes,
        "removal_plan": removal_plan,
        "seed": args.seed,
        "classifier": "DecisionTreeClassifier(gini, class_weight=balanced)",
    }
    fingerprint = common.stable_fingerprint(resume_configuration)
    previous = common.prepare_resumable_output_directory(
        experiment_dir,
        resume=resume,
        overwrite=overwrite,
        fingerprint=fingerprint,
    )
    manifest: dict[str, Any] = {
        "status": "running",
        "pipeline": "decision_tree",
        "started_at": (previous or {}).get("started_at", common.utc_now()),
        "updated_at": common.utc_now(),
        "completed_at": None,
        "train_datasets": train_datasets,
        "test_datasets": test_datasets,
        "feature_root_tag": common.feature_root_tag(features_root),
        "feature_provenance": provenance,
        "model_runs": len(removal_plan),
        "completed_model_runs": 0,
        "seed": args.seed,
        "label_file": label_file,
        "classes": classes,
        "configuration_fingerprint": fingerprint,
    }
    common.atomic_write_json(experiment_dir / "_run_manifest.json", manifest)
    common.atomic_write_json(experiment_dir / "removal_plan.json", removal_plan)
    common.atomic_write_json(
        experiment_dir / "config.json",
        {
            **resume_configuration,
            "features_root": str(features_root),
            "labels_root": str(labels_root),
            "jobs": jobs,
            "stop_file": str(stop_file) if stop_file is not None else None,
        },
    )
    results_by_run: dict[int, dict[str, Any]] = {}
    pending = []
    for entry in removal_plan:
        run_number = int(entry["run"])
        run_dir = experiment_dir / f"model_run_{run_number:03d}"
        loaded = None
        if resume and common.completed_model_run(run_dir):
            try:
                loaded = _load_completed_tree_result(run_dir)
            except (OSError, ValueError, KeyError, common.PipelineError):
                loaded = None
        if loaded is not None:
            results_by_run[run_number] = loaded
        else:
            if run_dir.exists():
                shutil.rmtree(run_dir)
            pending.append(entry)
    manifest["completed_model_runs"] = len(results_by_run)
    common.atomic_write_json(experiment_dir / "_run_manifest.json", manifest)

    try:
        worker_args = (
            experiment_dir,
            train_data.frame,
            test_data.frame,
            test_data.unevaluable_devices,
            train_datasets,
            test_datasets,
            features_root,
            labels_root,
            label_file,
            classes,
            args.seed,
            str(stop_file) if stop_file is not None else None,
        )

        def completed(result: dict[str, Any]) -> None:
            results_by_run[int(result["run"])] = result
            manifest["completed_model_runs"] = len(results_by_run)
            manifest["updated_at"] = common.utc_now()
            common.atomic_write_json(experiment_dir / "_run_manifest.json", manifest)

        stopped = common.run_bounded_model_jobs(
            pending,
            _execute_tree_model_run,
            worker_args,
            jobs,
            stop_file,
            completed,
        )
        if stopped:
            manifest.update(
                {
                    "status": "stopped",
                    "stopped_at": common.utc_now(),
                    "updated_at": common.utc_now(),
                    "completed_model_runs": len(results_by_run),
                }
            )
            common.atomic_write_json(experiment_dir / "_run_manifest.json", manifest)
            print("Stopped gracefully; remove the stop file and rerun with --resume.")
            return experiment_dir

        results = [results_by_run[index] for index in sorted(results_by_run)]
        if len(results) != len(removal_plan):
            raise common.PipelineError("not all model runs completed")
        common.check_stop(stop_file)
        metric_rows = [row for result in results for row in result["metric_rows"]]
        common.atomic_write_parquet(
            pd.DataFrame(metric_rows),
            experiment_dir / "model_run_metrics.parquet",
        )
        aggregate = {
            "model_runs": [
                {"run": result["run"], "device_metrics": result["device_metrics"]}
                for result in results
            ],
            "combined_across_runs": common.summarize_metric_dicts(
                [result["device_metrics"]["combined"] for result in results]
            ),
            "by_dataset_across_runs": {
                dataset: common.summarize_metric_dicts(
                    [
                        result["device_metrics"]["by_dataset"][dataset]
                        for result in results
                    ]
                )
                for dataset in test_datasets
            },
        }
        common.check_stop(stop_file)
        common.atomic_write_json(experiment_dir / "aggregate_metrics.json", aggregate)
        mean_accuracy = aggregate["combined_across_runs"]["accuracy"]["mean"]
        now = common.utc_now()
        manifest.update(
            {
                "status": "complete",
                "completed_model_runs": len(results),
                "mean_device_accuracy": mean_accuracy,
                "completed_at": now,
                "updated_at": now,
            }
        )
        common.atomic_write_json(experiment_dir / "_run_manifest.json", manifest)
        print(
            f"complete: mean_device_accuracy={mean_accuracy:.4f}; output={experiment_dir}",
            flush=True,
        )
        return experiment_dir
    except common.StopRequested:
        manifest.update(
            {"status": "stopped", "stopped_at": common.utc_now(), "updated_at": common.utc_now()}
        )
        common.atomic_write_json(experiment_dir / "_run_manifest.json", manifest)
        print("Stopped gracefully; remove the stop file and rerun with --resume.")
        return experiment_dir
    except BaseException as exc:
        manifest.update(
            {"status": "failed", "error": str(exc), "updated_at": common.utc_now()}
        )
        common.atomic_write_json(experiment_dir / "_run_manifest.json", manifest)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-datasets", nargs="+", required=True, choices=common.DATASETS
    )
    parser.add_argument(
        "--test-datasets", nargs="+", required=True, choices=common.DATASETS
    )
    parser.add_argument(
        "--features-root",
        type=Path,
        default=Path("built_features_mac/binary-class/window_30s"),
    )
    parser.add_argument("--labels-root", type=Path, default=Path("labels"))
    parser.add_argument(
        "--label-file",
        default=common.DEFAULT_MAC_LABEL_FILE,
        help="MAC label JSON filename directly within each dataset directory",
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("decision_tree_runs_mac")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--remove-devices",
        action="append",
        nargs=3,
        metavar=("DATASET", "CLASS", "COUNT"),
        help=(
            "remove COUNT random devices of CLASS from a training DATASET; repeatable"
        ),
    )
    parser.add_argument("--removal-runs", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--stop-file", type=Path)
    lifecycle = parser.add_mutually_exclusive_group()
    lifecycle.add_argument("--resume", action="store_true")
    lifecycle.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        execute_experiment(parse_args(argv))
    except (common.PipelineError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
