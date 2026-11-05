from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code_v2 import experiment_common as common
from code_v2 import train_decision_tree as pipeline


def write_dataset(
    features_root: Path,
    labels_root: Path,
    dataset: str,
    offset: float,
    class_count: int = 3,
    label_file: str = "mac_mapping_binary_labels.json",
) -> None:
    labels = {}
    for label in range(class_count):
        for device_number in range(3):
            device_id = f"02:00:00:{label:02x}:{device_number:02x}:01"
            labels[device_id] = {"is_iot_device": label}
            rows = []
            for window in range(4):
                center = offset + 1.0 + label * 100.0
                values = [
                    center + window * 0.01 + number
                    for number in range(len(common.FEATURE_COLUMNS))
                ]
                feature_values = dict(zip(common.FEATURE_COLUMNS, values))
                if window == 0:
                    feature_values["outgoing_std_frame_size"] = None
                    feature_values["max_inter_arrival_time"] = None
                rows.append(
                    {
                        "capture_id": "24-01-01",
                        "window": window,
                        "dataset": dataset,
                        "device_id": device_id,
                        "device_id_type": "mac",
                        "is_iot_device": label,
                        **feature_values,
                    }
                )
            target = features_root / dataset / device_id
            target.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(target / "features.parquet", index=False)
    label_dir = labels_root / dataset
    label_dir.mkdir(parents=True, exist_ok=True)
    label_bytes = json.dumps(labels).encode()
    (label_dir / label_file).write_bytes(label_bytes)
    manifest_path = features_root / "_run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "status": "complete",
        "algorithm_version": 2,
        "configuration": {
            "label_file": label_file,
            "model_feature_columns": list(common.FEATURE_COLUMNS),
            "nullable_feature_columns": list(common.NULLABLE_FEATURE_COLUMNS),
            "missing_indicator_columns": list(common.MISSING_INDICATOR_COLUMNS),
        },
        "datasets": {},
    }
    manifest["datasets"][dataset] = {
        "identifier_type": "mac",
        "label_sha256": hashlib.sha256(label_bytes).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest))


def test_label_file_cli_default_and_override():
    required = [
        "--train-datasets",
        "NCSU",
        "--test-datasets",
        "UNSW1",
    ]
    assert pipeline.parse_args(required).label_file == common.DEFAULT_MAC_LABEL_FILE
    assert (
        pipeline.parse_args([*required, "--label-file", "binary_label.json"]).label_file
        == "binary_label.json"
    )
    defaults = pipeline.parse_args(required)
    assert defaults.features_root == Path(
        "built_features_mac_v2/binary-class/window_30s"
    )
    assert defaults.labels_root == Path("labels")
    assert defaults.output_root == Path("decision_tree_runs_mac_v2")
    assert defaults.stop_file is None


def test_plurality_vote_uses_probability_then_smallest_class_for_ties():
    predictions = pd.DataFrame(
        [
            ["D", "a", "c", 0, 1, 0, 0.4, 0.3, 0.3],
            ["D", "a", "c", 1, 1, 1, 0.2, 0.7, 0.1],
            ["D", "b", "c", 0, 2, 1, 0.1, 0.45, 0.45],
            ["D", "b", "c", 1, 2, 2, 0.1, 0.45, 0.45],
        ],
        columns=pipeline.window_prediction_columns([0, 1, 2]),
    )
    devices = pipeline.aggregate_device_predictions(predictions, [0, 1, 2]).set_index(
        "device_id"
    )
    assert devices.loc["a", "tie_break_used"]
    assert devices.loc["a", "predicted_label"] == 1
    assert devices.loc["b", "predicted_label"] == 1


@pytest.mark.parametrize("class_count", [2, 3])
def test_decision_tree_end_to_end_with_repeated_class_removal(tmp_path, class_count):
    features = tmp_path / "window_fixture"
    labels = tmp_path / "labels"
    label_file = "binary_label.json"
    for dataset, offset in (("NCSU", 0.0), ("UNSW1", 0.5), ("YourThing", 0.25)):
        write_dataset(features, labels, dataset, offset, class_count, label_file)
    args = SimpleNamespace(
        train_datasets=["NCSU", "YourThing"],
        test_datasets=["UNSW1"],
        features_root=features,
        labels_root=labels,
        label_file=label_file,
        output_root=tmp_path / "runs",
        seed=42,
        remove_devices=[["YourThing", str(class_count - 1), "1"]],
        removal_runs=2,
        jobs=2,
        overwrite=False,
    )
    experiment_dir = pipeline.execute_experiment(args)
    manifest = json.loads((experiment_dir / "_run_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["completed_model_runs"] == 2
    assert manifest["label_file"] == label_file
    plan = json.loads((experiment_dir / "removal_plan.json").read_text())
    assert plan[0]["removed_devices"] != plan[1]["removed_devices"]
    for run in (1, 2):
        run_dir = experiment_dir / f"model_run_{run:03d}"
        assert (run_dir / "decision_tree.joblib").is_file()
        predictions = pd.read_parquet(run_dir / "device_predictions.parquet")
        classes = list(range(class_count))
        assert len(predictions) == 3 * class_count
        assert set(pipeline.probability_columns(classes)) <= {
            column.replace("mean_", "") for column in predictions.columns
        }
        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert metrics["classes"] == classes
        config = json.loads((run_dir / "config.json").read_text())
        assert config["label_file"] == label_file
        assert config["feature_algorithm_version"] == 2
        assert config["model_feature_columns"] == list(common.MODEL_FEATURE_COLUMNS)
        assert (run_dir / "removed_devices.json").is_file()

    preserved = experiment_dir / "model_run_001" / "decision_tree.joblib"
    preserved_mtime = preserved.stat().st_mtime_ns
    args.resume = True
    args.jobs = 1
    resumed = pipeline.execute_experiment(args)
    assert resumed == experiment_dir
    assert preserved.stat().st_mtime_ns == preserved_mtime
    assert json.loads((experiment_dir / "_run_manifest.json").read_text())[
        "status"
    ] == "complete"


def test_stop_before_startup_does_not_overwrite_existing_output(tmp_path):
    stop_file = tmp_path / "stop"
    stop_file.touch()
    output_root = tmp_path / "runs"
    output_root.mkdir()
    marker = output_root / "keep"
    marker.write_text("untouched")
    args = pipeline.parse_args(
        [
            "--train-datasets",
            "NCSU",
            "--test-datasets",
            "UNSW1",
            "--output-root",
            str(output_root),
            "--stop-file",
            str(stop_file),
            "--overwrite",
        ]
    )
    pipeline.execute_experiment(args)
    assert marker.read_text() == "untouched"


def test_active_stop_records_stopped_manifest(tmp_path, monkeypatch):
    features = tmp_path / "features"
    labels = tmp_path / "labels"
    label_file = "binary_label.json"
    for dataset, offset in (("NCSU", 0.0), ("UNSW1", 0.5)):
        write_dataset(features, labels, dataset, offset, 2, label_file)
    args = SimpleNamespace(
        train_datasets=["NCSU"],
        test_datasets=["UNSW1"],
        features_root=features,
        labels_root=labels,
        label_file=label_file,
        output_root=tmp_path / "runs",
        seed=42,
        remove_devices=None,
        removal_runs=1,
        jobs=1,
        stop_file=None,
        resume=False,
        overwrite=False,
    )
    monkeypatch.setattr(common, "run_bounded_model_jobs", lambda *args, **kwargs: True)
    experiment_dir = pipeline.execute_experiment(args)
    manifest = json.loads((experiment_dir / "_run_manifest.json").read_text())
    assert manifest["status"] == "stopped"
    assert manifest["completed_model_runs"] == 0
