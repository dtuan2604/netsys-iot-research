from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code_v2 import train_cnn_pn_rf as pipeline


def make_frame() -> pd.DataFrame:
    rows = []
    for dataset in ("NCSU", "UNSW"):
        for label in (0, 1, 2):
            for device_number in range(3):
                device_id = f"{dataset.lower()}-{label}-{device_number}"
                for window in range(2):
                    base = 2 + label * 10 + device_number
                    rows.append(
                        {
                            "dataset": dataset,
                            "capture_id": "date",
                            "window": window,
                            "device_id": device_id,
                            **{
                                feature: float(base + feature_number)
                                for feature_number, feature in enumerate(
                                    pipeline.common.FEATURE_COLUMNS
                                )
                            },
                            "label": label,
                        }
                    )
    return pd.DataFrame(rows)


def write_synthetic_dataset(
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
            labels[device_id] = {
                "device_name": f"device-{label}-{device_number}",
                "is_iot_device": label,
                "device_type": "fixture",
                "vendor": None,
            }
            rows = []
            for capture_number in range(2):
                for window in range(6):
                    center = offset + 1.0 + label * 50.0
                    values = [
                        center + capture_number * 0.001 + window * 0.01 + i
                        for i in range(len(pipeline.common.FEATURE_COLUMNS))
                    ]
                    feature_values = dict(
                        zip(pipeline.common.FEATURE_COLUMNS, values)
                    )
                    if window == 0:
                        feature_values["incoming_mean_frame_size"] = None
                        feature_values["inter_arrival_cv"] = None
                    rows.append(
                        {
                            "capture_id": f"24-01-0{capture_number + 1}",
                            "window": window,
                            "dataset": dataset,
                            "device_id": device_id,
                            "device_id_type": "mac",
                            "is_iot_device": label,
                            **feature_values,
                        }
                    )
            output = features_root / dataset / device_id
            output.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(output / "features.parquet", index=False)
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
            "model_feature_columns": list(pipeline.common.FEATURE_COLUMNS),
            "nullable_feature_columns": list(
                pipeline.common.NULLABLE_FEATURE_COLUMNS
            ),
            "missing_indicator_columns": list(
                pipeline.common.MISSING_INDICATOR_COLUMNS
            ),
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
    assert (
        pipeline.parse_args(required).label_file
        == pipeline.common.DEFAULT_MAC_LABEL_FILE
    )
    assert (
        pipeline.parse_args([*required, "--label-file", "binary_label.json"]).label_file
        == "binary_label.json"
    )
    defaults = pipeline.parse_args(required)
    assert defaults.features_root == Path(
        "built_features_mac_v2/binary-class/window_30s"
    )
    assert defaults.labels_root == Path("labels")
    assert defaults.output_root == Path("classification_runs_mac_v2")
    assert defaults.device == "auto"
    assert defaults.stop_file is None


def test_dataset_lists_must_be_disjoint():
    with pytest.raises(pipeline.PipelineError, match="overlap"):
        pipeline.validate_dataset_sets(["NCSU", "UNSW1"], ["UNSW1"])
    assert pipeline.validate_dataset_sets(["NCSU"], ["YourThing"]) == (
        ["NCSU"],
        ["YourThing"],
    )
    assert pipeline.validate_dataset_sets(["UNSW1", "NCSU"], ["YourThing"]) == (
        ["NCSU", "UNSW1"],
        ["YourThing"],
    )


def test_feature_selection_has_fixed_order_and_log_scaling_is_train_only():
    frame = make_frame()
    matrix = pipeline.raw_feature_matrix(frame)
    expected = np.concatenate(
        [
            frame.loc[:, pipeline.common.FEATURE_COLUMNS].to_numpy(),
            np.zeros((len(frame), len(pipeline.common.MISSING_INDICATOR_COLUMNS))),
        ],
        axis=1,
    )
    np.testing.assert_array_equal(matrix, expected)
    width = len(pipeline.common.MODEL_FEATURE_COLUMNS)
    train = np.asarray([[0.0] * width, [3.0] * width])
    test = np.asarray([[99.0] * width])
    scaler = pipeline.StandardScaler().fit(pipeline.log_features(train))
    transformed_train = train.copy()
    transformed_train[:, : len(pipeline.common.FEATURE_COLUMNS)] = np.log1p(
        transformed_train[:, : len(pipeline.common.FEATURE_COLUMNS)]
    )
    np.testing.assert_allclose(scaler.mean_, transformed_train.mean(axis=0))
    assert not np.allclose(
        scaler.mean_, pipeline.log_features(np.vstack([train, test])).mean(axis=0)
    )


def test_validation_split_holds_out_entire_devices_and_preserves_strata():
    frame = make_frame()
    fit, validation, assignments = pipeline.stratified_device_validation_split(
        frame, 0.2, 42
    )
    fit_devices = set(zip(frame.loc[fit, "dataset"], frame.loc[fit, "device_id"]))
    validation_devices = set(
        zip(frame.loc[validation, "dataset"], frame.loc[validation, "device_id"])
    )
    assert fit_devices.isdisjoint(validation_devices)
    validation_assignments = [
        row for row in assignments if row["split"] == "validation"
    ]
    assert {row["strategy"] for row in assignments} == {"device_holdout"}
    assert {row["capture_id"] for row in assignments} == {None}
    assert {(row["dataset"], row["label"]) for row in validation_assignments} == {
        ("NCSU", 0),
        ("NCSU", 1),
        ("NCSU", 2),
        ("UNSW", 0),
        ("UNSW", 1),
        ("UNSW", 2),
    }


def test_validation_split_falls_back_to_complete_capture_dates_for_singleton():
    frame = make_frame()
    singleton_key = (frame["dataset"] == "UNSW") & (frame["label"] == 2)
    singleton = frame.loc[singleton_key & frame["device_id"].str.endswith("-0")].copy()
    captures = []
    for capture_id in ("date-1", "date-2", "date-3"):
        capture = singleton.copy()
        capture["capture_id"] = capture_id
        captures.append(capture)
    frame = pd.concat([frame.loc[~singleton_key], *captures], ignore_index=True)

    fit, validation, assignments = pipeline.stratified_device_validation_split(
        frame, 0.34, 42
    )
    fallback = [
        row for row in assignments if row["dataset"] == "UNSW" and row["label"] == 2
    ]
    assert {row["strategy"] for row in fallback} == {"capture_date_holdout"}
    assert sum(row["split"] == "validation" for row in fallback) == 1
    assert sum(row["split"] == "fit" for row in fallback) == 2

    current_singleton = (frame["dataset"] == "UNSW") & (frame["label"] == 2)
    fit_captures = set(frame.loc[fit & current_singleton, "capture_id"])
    validation_captures = set(frame.loc[validation & current_singleton, "capture_id"])
    assert fit_captures.isdisjoint(validation_captures)
    assert fit_captures | validation_captures == {"date-1", "date-2", "date-3"}
    repeated = pipeline.stratified_device_validation_split(frame, 0.34, 42)
    np.testing.assert_array_equal(validation, repeated[1])
    assert assignments == repeated[2]


def test_singleton_validation_requires_two_capture_dates():
    frame = make_frame()
    singleton_key = (frame["dataset"] == "UNSW") & (frame["label"] == 2)
    frame = pd.concat(
        [
            frame.loc[~singleton_key],
            frame.loc[singleton_key & frame["device_id"].str.endswith("-0")],
        ],
        ignore_index=True,
    )
    with pytest.raises(
        pipeline.PipelineError,
        match=r"UNSW, class 2, device unsw-2-0: only 1 capture date",
    ):
        pipeline.stratified_device_validation_split(frame, 0.2, 42)


def test_class_weights_embedding_shape_prototypes_and_distances():
    weights = pipeline.calculate_class_weights(np.asarray([0, 1, 1, 2, 2, 2]))
    np.testing.assert_allclose(weights, [2.0, 1.0, 2.0 / 3.0])
    width = len(pipeline.common.MODEL_FEATURE_COLUMNS)
    model = pipeline.CompactCNN(3, width).eval()
    logits, embeddings = model(torch.zeros((5, width)), return_embedding=True)
    assert logits.shape == (5, 3)
    assert embeddings.shape == (5, 32)

    points = np.asarray(
        [[0.0, 0.0], [2.0, 0.0], [10.0, 0.0], [12.0, 0.0], [20.0, 0.0], [22.0, 0.0]]
    )
    labels = np.asarray([0, 0, 1, 1, 2, 2])
    prototypes = pipeline.compute_prototypes(points, labels, [0, 1, 2])
    np.testing.assert_allclose(prototypes, [[1.0, 0.0], [11.0, 0.0], [21.0, 0.0]])
    distances = pipeline.prototype_distances(np.asarray([[0.0, 0.0]]), prototypes)
    np.testing.assert_allclose(distances, [[1.0, 11.0, 21.0]])


def test_random_forest_fit_contract_accepts_only_training_pairs():
    train_distances = np.asarray(
        [
            [0.0, 5.0, 9.0],
            [1.0, 4.0, 8.0],
            [5.0, 0.0, 5.0],
            [4.0, 1.0, 4.0],
            [9.0, 5.0, 0.0],
            [8.0, 4.0, 1.0],
        ]
    )
    train_labels = np.asarray([0, 0, 1, 1, 2, 2])
    classifier = pipeline.fit_random_forest(train_distances, train_labels, 10, 42)
    np.testing.assert_array_equal(classifier.predict(train_distances), train_labels)
    with pytest.raises(pipeline.PipelineError, match="differ in length"):
        pipeline.fit_random_forest(train_distances, train_labels[:2], 10, 42)


def test_auto_device_falls_back_to_cpu_and_explicit_unavailable_errors(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    if hasattr(torch, "xpu"):
        monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
    assert pipeline.resolve_torch_device("auto").type == "cpu"
    with pytest.raises(pipeline.PipelineError, match="CUDA.*unavailable"):
        pipeline.resolve_torch_device("cuda")


def test_training_batch_honors_stop_file(tmp_path):
    stop_file = tmp_path / "stop"
    stop_file.touch()
    width = len(pipeline.common.MODEL_FEATURE_COLUMNS)
    model = pipeline.CompactCNN(2, width)
    loader = pipeline._loader(
        np.zeros((2, width), dtype=np.float32),
        np.asarray([0, 1]),
        2,
        False,
        42,
        0,
    )
    optimizer = torch.optim.Adam(model.parameters())
    with pytest.raises(pipeline.common.StopRequested):
        pipeline._run_training_epoch(
            model,
            loader,
            torch.nn.CrossEntropyLoss(),
            optimizer,
            torch.device("cpu"),
            stop_file,
        )


def test_majority_vote_and_probability_tie_break():
    rows = pd.DataFrame(
        [
            ["D", "a", "c", 0, 1, 0, 0.4, 0.3, 0.3, 1.0, 2.0, 3.0],
            ["D", "a", "c", 1, 1, 1, 0.2, 0.7, 0.1, 2.0, 1.0, 3.0],
            ["D", "b", "c", 0, 2, 1, 0.1, 0.45, 0.45, 3.0, 1.0, 1.0],
            ["D", "b", "c", 1, 2, 2, 0.1, 0.45, 0.45, 3.0, 1.0, 1.0],
        ],
        columns=pipeline.prediction_columns([0, 1, 2]),
    )
    devices = pipeline.aggregate_device_predictions(rows, [0, 1, 2]).set_index(
        "device_id"
    )
    assert devices.loc["a", "tie_break_used"]
    assert devices.loc["a", "predicted_label"] == 1
    assert devices.loc["b", "predicted_label"] == 1


def test_metrics_return_null_for_undefined_values():
    metrics = pipeline.compute_metrics([1, 1], [0, 0], [0.1, 0.2])
    assert metrics["per_class"]["1"]["precision"] is None
    assert metrics["per_class"]["1"]["specificity"] is None
    assert metrics["roc_auc"] is None
    assert metrics["warnings"]


def test_metrics_report_contains_per_dataset_and_combined_results():
    samples = pd.DataFrame(
        [
            ["NCSU", "a", "c", 0, 0, 0, 0.8, 0.1, 0.1, 1.0, 2.0, 3.0],
            ["UNSW", "b", "c", 0, 1, 1, 0.1, 0.8, 0.1, 2.0, 1.0, 3.0],
            ["UNSW", "c", "c", 0, 2, 2, 0.1, 0.1, 0.8, 3.0, 2.0, 1.0],
        ],
        columns=pipeline.prediction_columns([0, 1, 2]),
    )
    devices = pipeline.aggregate_device_predictions(samples, [0, 1, 2])
    report = pipeline.metrics_report(samples, devices, ["NCSU", "UNSW"], [], [0, 1, 2])
    assert set(report["sample_level"]["by_dataset"]) == {"NCSU", "UNSW"}
    assert report["sample_level"]["combined"]["n_samples"] == 3
    assert report["device_level"]["combined"]["n_samples"] == 3


def test_test_window_sampling_uses_hybrid_replacement_rule():
    frame = make_frame()
    without_replacement = pipeline.sample_test_windows(frame, 2, 42)
    assert not without_replacement["replacement_required"].any()
    assert not without_replacement.duplicated(
        ["dataset", "device_id", "source_row"]
    ).any()

    with_replacement = pipeline.sample_test_windows(frame, 3, 42)
    assert with_replacement["replacement_required"].all()
    assert (
        len(with_replacement)
        == frame[["dataset", "device_id"]].drop_duplicates().shape[0] * 3
    )
    pd.testing.assert_frame_equal(
        with_replacement, pipeline.sample_test_windows(frame, 3, 42)
    )


def test_json_is_authoritative_and_parquet_mismatch_fails(tmp_path):
    features = tmp_path / "features"
    labels = tmp_path / "labels"
    write_synthetic_dataset(features, labels, "UNSW", 0.0)
    device = next((features / "UNSW").iterdir())
    frame = pd.read_parquet(device / "features.parquet")
    frame["is_iot_device"] = 1 - frame["is_iot_device"]
    frame.to_parquet(device / "features.parquet", index=False)
    with pytest.raises(pipeline.PipelineError, match="disagree with label JSON"):
        pipeline.load_datasets(["UNSW"], features, labels)


def test_custom_label_file_is_authoritative(tmp_path):
    features = tmp_path / "features"
    labels = tmp_path / "labels"
    write_synthetic_dataset(
        features, labels, "UNSW", 0.0, label_file="binary_label.json"
    )
    loaded = pipeline.load_datasets(["UNSW"], features, labels, "binary_label.json")
    assert sorted(loaded.frame["label"].unique().tolist()) == [0, 1, 2]


def test_feature_manifest_rejects_changed_label_file(tmp_path):
    features = tmp_path / "features"
    labels = tmp_path / "labels"
    write_synthetic_dataset(features, labels, "UNSW", 0.0)
    label_path = labels / "UNSW" / pipeline.common.DEFAULT_MAC_LABEL_FILE
    payload = json.loads(label_path.read_text())
    first = next(iter(payload))
    payload[first]["is_iot_device"] = 1
    label_path.write_text(json.dumps(payload))
    with pytest.raises(pipeline.PipelineError, match="label hash"):
        pipeline.load_datasets(["UNSW"], features, labels)


@pytest.mark.parametrize(
    "label_file", ["", "../labels.json", "labels/file.json", "labels\\file.json"]
)
def test_training_label_file_must_be_a_basename(label_file, tmp_path):
    with pytest.raises(pipeline.PipelineError, match="non-empty filename"):
        pipeline.load_label_map(tmp_path, "UNSW", label_file)


def test_empty_device_is_reported_and_safe_overwrite_is_explicit(tmp_path):
    features = tmp_path / "features"
    labels = tmp_path / "labels"
    write_synthetic_dataset(features, labels, "UNSW", 0.0)
    device = next((features / "UNSW").iterdir())
    full = pd.read_parquet(device / "features.parquet")
    authoritative_label = int(full["is_iot_device"].iloc[0])
    empty = full.iloc[:0]
    empty.to_parquet(device / "features.parquet", index=False)
    loaded = pipeline.load_datasets(["UNSW"], features, labels)
    assert loaded.unevaluable_devices == [
        {
            "dataset": "UNSW",
            "device_id": device.name,
            "true_label": authoritative_label,
            "reason": "no feature rows",
        }
    ]

    run_dir = tmp_path / "run"
    pipeline.prepare_run_directory(run_dir, overwrite=False)
    marker = run_dir / "marker"
    marker.write_text("keep")
    with pytest.raises(pipeline.PipelineError, match="--overwrite"):
        pipeline.prepare_run_directory(run_dir, overwrite=False)
    assert marker.exists()
    pipeline.prepare_run_directory(run_dir, overwrite=True)
    assert run_dir.is_dir() and not marker.exists()


@pytest.mark.parametrize("class_count", [2, 3])
def test_end_to_end_synthetic_run_and_artifact_reload(tmp_path, class_count):
    features = tmp_path / "features"
    labels = tmp_path / "labels"
    output = tmp_path / "runs"
    label_file = "binary_label.json"
    write_synthetic_dataset(features, labels, "NCSU", 0.0, class_count, label_file)
    write_synthetic_dataset(features, labels, "UNSW1", 0.5, class_count, label_file)
    write_synthetic_dataset(
        features, labels, "YourThing", 0.25, class_count, label_file
    )
    args = SimpleNamespace(
        train_datasets=["NCSU", "UNSW1"],
        test_datasets=["YourThing"],
        features_root=features,
        labels_root=labels,
        label_file=label_file,
        output_root=output,
        seed=42,
        device="cpu",
        validation_fraction=0.5,
        batch_size=16,
        max_epochs=2,
        patience=1,
        learning_rate=1e-3,
        weight_decay=1e-4,
        rf_trees=10,
        loader_workers=0,
        remove_devices=[["UNSW1", str(class_count - 1), "2"]],
        removal_runs=2,
        test_trials=3,
        jobs=1,
        overwrite=False,
    )
    run_dir = pipeline.execute_experiment(args)
    manifest = json.loads((run_dir / "_run_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["completed_model_runs"] == 2
    assert manifest["label_file"] == label_file
    assert "features-features" in run_dir.name
    removal_plan = json.loads((run_dir / "removal_plan.json").read_text())
    assert removal_plan[0]["removed_devices"] != removal_plan[1]["removed_devices"]
    model_dir = run_dir / "model_run_001"
    model_manifest = json.loads((model_dir / "_run_manifest.json").read_text())
    assert model_manifest["artifact_reload_verified"] is True
    model_config = json.loads((model_dir / "config.json").read_text())
    assert model_config["label_file"] == label_file
    assert model_config["feature_algorithm_version"] == 2
    assert model_config["input_features"] == 22
    assert model_config["model_feature_columns"] == list(
        pipeline.common.MODEL_FEATURE_COLUMNS
    )
    assert "singleton strata" in model_config["validation_split_policy"]
    validation_metadata = json.loads(
        (model_dir / "validation_devices.json").read_text()
    )
    assert any(
        assignment["strategy"] == "capture_date_holdout"
        for assignment in validation_metadata["assignments"]
    )
    predictions = pd.read_parquet(model_dir / "selected_device_predictions.parquet")
    classes = list(range(class_count))
    expected_per_trial = 3 * class_count
    assert len(predictions) == expected_per_trial * 3
    assert predictions.groupby("trial").size().tolist() == [expected_per_trial] * 3
    assert set(pipeline.probability_columns(classes)) <= set(predictions.columns)
    assert not (model_dir / "sample_predictions.parquet").exists()
    metrics = json.loads((model_dir / "metrics.json").read_text())
    assert metrics["classes"] == classes
    assert len(metrics["trials"]) == 3
    assert metrics["summary"]["combined"]["accuracy"]["mean"] == 1.0
    assert json.loads((model_dir / "label_mapping.json").read_text()) == {
        "classes": classes
    }
