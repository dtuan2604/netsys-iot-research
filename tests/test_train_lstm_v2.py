from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code_v2"))

import train_lstm as pipeline  # noqa: E402


def mac(index: int) -> str:
    return f"02:00:00:00:{index // 256:02x}:{index % 256:02x}"


def label_map(class_count: int, devices_per_class: int = 2):
    result = {}
    index = 1
    for label in range(class_count):
        for _ in range(devices_per_class):
            result[mac(index)] = {
                "device_name": f"Device {index}",
                "is_iot_device": label,
                "device_type": "fixture",
                "vendor": "test",
            }
            index += 1
    return result


def packet_rows(dataset: str, device_id: str, label: int, offset: float = 0.0):
    rows = []
    for date_index, capture_id in enumerate(("24-01-01", "24-01-02")):
        base = date_index * 10_000_000_000
        previous_direction: dict[int, int] = {}
        for packet_index in range(4):
            direction = packet_index % 2
            timestamp = base + packet_index * 1_000_000_000
            same_valid = direction in previous_direction
            rows.append(
                {
                    "dataset": dataset,
                    "capture_id": capture_id,
                    "device_id": device_id,
                    "device_id_type": "mac",
                    "device_name": device_id,
                    "is_iot_device": label,
                    "device_type": "fixture",
                    "vendor": "test",
                    "packet_index": packet_index,
                    "timestamp_ns": timestamp,
                    "frame_size": int(100 + label * 100 + packet_index + offset),
                    "direction": direction,
                    "inter_arrival_time": 0.0 if packet_index == 0 else 1.0,
                    "inter_arrival_valid": packet_index > 0,
                    "dst_unicast": 1,
                    "dst_multicast": 0,
                    "dst_broadcast": 0,
                    "same_direction_inter_arrival_time": (
                        0.0 if not same_valid else 2.0
                    ),
                    "same_direction_iat_valid": same_valid,
                }
            )
            previous_direction[direction] = timestamp
    return rows


def packet_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("dataset", pa.string(), nullable=False),
            pa.field("capture_id", pa.string(), nullable=False),
            pa.field("device_id", pa.string(), nullable=False),
            pa.field("device_id_type", pa.string(), nullable=False),
            pa.field("device_name", pa.string(), nullable=False),
            pa.field("is_iot_device", pa.int32(), nullable=False),
            pa.field("device_type", pa.string()),
            pa.field("vendor", pa.string()),
            pa.field("packet_index", pa.int64(), nullable=False),
            pa.field("timestamp_ns", pa.int64(), nullable=False),
            pa.field("frame_size", pa.int64(), nullable=False),
            pa.field("direction", pa.int8(), nullable=False),
            pa.field("inter_arrival_time", pa.float64(), nullable=False),
            pa.field("inter_arrival_valid", pa.bool_(), nullable=False),
            pa.field("dst_unicast", pa.int8(), nullable=False),
            pa.field("dst_multicast", pa.int8(), nullable=False),
            pa.field("dst_broadcast", pa.int8(), nullable=False),
            pa.field(
                "same_direction_inter_arrival_time", pa.float64(), nullable=False
            ),
            pa.field("same_direction_iat_valid", pa.bool_(), nullable=False),
        ]
    )


def write_packet_fixture(
    root: Path,
    labels_root: Path,
    datasets: list[str],
    class_count: int,
    label_file: str = "fixture_labels.json",
    *,
    devices_per_class: int = 2,
) -> None:
    inventory = {}
    for dataset_index, dataset in enumerate(datasets):
        labels = label_map(class_count, devices_per_class)
        label_dir = labels_root / dataset
        label_dir.mkdir(parents=True, exist_ok=True)
        raw = (json.dumps(labels, sort_keys=True) + "\n").encode()
        (label_dir / label_file).write_bytes(raw)
        inventory[dataset] = {
            "identifier_type": "mac",
            "label_sha256": hashlib.sha256(raw).hexdigest(),
        }
        for device_id, metadata in labels.items():
            rows = packet_rows(
                dataset,
                device_id,
                metadata["is_iot_device"],
                offset=dataset_index,
            )
            path = root / dataset / device_id / "features.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                pa.Table.from_pylist(rows, schema=packet_schema()), path
            )
    manifest = {
        "status": "complete",
        "algorithm_version": 1,
        "configuration": {"label_file": label_file},
        "model_feature_columns": list(pipeline.MODEL_FEATURE_COLUMNS),
        "auxiliary_feature_columns": list(pipeline.AUXILIARY_FEATURE_COLUMNS),
        "datasets": inventory,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "_run_manifest.json").write_text(json.dumps(manifest))


def make_stream(
    dataset: str,
    device: str,
    capture: str,
    label: int,
    length: int,
    value: float = 1.0,
) -> pipeline.PacketStream:
    features = np.zeros((length, 7), dtype=np.float64)
    features[:, 0] = value
    features[:, 1] = np.arange(length) % 2
    features[:, 2] = np.arange(length)
    features[:, 3] = 1
    features[:, 6] = np.arange(length) * 2
    return pipeline.PacketStream(
        dataset,
        device,
        capture,
        label,
        np.arange(length, dtype=np.int64),
        np.arange(length, dtype=np.int64),
        features,
    )


def test_manifest_loading_and_exact_packet_contract(tmp_path):
    features = tmp_path / "features"
    labels = tmp_path / "labels"
    write_packet_fixture(features, labels, ["NCSU"], 2)

    loaded = pipeline.load_packet_datasets(
        ["NCSU"], features, labels, "fixture_labels.json"
    )

    assert len(loaded.streams) == 8
    assert sorted(loaded.device_frame["label"].unique().tolist()) == [0, 1]
    assert not loaded.unevaluable_devices
    assert all(stream.features.shape == (4, 7) for stream in loaded.streams)


def test_manifest_label_hash_and_table_validation_reject_changes(tmp_path):
    features = tmp_path / "features"
    labels = tmp_path / "labels"
    write_packet_fixture(features, labels, ["NCSU"], 2)
    label_path = labels / "NCSU" / "fixture_labels.json"
    label_path.write_text(label_path.read_text() + " ")
    with pytest.raises(pipeline.PipelineError, match="label hash changed"):
        pipeline.validate_packet_feature_manifest(
            features, labels, "fixture_labels.json", ["NCSU"]
        )

    write_packet_fixture(features, labels, ["NCSU"], 2)
    path = next((features / "NCSU").glob("*/features.parquet"))
    frame = pq.read_table(path).to_pandas()
    frame.loc[0, ["dst_unicast", "dst_multicast"]] = [1, 1]
    pq.write_table(pa.Table.from_pandas(frame, schema=packet_schema()), path)
    with pytest.raises(pipeline.PipelineError, match="destination type"):
        pipeline.load_packet_datasets(
            ["NCSU"], features, labels, "fixture_labels.json"
        )


def test_sliding_candidates_cap_boundaries_and_short_padding():
    streams = [
        make_stream("NCSU", "a", "d1", 0, 5),
        make_stream("NCSU", "a", "d2", 0, 2),
        make_stream("NCSU", "b", "d1", 1, 4),
    ]
    candidates, counts = pipeline.build_sequence_candidates(
        streams, 3, 100, 42, "test"
    )
    assert counts == {"NCSU/a": 4, "NCSU/b": 2}
    assert [(c.stream_index, c.start, c.valid_length) for c in candidates] == [
        (0, 0, 3),
        (0, 1, 3),
        (0, 2, 3),
        (1, 0, 2),
        (2, 0, 3),
        (2, 1, 3),
    ]

    capped, capped_counts = pipeline.build_sequence_candidates(
        streams, 3, 2, 42, "cap"
    )
    repeated, _ = pipeline.build_sequence_candidates(streams, 3, 2, 42, "cap")
    assert capped == repeated
    assert capped_counts == {"NCSU/a": 2, "NCSU/b": 2}

    dataset = pipeline.PacketSequenceDataset(streams, [candidates[3]], 3)
    values, label, length = dataset[0]
    assert values.shape == (3, 7)
    assert label == 0 and length == 2
    assert np.array_equal(values[2], np.zeros(7))


def test_split_uses_device_holdout_and_capture_fallback():
    streams = [
        make_stream("NCSU", "a", "d1", 0, 3),
        make_stream("NCSU", "b", "d1", 0, 3),
        make_stream("NCSU", "c", "d1", 1, 3),
        make_stream("NCSU", "c", "d2", 1, 3),
    ]
    fit, validation, assignments = pipeline.split_training_streams(streams, 0.5, 42)
    assert {stream.label for stream in fit} == {0, 1}
    assert {stream.label for stream in validation} == {0, 1}
    assert {row["strategy"] for row in assignments} == {
        "device_holdout",
        "capture_date_holdout",
    }
    assert not {
        (stream.dataset, stream.device_id, stream.capture_id) for stream in fit
    } & {
        (stream.dataset, stream.device_id, stream.capture_id)
        for stream in validation
    }


def test_scaler_changes_continuous_features_only():
    streams = [make_stream("NCSU", "a", "d1", 0, 4, value=10)]
    scaler = pipeline.fit_packet_scaler(streams)
    transformed = pipeline.transform_streams(streams, scaler)[0].features
    assert np.allclose(transformed[:, 0], 0)
    assert np.array_equal(transformed[:, 1], streams[0].features[:, 1])
    assert np.array_equal(transformed[:, 3:6], streams[0].features[:, 3:6])


@pytest.mark.parametrize("model_type", ["lstm", "blstm"])
@pytest.mark.parametrize("classes", [2, 3])
def test_model_shapes_and_masked_packet_targets(model_type, classes):
    model = pipeline.PacketSequenceClassifier(classes, 4, 0.5, model_type)
    values = torch.zeros((2, 5, 7))
    lengths = torch.tensor([5, 2])
    logits = model(values, lengths)
    valid, targets = pipeline._masked_logits_targets(
        logits, torch.tensor([0, classes - 1]), lengths
    )
    assert logits.shape == (2, 5, classes)
    assert valid.shape == (7, classes)
    assert targets.tolist() == [0] * 5 + [classes - 1] * 2
    assert isinstance(model.dropout, torch.nn.Dropout)


class FixedPacketModel(torch.nn.Module):
    def forward(self, values, lengths):
        score = values[:, :, 0]
        return torch.stack([-score, score], dim=-1)


def test_unseen_trials_select_one_sequence_per_device_and_average_packets():
    streams = [
        make_stream("UNSW1", "a", "d1", 1, 5, value=2),
        make_stream("UNSW1", "b", "d1", 0, 2, value=0),
    ]
    scaler = pipeline.fit_packet_scaler(streams)
    config = pipeline.TrainingConfig(
        sequence_packets=3, batch_size=4, max_sequences_per_device=10
    )
    result = pipeline.sample_unseen_sequences(
        streams,
        scaler,
        FixedPacketModel(),
        [0, 1],
        config,
        torch.device("cpu"),
        3,
    )
    assert result.groupby("trial").size().tolist() == [2, 2, 2]
    assert result[result["device_id"] == "a"]["replacement_required"].eq(False).all()
    assert result[result["device_id"] == "b"]["replacement_required"].eq(True).all()
    assert np.allclose(
        result[["probability_class_0", "probability_class_1"]].sum(axis=1), 1
    )


def arguments(tmp_path, class_count: int):
    features = tmp_path / "features"
    labels = tmp_path / "labels"
    write_packet_fixture(
        features, labels, ["NCSU", "UNSW1"], class_count, devices_per_class=2
    )
    return SimpleNamespace(
        train_datasets=["NCSU"],
        test_datasets=["UNSW1"],
        features_root=features,
        labels_root=labels,
        label_file="fixture_labels.json",
        output_root=tmp_path / "output",
        model_type="blstm",
        sequence_packets=3,
        validation_fraction=0.5,
        batch_size=8,
        max_epochs=1,
        patience=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        hidden_size=4,
        dropout=0.1,
        max_sequences_per_device=10,
        loader_workers=0,
        device="cpu",
        test_trials=3,
        seed=42,
        remove_devices=[["NCSU", "1", "1"]],
        removal_runs=1,
        jobs=1,
        stop_file=None,
        resume=False,
        overwrite=False,
    )


@pytest.mark.parametrize("class_count", [2, 3])
def test_end_to_end_binary_and_multiclass_artifacts(tmp_path, class_count):
    args = arguments(tmp_path, class_count)
    output = pipeline.execute_experiment(args)
    manifest = json.loads((output / "_run_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["classes"] == list(range(class_count))
    run = output / "model_run_001"
    run_manifest = json.loads((run / "_run_manifest.json").read_text())
    assert run_manifest["artifact_reload_verified"] is True
    config = json.loads((run / "config.json").read_text())
    assert config["training"]["model_type"] == "blstm"
    assert config["training"]["sequence_packets"] == 3
    predictions = pd.read_parquet(run / "selected_device_predictions.parquet")
    expected_devices = class_count * 2
    assert predictions.groupby("trial").size().tolist() == [expected_devices] * 3
    assert set(pipeline.probability_columns(range(class_count))) <= set(
        predictions.columns
    )
    aggregate = json.loads((output / "aggregate_metrics.json").read_text())
    assert set(aggregate["by_dataset_across_model_run_means"]) == {"UNSW1"}


def test_resume_invalidation_overwrite_safety_and_preexisting_stop(tmp_path):
    args = arguments(tmp_path, 2)
    output = pipeline.execute_experiment(args)
    with pytest.raises(pipeline.PipelineError, match="pass --resume or --overwrite"):
        pipeline.execute_experiment(args)

    incompatible = dataclasses.replace(
        pipeline.TrainingConfig(), sequence_packets=20
    )
    assert incompatible.sequence_packets == 20
    args.resume = True
    args.sequence_packets = 4
    with pytest.raises(pipeline.PipelineError, match="incompatible"):
        pipeline.execute_experiment(args)

    stop = tmp_path / "stop"
    stop.touch()
    marker = output / "marker"
    marker.write_text("keep")
    stopped_args = arguments(tmp_path / "stopped", 2)
    stopped_args.output_root = args.output_root
    stopped_args.stop_file = stop
    stopped_args.overwrite = True
    result = pipeline.execute_experiment(stopped_args)
    assert result == output
    assert marker.read_text() == "keep"
