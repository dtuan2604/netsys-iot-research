#!/usr/bin/env python3
"""Train and evaluate a Layer-2 packet-sequence LSTM or BLSTM."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import platform
import random
import resource
import shutil
import sys
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import sklearn
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

try:
    from . import experiment_common as common
except ImportError:  # pragma: no cover - direct script execution
    import experiment_common as common


PACKET_ALGORITHM_VERSION = 1
DATA_ACCESS_VERSION = 2
DEFAULT_SCAN_BATCH_ROWS = 100_000
MODEL_FEATURE_COLUMNS = (
    "frame_size",
    "direction",
    "inter_arrival_time",
    "dst_unicast",
    "dst_multicast",
    "dst_broadcast",
    "same_direction_inter_arrival_time",
)
AUXILIARY_FEATURE_COLUMNS = (
    "inter_arrival_valid",
    "same_direction_iat_valid",
)
CONTINUOUS_FEATURE_INDEXES = (0, 2, 6)
PACKET_COLUMNS = (
    "dataset",
    "capture_id",
    "device_id",
    "device_id_type",
    "is_iot_device",
    "packet_index",
    "timestamp_ns",
    *MODEL_FEATURE_COLUMNS,
    *AUXILIARY_FEATURE_COLUMNS,
)

torch.set_num_threads(1)


PipelineError = common.PipelineError


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    model_type: str = "blstm"
    sequence_packets: int = 100
    validation_fraction: float = 0.2
    batch_size: int = 128
    max_epochs: int = 15
    patience: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    hidden_size: int = 64
    dropout: float = 0.5
    max_sequences_per_device: int = 5_000
    loader_workers: int = 0


@dataclass
class PacketStream:
    dataset: str
    device_id: str
    capture_id: str
    label: int
    packet_indexes: np.ndarray
    timestamps_ns: np.ndarray
    features: np.ndarray


@dataclass
class LoadedPackets:
    streams: list[PacketStream]
    device_frame: pd.DataFrame
    unevaluable_devices: list[dict[str, Any]]


@dataclass(frozen=True)
class ContinuousStats:
    count: int
    mean: tuple[float, float, float]
    m2: tuple[float, float, float]


@dataclass(frozen=True)
class PacketCapture:
    dataset: str
    device_id: str
    capture_id: str
    label: int
    path: str
    row_offset: int
    packet_count: int
    continuous_stats: ContinuousStats


@dataclass
class LoadedPacketCatalog:
    captures: list[PacketCapture]
    device_frame: pd.DataFrame
    unevaluable_devices: list[dict[str, Any]]
    total_rows: int
    source_bytes: int


@dataclass(frozen=True)
class SequenceCandidate:
    stream_index: int
    start: int
    valid_length: int


@dataclass
class TrainedComponents:
    scaler: StandardScaler
    model: "PacketSequenceClassifier"
    best_epoch: int
    validation_history: list[dict[str, Any]]
    validation_assignments: list[dict[str, Any]]
    final_history: list[dict[str, Any]]
    class_weights: list[float]
    classes: list[int]
    selection_sequence_counts: dict[str, int]
    final_sequence_counts: dict[str, int]


def probability_columns(classes: Sequence[int]) -> list[str]:
    return [f"probability_class_{label}" for label in classes]


def _label_sha256(labels_root: Path, dataset: str, label_file: str) -> str:
    path = labels_root / dataset / common.validate_label_file_name(label_file)
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError as exc:
        raise PipelineError(f"missing label file: {path}") from exc


def validate_packet_feature_manifest(
    features_root: Path,
    labels_root: Path,
    label_file: str,
    datasets: Sequence[str],
) -> dict[str, Any]:
    """Validate the fresh packet-feature contract and authoritative labels."""
    path = features_root / "_run_manifest.json"
    try:
        raw = path.read_bytes()
        manifest = json.loads(raw)
    except FileNotFoundError as exc:
        raise PipelineError(f"missing packet-feature manifest: {path}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PipelineError(f"invalid packet-feature manifest {path}: {exc}") from exc
    if manifest.get("status") != "complete":
        raise PipelineError(f"packet-feature manifest is not complete: {path}")
    if manifest.get("algorithm_version") != PACKET_ALGORITHM_VERSION:
        raise PipelineError(
            f"unsupported packet-feature algorithm version in {path}: "
            f"{manifest.get('algorithm_version')!r}"
        )
    if manifest.get("model_feature_columns") != list(MODEL_FEATURE_COLUMNS):
        raise PipelineError(f"packet model columns do not match the LSTM contract: {path}")
    if manifest.get("auxiliary_feature_columns") != list(AUXILIARY_FEATURE_COLUMNS):
        raise PipelineError(
            f"packet auxiliary columns do not match the LSTM contract: {path}"
        )
    configuration = manifest.get("configuration")
    if not isinstance(configuration, dict) or configuration.get("label_file") != label_file:
        raise PipelineError(f"packet-feature label file does not match: {path}")
    inventory = manifest.get("datasets")
    if not isinstance(inventory, dict):
        raise PipelineError(f"packet-feature manifest has no dataset inventory: {path}")
    label_hashes: dict[str, str] = {}
    for dataset in datasets:
        entry = inventory.get(dataset)
        if not isinstance(entry, dict):
            raise PipelineError(f"packet-feature manifest is missing {dataset}")
        if entry.get("identifier_type") != "mac":
            raise PipelineError(f"{dataset}: packet features are not MAC-addressed")
        digest = _label_sha256(labels_root, dataset, label_file)
        if entry.get("label_sha256") != digest:
            raise PipelineError(f"{dataset}: packet-feature label hash changed")
        label_hashes[dataset] = digest
    return {
        "algorithm_version": PACKET_ALGORITHM_VERSION,
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "label_file": label_file,
        "label_sha256": label_hashes,
        "model_feature_columns": list(MODEL_FEATURE_COLUMNS),
        "auxiliary_feature_columns": list(AUXILIARY_FEATURE_COLUMNS),
    }


def _validate_packet_table(
    table: pa.Table,
    dataset: str,
    device_id: str,
    authoritative_label: int,
    path: Path,
) -> None:
    missing = set(PACKET_COLUMNS) - set(table.column_names)
    if missing:
        raise PipelineError(f"{path}: missing columns {', '.join(sorted(missing))}")
    for column in ("dataset", "capture_id", "device_id", "device_id_type"):
        if not pa.types.is_string(table.schema.field(column).type):
            raise PipelineError(f"{path}: {column} must have string type")
    for column in ("is_iot_device", "packet_index", "timestamp_ns"):
        if not pa.types.is_integer(table.schema.field(column).type):
            raise PipelineError(f"{path}: {column} must have integer type")
    for column in MODEL_FEATURE_COLUMNS:
        if not (
            pa.types.is_integer(table.schema.field(column).type)
            or pa.types.is_floating(table.schema.field(column).type)
        ):
            raise PipelineError(f"{path}: model feature {column} must be numeric")
    for column in AUXILIARY_FEATURE_COLUMNS:
        field = table.schema.field(column)
        if not pa.types.is_boolean(field.type) or table.column(column).null_count:
            raise PipelineError(f"{path}: {column} must be non-null boolean")
    if not table.num_rows:
        return
    constants = {
        "dataset": dataset,
        "device_id": device_id,
        "device_id_type": "mac",
        "is_iot_device": authoritative_label,
    }
    for column, expected in constants.items():
        actual = set(table.column(column).to_pylist())
        if actual != {expected}:
            raise PipelineError(
                f"{path}: {column} values {sorted(actual, key=str)} do not match "
                f"{expected!r}"
            )
    for column in MODEL_FEATURE_COLUMNS:
        values = table.column(column)
        if values.null_count:
            raise PipelineError(f"{path}: model feature {column} contains nulls")
        numeric = pc.cast(values, pa.float64())
        if pc.any(pc.invert(pc.is_finite(numeric))).as_py():
            raise PipelineError(f"{path}: model feature {column} is not finite")
        if pc.any(pc.less(numeric, 0)).as_py():
            raise PipelineError(f"{path}: model feature {column} is negative")


def _streams_from_table(
    table: pa.Table,
    dataset: str,
    device_id: str,
    label: int,
    path: Path,
) -> list[PacketStream]:
    frame = table.select(
        [
            "capture_id",
            "packet_index",
            "timestamp_ns",
            *MODEL_FEATURE_COLUMNS,
            *AUXILIARY_FEATURE_COLUMNS,
        ]
    ).to_pandas()
    streams: list[PacketStream] = []
    for capture_id, group in frame.groupby("capture_id", sort=True):
        indexes = group["packet_index"].to_numpy(dtype=np.int64)
        expected = np.arange(len(group), dtype=np.int64)
        if not np.array_equal(indexes, expected):
            raise PipelineError(
                f"{path}: {capture_id} packet_index must be contiguous from zero"
            )
        timestamps = group["timestamp_ns"].to_numpy(dtype=np.int64)
        if len(timestamps) > 1 and np.any(np.diff(timestamps) < 0):
            raise PipelineError(f"{path}: {capture_id} timestamps are out of order")
        direction = group["direction"].to_numpy(dtype=np.int64)
        if not np.isin(direction, [0, 1]).all():
            raise PipelineError(f"{path}: direction must contain only zero or one")
        destinations = group[
            ["dst_unicast", "dst_multicast", "dst_broadcast"]
        ].to_numpy(dtype=np.int64)
        if not np.isin(destinations, [0, 1]).all() or not np.all(
            destinations.sum(axis=1) == 1
        ):
            raise PipelineError(f"{path}: destination type must be one-hot")
        ordinary_valid = group["inter_arrival_valid"].to_numpy(dtype=bool)
        expected_ordinary = np.arange(len(group)) > 0
        if not np.array_equal(ordinary_valid, expected_ordinary):
            raise PipelineError(f"{path}: invalid inter-arrival validity flags")
        same_valid = group["same_direction_iat_valid"].to_numpy(dtype=bool)
        seen: set[int] = set()
        expected_same = []
        for value in direction:
            expected_same.append(int(value) in seen)
            seen.add(int(value))
        if not np.array_equal(same_valid, np.asarray(expected_same, dtype=bool)):
            raise PipelineError(f"{path}: invalid same-direction validity flags")
        features = group.loc[:, MODEL_FEATURE_COLUMNS].to_numpy(
            dtype=np.float64, copy=True
        )
        if np.any(features[:, 0] <= 0):
            raise PipelineError(f"{path}: frame_size must be positive")
        invalid_ordinary = ~ordinary_valid
        invalid_same = ~same_valid
        if np.any(features[invalid_ordinary, 2] != 0) or np.any(
            features[invalid_same, 6] != 0
        ):
            raise PipelineError(f"{path}: invalid intervals must be encoded as zero")
        streams.append(
            PacketStream(
                dataset=dataset,
                device_id=device_id,
                capture_id=str(capture_id),
                label=int(label),
                packet_indexes=indexes,
                timestamps_ns=timestamps,
                features=features,
            )
        )
    return streams


def load_packet_datasets(
    datasets: Sequence[str],
    features_root: Path,
    labels_root: Path,
    label_file: str,
) -> LoadedPackets:
    validate_packet_feature_manifest(features_root, labels_root, label_file, datasets)
    streams: list[PacketStream] = []
    unevaluable: list[dict[str, Any]] = []
    device_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        labels = common.load_mac_label_map(labels_root, dataset, label_file)
        folder = features_root / dataset
        if not folder.is_dir():
            raise PipelineError(f"missing packet-feature directory: {folder}")
        unexpected = {
            path.name
            for path in folder.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        } - set(labels)
        if unexpected:
            raise PipelineError(
                f"{dataset}: packet-feature folders absent from labels: "
                f"{sorted(unexpected)}"
            )
        for device_id, label in labels.items():
            path = folder / device_id / "features.parquet"
            if not path.is_file():
                raise PipelineError(f"missing packet-feature file: {path}")
            table = pq.read_table(path, columns=list(PACKET_COLUMNS))
            _validate_packet_table(table, dataset, device_id, label, path)
            if not table.num_rows:
                unevaluable.append(
                    {
                        "dataset": dataset,
                        "device_id": device_id,
                        "true_label": int(label),
                        "reason": "no packet rows",
                    }
                )
                continue
            device_rows.append(
                {"dataset": dataset, "device_id": device_id, "label": int(label)}
            )
            streams.extend(
                _streams_from_table(table, dataset, device_id, label, path)
            )
    if not streams:
        raise PipelineError("selected datasets contain no packet rows")
    return LoadedPackets(
        streams=streams,
        device_frame=pd.DataFrame(device_rows).drop_duplicates(ignore_index=True),
        unevaluable_devices=unevaluable,
    )


def _validate_packet_schema(schema: pa.Schema, path: Path) -> None:
    missing = set(PACKET_COLUMNS) - set(schema.names)
    if missing:
        raise PipelineError(f"{path}: missing columns {', '.join(sorted(missing))}")
    for column in ("dataset", "capture_id", "device_id", "device_id_type"):
        if not pa.types.is_string(schema.field(column).type):
            raise PipelineError(f"{path}: {column} must have string type")
    for column in ("is_iot_device", "packet_index", "timestamp_ns"):
        if not pa.types.is_integer(schema.field(column).type):
            raise PipelineError(f"{path}: {column} must have integer type")
    for column in MODEL_FEATURE_COLUMNS:
        column_type = schema.field(column).type
        if not (pa.types.is_integer(column_type) or pa.types.is_floating(column_type)):
            raise PipelineError(f"{path}: model feature {column} must be numeric")
    for column in AUXILIARY_FEATURE_COLUMNS:
        if not pa.types.is_boolean(schema.field(column).type):
            raise PipelineError(f"{path}: {column} must be boolean")


def _merge_continuous_stats(
    left: ContinuousStats | None, right: ContinuousStats
) -> ContinuousStats:
    if left is None or left.count == 0:
        return right
    if right.count == 0:
        return left
    left_mean = np.asarray(left.mean, dtype=np.float64)
    right_mean = np.asarray(right.mean, dtype=np.float64)
    left_m2 = np.asarray(left.m2, dtype=np.float64)
    right_m2 = np.asarray(right.m2, dtype=np.float64)
    total = left.count + right.count
    delta = right_mean - left_mean
    mean = left_mean + delta * (right.count / total)
    m2 = left_m2 + right_m2 + delta * delta * (left.count * right.count / total)
    return ContinuousStats(
        total,
        tuple(float(value) for value in mean),
        tuple(float(value) for value in m2),
    )


def _continuous_stats(values: np.ndarray) -> ContinuousStats:
    if not len(values):
        return ContinuousStats(0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    mean = values.mean(axis=0, dtype=np.float64)
    centered = values - mean
    m2 = np.sum(centered * centered, axis=0, dtype=np.float64)
    return ContinuousStats(
        len(values),
        tuple(float(value) for value in mean),
        tuple(float(value) for value in m2),
    )


def _batch_column(batch: pa.RecordBatch, name: str) -> pa.Array:
    return batch.column(batch.schema.get_field_index(name))


def _all_equal(array: pa.Array, value: Any) -> bool:
    result = pc.all(pc.equal(array, pa.scalar(value, type=array.type)))
    return bool(result.as_py())


def _validate_capture_slice(
    batch: pa.RecordBatch,
    *,
    dataset: str,
    device_id: str,
    capture_id: str,
    label: int,
    capture_position: int,
    last_timestamp: int | None,
    seen_directions: set[int],
    path: Path,
) -> tuple[int | None, set[int], ContinuousStats]:
    for column, expected in (
        ("dataset", dataset),
        ("capture_id", capture_id),
        ("device_id", device_id),
        ("device_id_type", "mac"),
        ("is_iot_device", label),
    ):
        values = _batch_column(batch, column)
        if values.null_count or not _all_equal(values, expected):
            raise PipelineError(
                f"{path}: {column} does not match {expected!r} in {capture_id}"
            )

    indexes = _batch_column(batch, "packet_index").to_numpy(zero_copy_only=False)
    expected_indexes = np.arange(
        capture_position, capture_position + batch.num_rows, dtype=np.int64
    )
    if not np.array_equal(indexes.astype(np.int64, copy=False), expected_indexes):
        raise PipelineError(
            f"{path}: {capture_id} packet_index must be contiguous from zero"
        )
    timestamps = _batch_column(batch, "timestamp_ns").to_numpy(zero_copy_only=False)
    timestamps = timestamps.astype(np.int64, copy=False)
    if len(timestamps):
        if last_timestamp is not None and timestamps[0] < last_timestamp:
            raise PipelineError(f"{path}: {capture_id} timestamps are out of order")
        if len(timestamps) > 1 and np.any(np.diff(timestamps) < 0):
            raise PipelineError(f"{path}: {capture_id} timestamps are out of order")
        last_timestamp = int(timestamps[-1])

    feature_columns = []
    for column in MODEL_FEATURE_COLUMNS:
        values = _batch_column(batch, column)
        if values.null_count:
            raise PipelineError(f"{path}: model feature {column} contains nulls")
        feature_columns.append(values.to_numpy(zero_copy_only=False))
    features = np.column_stack(feature_columns).astype(np.float64, copy=False)
    if not np.isfinite(features).all():
        raise PipelineError(f"{path}: model features are not finite")
    if np.any(features < 0):
        raise PipelineError(f"{path}: model features are negative")
    if np.any(features[:, 0] <= 0):
        raise PipelineError(f"{path}: frame_size must be positive")

    directions = features[:, 1].astype(np.int64, copy=False)
    if not np.isin(directions, [0, 1]).all():
        raise PipelineError(f"{path}: direction must contain only zero or one")
    destinations = features[:, 3:6].astype(np.int64, copy=False)
    if not np.isin(destinations, [0, 1]).all() or not np.all(
        destinations.sum(axis=1) == 1
    ):
        raise PipelineError(f"{path}: destination type must be one-hot")

    ordinary_valid = _batch_column(
        batch, "inter_arrival_valid"
    ).to_numpy(zero_copy_only=False).astype(bool, copy=False)
    expected_ordinary = expected_indexes > 0
    if not np.array_equal(ordinary_valid, expected_ordinary):
        raise PipelineError(f"{path}: invalid inter-arrival validity flags")
    same_valid = _batch_column(
        batch, "same_direction_iat_valid"
    ).to_numpy(zero_copy_only=False).astype(bool, copy=False)
    expected_same = np.empty(len(directions), dtype=bool)
    updated_seen = set(seen_directions)
    for index, direction in enumerate(directions.tolist()):
        expected_same[index] = direction in updated_seen
        updated_seen.add(direction)
    if not np.array_equal(same_valid, expected_same):
        raise PipelineError(f"{path}: invalid same-direction validity flags")
    if np.any(features[~ordinary_valid, 2] != 0) or np.any(
        features[~same_valid, 6] != 0
    ):
        raise PipelineError(f"{path}: invalid intervals must be encoded as zero")
    stats = _continuous_stats(
        np.log1p(features[:, CONTINUOUS_FEATURE_INDEXES])
    )
    return last_timestamp, updated_seen, stats


def load_packet_catalog(
    datasets: Sequence[str],
    features_root: Path,
    labels_root: Path,
    label_file: str,
    *,
    scan_batch_rows: int = DEFAULT_SCAN_BATCH_ROWS,
    stop_file: Path | str | None = None,
) -> LoadedPacketCatalog:
    """Stream-validate packet features and retain only capture metadata/statistics."""
    if scan_batch_rows <= 0:
        raise PipelineError("scan batch rows must be positive")
    validate_packet_feature_manifest(features_root, labels_root, label_file, datasets)
    manifest = common.read_json(features_root / "_run_manifest.json")
    inventory = manifest["datasets"]
    captures: list[PacketCapture] = []
    unevaluable: list[dict[str, Any]] = []
    device_rows: list[dict[str, Any]] = []
    total_rows = 0
    source_bytes = 0
    for dataset in datasets:
        common.check_stop(stop_file)
        labels = common.load_mac_label_map(labels_root, dataset, label_file)
        dataset_entry = inventory.get(dataset, {})
        dates = dataset_entry.get("dates")
        devices = dataset_entry.get("devices")
        if not isinstance(dates, dict) or not isinstance(devices, dict):
            raise PipelineError(
                f"{dataset}: packet manifest lacks date/device row inventory"
            )
        capture_ids = sorted(dates)
        if not capture_ids:
            raise PipelineError(f"{dataset}: packet manifest has no capture dates")
        folder = features_root / dataset
        unexpected = {
            path.name
            for path in folder.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        } - set(labels)
        if unexpected:
            raise PipelineError(
                f"{dataset}: packet-feature folders absent from labels: "
                f"{sorted(unexpected)}"
            )
        for device_id, label in labels.items():
            common.check_stop(stop_file)
            path = folder / device_id / "features.parquet"
            if not path.is_file():
                raise PipelineError(f"missing packet-feature file: {path}")
            parquet = pq.ParquetFile(path)
            _validate_packet_schema(parquet.schema_arrow, path)
            counts = []
            for capture_id in capture_ids:
                date_entry = dates[capture_id]
                row_counts = date_entry.get("row_counts") if isinstance(date_entry, dict) else None
                if not isinstance(row_counts, dict) or device_id not in row_counts:
                    raise PipelineError(
                        f"{dataset}/{capture_id}: missing row count for {device_id}"
                    )
                count = row_counts[device_id]
                if not isinstance(count, int) or count < 0:
                    raise PipelineError(
                        f"{dataset}/{capture_id}/{device_id}: invalid row count"
                    )
                counts.append(count)
            expected_rows = sum(counts)
            if parquet.metadata.num_rows != expected_rows:
                raise PipelineError(
                    f"{path}: row count {parquet.metadata.num_rows} != {expected_rows}"
                )
            device_record = devices.get(device_id)
            if not isinstance(device_record, dict) or device_record.get("row_count") != expected_rows:
                raise PipelineError(f"{dataset}/{device_id}: manifest device row count mismatch")
            source_bytes += path.stat().st_size
            if expected_rows == 0:
                unevaluable.append(
                    {
                        "dataset": dataset,
                        "device_id": device_id,
                        "true_label": int(label),
                        "reason": "no packet rows",
                    }
                )
                continue
            device_rows.append(
                {"dataset": dataset, "device_id": device_id, "label": int(label)}
            )
            offsets = np.cumsum([0, *counts], dtype=np.int64)
            stats_by_capture: list[ContinuousStats | None] = [None] * len(capture_ids)
            last_timestamps: list[int | None] = [None] * len(capture_ids)
            seen_directions: list[set[int]] = [set() for _ in capture_ids]
            global_position = 0
            for batch in parquet.iter_batches(
                batch_size=scan_batch_rows, columns=list(PACKET_COLUMNS)
            ):
                common.check_stop(stop_file)
                local_position = 0
                while local_position < batch.num_rows:
                    capture_index = bisect.bisect_right(
                        offsets.tolist(), global_position
                    ) - 1
                    if capture_index >= len(capture_ids):
                        raise PipelineError(f"{path}: rows exceed manifest boundaries")
                    capture_end = int(offsets[capture_index + 1])
                    take = min(batch.num_rows - local_position, capture_end - global_position)
                    piece = batch.slice(local_position, take)
                    capture_position = global_position - int(offsets[capture_index])
                    last_timestamp, seen, piece_stats = _validate_capture_slice(
                        piece,
                        dataset=dataset,
                        device_id=device_id,
                        capture_id=capture_ids[capture_index],
                        label=int(label),
                        capture_position=capture_position,
                        last_timestamp=last_timestamps[capture_index],
                        seen_directions=seen_directions[capture_index],
                        path=path,
                    )
                    last_timestamps[capture_index] = last_timestamp
                    seen_directions[capture_index] = seen
                    stats_by_capture[capture_index] = _merge_continuous_stats(
                        stats_by_capture[capture_index], piece_stats
                    )
                    local_position += take
                    global_position += take
            if global_position != expected_rows:
                raise PipelineError(f"{path}: streamed row count mismatch")
            for index, (capture_id, count) in enumerate(zip(capture_ids, counts)):
                if not count:
                    continue
                stats = stats_by_capture[index]
                if stats is None or stats.count != count:
                    raise PipelineError(
                        f"{path}: incomplete statistics for {capture_id}"
                    )
                captures.append(
                    PacketCapture(
                        dataset=dataset,
                        device_id=device_id,
                        capture_id=capture_id,
                        label=int(label),
                        path=str(path),
                        row_offset=int(offsets[index]),
                        packet_count=count,
                        continuous_stats=stats,
                    )
                )
            total_rows += expected_rows
    if not captures:
        raise PipelineError("selected datasets contain no packet rows")
    return LoadedPacketCatalog(
        captures=captures,
        device_frame=pd.DataFrame(device_rows).drop_duplicates(ignore_index=True),
        unevaluable_devices=unevaluable,
        total_rows=total_rows,
        source_bytes=source_bytes,
    )


def split_training_streams(
    streams: Sequence[PacketStream], validation_fraction: float, seed: int
) -> tuple[list[PacketStream], list[PacketStream], list[dict[str, Any]]]:
    """Use CNN-PN-RF's device holdout with capture-date fallback."""
    if not 0 < validation_fraction < 1:
        raise PipelineError("validation fraction must be between zero and one")
    devices = pd.DataFrame(
        [
            {"dataset": s.dataset, "device_id": s.device_id, "label": s.label}
            for s in streams
        ]
    ).drop_duplicates(ignore_index=True)
    rng = np.random.default_rng(seed)
    validation_devices: set[tuple[str, str]] = set()
    validation_captures: set[tuple[str, str, str]] = set()
    assignments: list[dict[str, Any]] = []
    for (dataset, label), group in devices.groupby(["dataset", "label"], sort=True):
        identifiers = sorted(group["device_id"].tolist())
        if len(identifiers) >= 2:
            shuffled = np.asarray(identifiers, dtype=object)
            rng.shuffle(shuffled)
            count = max(1, int(round(len(shuffled) * validation_fraction)))
            count = min(count, len(shuffled) - 1)
            selected = set(shuffled[:count].tolist())
            for device_id in identifiers:
                split = "validation" if device_id in selected else "fit"
                assignments.append(
                    {
                        "strategy": "device_holdout",
                        "dataset": dataset,
                        "device_id": device_id,
                        "capture_id": None,
                        "label": int(label),
                        "split": split,
                    }
                )
                if split == "validation":
                    validation_devices.add((str(dataset), device_id))
            continue
        device_id = identifiers[0]
        captures = sorted(
            {
                stream.capture_id
                for stream in streams
                if stream.dataset == dataset
                and stream.device_id == device_id
                and stream.label == label
            }
        )
        if len(captures) < 2:
            raise PipelineError(
                "cannot create capture-date validation for "
                f"{dataset}, class {label}, device {device_id}: "
                f"only {len(captures)} capture date(s)"
            )
        shuffled = np.asarray(captures, dtype=object)
        rng.shuffle(shuffled)
        count = max(1, int(round(len(shuffled) * validation_fraction)))
        count = min(count, len(shuffled) - 1)
        selected = set(shuffled[:count].tolist())
        for capture_id in captures:
            split = "validation" if capture_id in selected else "fit"
            assignments.append(
                {
                    "strategy": "capture_date_holdout",
                    "dataset": dataset,
                    "device_id": device_id,
                    "capture_id": capture_id,
                    "label": int(label),
                    "split": split,
                }
            )
            if split == "validation":
                validation_captures.add((str(dataset), device_id, capture_id))
    fit, validation = [], []
    for stream in streams:
        is_validation = (
            (stream.dataset, stream.device_id) in validation_devices
            or (stream.dataset, stream.device_id, stream.capture_id)
            in validation_captures
        )
        (validation if is_validation else fit).append(stream)
    expected_classes = {stream.label for stream in streams}
    if {stream.label for stream in fit} != expected_classes:
        raise PipelineError("LSTM fit split does not contain every class")
    if {stream.label for stream in validation} != expected_classes:
        raise PipelineError("LSTM validation split does not contain every class")
    return fit, validation, assignments


def _stable_seed(seed: int, *parts: str) -> np.random.SeedSequence:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).digest()
    return np.random.SeedSequence([seed, int.from_bytes(digest[:8], "big")])


def _stream_packet_count(stream: PacketStream | PacketCapture) -> int:
    return (
        int(stream.packet_count)
        if isinstance(stream, PacketCapture)
        else len(stream.features)
    )


def stream_candidate_count(
    stream: PacketStream | PacketCapture, sequence_packets: int
) -> int:
    size = _stream_packet_count(stream)
    if size <= 0:
        return 0
    return 1 if size < sequence_packets else size - sequence_packets + 1


def _ordinal_candidate(
    streams: Sequence[PacketStream],
    stream_indexes: Sequence[int],
    sequence_packets: int,
    ordinal: int,
) -> SequenceCandidate:
    counts = [stream_candidate_count(streams[index], sequence_packets) for index in stream_indexes]
    cumulative = np.cumsum(counts, dtype=np.int64).tolist()
    position = bisect.bisect_right(cumulative, ordinal)
    previous = 0 if position == 0 else cumulative[position - 1]
    stream_index = int(stream_indexes[position])
    stream = streams[stream_index]
    size = _stream_packet_count(stream)
    start = int(ordinal - previous) if size >= sequence_packets else 0
    return SequenceCandidate(
        stream_index=stream_index,
        start=start,
        valid_length=min(sequence_packets, size - start),
    )


def build_sequence_candidates(
    streams: Sequence[PacketStream],
    sequence_packets: int,
    maximum_per_device: int,
    seed: int,
    phase: str,
) -> tuple[list[SequenceCandidate], dict[str, int]]:
    grouped: dict[tuple[str, str], list[int]] = {}
    for index, stream in enumerate(streams):
        grouped.setdefault((stream.dataset, stream.device_id), []).append(index)
    candidates: list[SequenceCandidate] = []
    counts: dict[str, int] = {}
    for (dataset, device_id), indexes in sorted(grouped.items()):
        indexes.sort(key=lambda index: streams[index].capture_id)
        total = sum(
            stream_candidate_count(streams[index], sequence_packets)
            for index in indexes
        )
        take = min(total, maximum_per_device)
        rng = np.random.default_rng(
            _stable_seed(seed, phase, dataset, device_id, str(sequence_packets))
        )
        ordinals = (
            np.arange(total, dtype=np.int64)
            if take == total
            else np.sort(rng.choice(total, size=take, replace=False))
        )
        candidates.extend(
            _ordinal_candidate(streams, indexes, sequence_packets, int(ordinal))
            for ordinal in ordinals
        )
        counts[f"{dataset}/{device_id}"] = int(take)
    if not candidates:
        raise PipelineError("no packet sequences can be constructed")
    return candidates, counts


def _scaler_from_stats(stats: ContinuousStats) -> StandardScaler:
    if stats.count <= 0:
        raise PipelineError("cannot fit packet scaler without packets")
    mean = np.asarray(stats.mean, dtype=np.float64)
    variance = np.maximum(np.asarray(stats.m2, dtype=np.float64) / stats.count, 0)
    scale = np.sqrt(variance)
    scale[scale == 0] = 1.0
    scaler = StandardScaler()
    scaler.mean_ = mean
    scaler.var_ = variance
    scaler.scale_ = scale
    scaler.n_features_in_ = len(CONTINUOUS_FEATURE_INDEXES)
    scaler.n_samples_seen_ = np.int64(stats.count)
    return scaler


def fit_packet_scaler(
    streams: Sequence[PacketStream | PacketCapture],
) -> StandardScaler:
    if streams and isinstance(streams[0], PacketCapture):
        merged: ContinuousStats | None = None
        for stream in streams:
            assert isinstance(stream, PacketCapture)
            merged = _merge_continuous_stats(merged, stream.continuous_stats)
        if merged is None:
            raise PipelineError("cannot fit packet scaler without packets")
        return _scaler_from_stats(merged)
    scaler = StandardScaler()
    for stream in streams:
        assert isinstance(stream, PacketStream)
        values = np.log1p(stream.features[:, CONTINUOUS_FEATURE_INDEXES])
        scaler.partial_fit(values)
    return scaler


def transform_streams(
    streams: Sequence[PacketStream], scaler: StandardScaler
) -> list[PacketStream]:
    transformed = []
    for stream in streams:
        features = stream.features.astype(np.float32, copy=True)
        continuous = np.log1p(
            stream.features[:, CONTINUOUS_FEATURE_INDEXES]
        )
        features[:, CONTINUOUS_FEATURE_INDEXES] = scaler.transform(continuous).astype(
            np.float32
        )
        transformed.append(
            PacketStream(
                dataset=stream.dataset,
                device_id=stream.device_id,
                capture_id=stream.capture_id,
                label=stream.label,
                packet_indexes=stream.packet_indexes,
                timestamps_ns=stream.timestamps_ns,
                features=features,
            )
        )
    return transformed


class PacketSequenceDataset(Dataset):
    def __init__(
        self,
        streams: Sequence[PacketStream],
        candidates: Sequence[SequenceCandidate],
        sequence_packets: int,
    ) -> None:
        self.streams = streams
        self.candidates = candidates
        self.sequence_packets = sequence_packets

    def __len__(self) -> int:
        return len(self.candidates)

    def __getitem__(self, index: int):
        candidate = self.candidates[index]
        stream = self.streams[candidate.stream_index]
        values = np.zeros(
            (self.sequence_packets, len(MODEL_FEATURE_COLUMNS)), dtype=np.float32
        )
        end = candidate.start + candidate.valid_length
        values[: candidate.valid_length] = stream.features[candidate.start:end]
        return values, np.int64(stream.label), np.int64(candidate.valid_length)


class CachedSequenceDataset(Dataset):
    def __init__(
        self,
        path: Path,
        labels: np.ndarray,
        lengths: np.ndarray,
    ) -> None:
        self.path = str(path)
        self.labels = labels.astype(np.int64, copy=False)
        self.lengths = lengths.astype(np.int64, copy=False)
        self._values: np.memmap | None = None

    def __len__(self) -> int:
        return len(self.labels)

    def _array(self) -> np.memmap:
        if self._values is None:
            self._values = np.load(self.path, mmap_mode="r")
        return self._values

    def __getitem__(self, index: int):
        values = np.array(self._array()[index], dtype=np.float32, copy=True)
        return values, self.labels[index], self.lengths[index]

    def close(self) -> None:
        if self._values is not None:
            mmap = getattr(self._values, "_mmap", None)
            if mmap is not None:
                mmap.close()
            self._values = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_values"] = None
        return state


class ParquetWindowReader:
    def __init__(self, maximum_cached_row_groups: int = 4) -> None:
        self.maximum_cached_row_groups = maximum_cached_row_groups
        self.files: dict[str, tuple[pq.ParquetFile, list[int]]] = {}
        self.cache: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()

    def _file(self, path: str) -> tuple[pq.ParquetFile, list[int]]:
        loaded = self.files.get(path)
        if loaded is not None:
            return loaded
        parquet = pq.ParquetFile(path)
        offsets = [0]
        for index in range(parquet.metadata.num_row_groups):
            offsets.append(
                offsets[-1] + parquet.metadata.row_group(index).num_rows
            )
        loaded = (parquet, offsets)
        self.files[path] = loaded
        return loaded

    def _row_group(self, path: str, index: int) -> np.ndarray:
        key = (path, index)
        values = self.cache.pop(key, None)
        if values is not None:
            self.cache[key] = values
            return values
        parquet, _offsets = self._file(path)
        table = parquet.read_row_group(index, columns=list(MODEL_FEATURE_COLUMNS))
        values = np.column_stack(
            [
                table.column(column).to_numpy(zero_copy_only=False)
                for column in MODEL_FEATURE_COLUMNS
            ]
        ).astype(np.float32, copy=False)
        self.cache[key] = values
        while len(self.cache) > self.maximum_cached_row_groups:
            self.cache.popitem(last=False)
        return values

    def read(self, capture: PacketCapture, start: int, length: int) -> np.ndarray:
        absolute = capture.row_offset + start
        remaining = length
        pieces = []
        _parquet, offsets = self._file(capture.path)
        while remaining:
            row_group = bisect.bisect_right(offsets, absolute) - 1
            if row_group < 0 or row_group >= len(offsets) - 1:
                raise PipelineError(
                    f"sequence range exceeds packet source: {capture.path}"
                )
            local = absolute - offsets[row_group]
            take = min(remaining, offsets[row_group + 1] - absolute)
            pieces.append(self._row_group(capture.path, row_group)[local : local + take])
            absolute += take
            remaining -= take
        return pieces[0].copy() if len(pieces) == 1 else np.concatenate(pieces)


def sequence_cache_bytes(candidate_count: int, sequence_packets: int) -> int:
    return candidate_count * sequence_packets * len(MODEL_FEATURE_COLUMNS) * 4


def _check_cache_space(root: Path, required_bytes: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    margin = max(64 * 1024 * 1024, required_bytes // 10)
    if free < required_bytes + margin:
        raise PipelineError(
            "insufficient sequence-cache space: "
            f"required={required_bytes + margin:,} bytes including margin, "
            f"free={free:,} bytes, root={root}"
        )


def materialize_sequence_cache(
    streams: Sequence[PacketCapture],
    candidates: Sequence[SequenceCandidate],
    scaler: StandardScaler,
    sequence_packets: int,
    directory: Path,
) -> CachedSequenceDataset:
    if not candidates:
        raise PipelineError("cannot materialize an empty sequence cache")
    required = sequence_cache_bytes(len(candidates), sequence_packets)
    _check_cache_space(directory.parent, required)
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    path = directory / "features.npy"
    values = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=(len(candidates), sequence_packets, len(MODEL_FEATURE_COLUMNS)),
    )
    labels = np.empty(len(candidates), dtype=np.int64)
    lengths = np.empty(len(candidates), dtype=np.int64)
    reader = ParquetWindowReader()
    try:
        for index, candidate in enumerate(candidates):
            capture = streams[candidate.stream_index]
            raw = reader.read(capture, candidate.start, candidate.valid_length)
            values[index] = 0
            values[index, : candidate.valid_length] = raw
            continuous = np.log1p(
                raw[:, CONTINUOUS_FEATURE_INDEXES].astype(np.float64, copy=False)
            )
            scaled = ((continuous - scaler.mean_) / scaler.scale_).astype(np.float32)
            for position, column in enumerate(CONTINUOUS_FEATURE_INDEXES):
                values[index, : candidate.valid_length, column] = scaled[:, position]
            labels[index] = capture.label
            lengths[index] = candidate.valid_length
        values.flush()
    except BaseException:
        mmap = getattr(values, "_mmap", None)
        if mmap is not None:
            mmap.close()
        shutil.rmtree(directory, ignore_errors=True)
        raise
    mmap = getattr(values, "_mmap", None)
    if mmap is not None:
        mmap.close()
    return CachedSequenceDataset(path, labels, lengths)


@contextmanager
def sequence_cache(
    streams: Sequence[PacketCapture],
    candidates: Sequence[SequenceCandidate],
    scaler: StandardScaler,
    config: TrainingConfig,
    directory: Path,
) -> Iterator[CachedSequenceDataset]:
    dataset = materialize_sequence_cache(
        streams, candidates, scaler, config.sequence_packets, directory
    )
    try:
        yield dataset
    finally:
        dataset.close()
        shutil.rmtree(directory, ignore_errors=True)


class PacketSequenceClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_size: int = 64,
        dropout: float = 0.5,
        model_type: str = "blstm",
    ) -> None:
        super().__init__()
        if model_type not in {"lstm", "blstm"}:
            raise PipelineError(f"unknown model type: {model_type}")
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.dropout_rate = dropout
        self.model_type = model_type
        self.recurrent = nn.LSTM(
            input_size=len(MODEL_FEATURE_COLUMNS),
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=model_type == "blstm",
        )
        width = hidden_size * (2 if model_type == "blstm" else 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(width, num_classes)

    def forward(self, values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(
            values,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.recurrent(packed)
        output, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=values.shape[1]
        )
        return self.classifier(self.dropout(output))


def set_reproducible_seed(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_torch_device(requested: str) -> torch.device:
    def available(name: str) -> bool:
        if name == "cuda":
            return bool(torch.cuda.is_available())
        if name == "mps":
            backend = getattr(torch.backends, "mps", None)
            return bool(backend is not None and backend.is_available())
        if name == "xpu":
            backend = getattr(torch, "xpu", None)
            return bool(backend is not None and backend.is_available())
        return name == "cpu"

    def probe(name: str) -> torch.device:
        device = torch.device(name)
        try:
            _ = (torch.ones(1, device=device) + 1).cpu()
        except Exception as exc:
            raise PipelineError(f"{name.upper()} allocation failed: {exc}") from exc
        return device

    if requested == "auto":
        for candidate in ("cuda", "mps", "xpu"):
            if available(candidate):
                try:
                    return probe(candidate)
                except PipelineError:
                    pass
        return probe("cpu")
    if not available(requested):
        raise PipelineError(f"{requested.upper()} was requested but is unavailable")
    return probe(requested)


def torch_device_description(device: torch.device) -> str:
    if device.type == "cuda":
        return str(torch.cuda.get_device_name(device))
    if device.type == "xpu":
        getter = getattr(getattr(torch, "xpu", None), "get_device_name", None)
        return str(getter(device)) if getter else "Intel XPU"
    if device.type == "mps":
        return "Apple Metal Performance Shaders"
    return platform.processor() or platform.machine() or "CPU"


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _loader(
    streams: Sequence[PacketStream],
    candidates: Sequence[SequenceCandidate],
    config: TrainingConfig,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        PacketSequenceDataset(streams, candidates, config.sequence_packets),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.loader_workers,
        persistent_workers=config.loader_workers > 0,
        generator=generator,
    )


def _dataset_loader(
    dataset: Dataset,
    config: TrainingConfig,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.loader_workers,
        persistent_workers=config.loader_workers > 0,
        generator=generator,
    )


def sequence_class_weights(
    streams: Sequence[PacketStream],
    candidates: Sequence[SequenceCandidate],
    classes: Sequence[int],
) -> np.ndarray:
    counts = np.zeros(len(classes), dtype=np.int64)
    for candidate in candidates:
        counts[streams[candidate.stream_index].label] += candidate.valid_length
    if np.any(counts == 0):
        raise PipelineError(f"all classes must have sequence targets; counts={counts}")
    return (counts.sum() / (len(classes) * counts)).astype(np.float32)


def _masked_logits_targets(
    logits: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(logits.shape[1], device=logits.device)[None, :]
    mask = positions < lengths[:, None]
    targets = labels[:, None].expand(-1, logits.shape[1])
    return logits[mask], targets[mask]


def _training_epoch(
    model: PacketSequenceClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    stop_file: Path | str | None,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    packets = 0
    for values, labels, lengths in loader:
        common.check_stop(stop_file)
        values = values.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(values, lengths)
        valid_logits, targets = _masked_logits_targets(logits, labels, lengths)
        loss = criterion(valid_logits, targets)
        loss.backward()
        optimizer.step()
        count = len(targets)
        total_loss += float(loss.item()) * count
        correct += int((valid_logits.argmax(dim=1) == targets).sum().item())
        packets += count
    return total_loss / packets, correct / packets


@torch.no_grad()
def _evaluation_epoch(
    model: PacketSequenceClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    stop_file: Path | str | None,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    packets = 0
    for values, labels, lengths in loader:
        common.check_stop(stop_file)
        values = values.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)
        logits = model(values, lengths)
        valid_logits, targets = _masked_logits_targets(logits, labels, lengths)
        loss = criterion(valid_logits, targets)
        count = len(targets)
        total_loss += float(loss.item()) * count
        correct += int((valid_logits.argmax(dim=1) == targets).sum().item())
        packets += count
    return total_loss / packets, correct / packets


def select_best_epoch(
    fit_streams: Sequence[PacketStream],
    fit_candidates: Sequence[SequenceCandidate],
    validation_streams: Sequence[PacketStream],
    validation_candidates: Sequence[SequenceCandidate],
    classes: Sequence[int],
    config: TrainingConfig,
    device: torch.device,
    stop_file: Path | str | None,
) -> tuple[int, list[dict[str, Any]]]:
    set_reproducible_seed(config.seed)
    model = PacketSequenceClassifier(
        len(classes), config.hidden_size, config.dropout, config.model_type
    ).to(device)
    weights = sequence_class_weights(fit_streams, fit_candidates, classes)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    fit_loader = _loader(
        fit_streams, fit_candidates, config, shuffle=True, seed=config.seed
    )
    validation_loader = _loader(
        validation_streams,
        validation_candidates,
        config,
        shuffle=False,
        seed=config.seed,
    )
    best_loss = math.inf
    best_epoch = 1
    without_improvement = 0
    history = []
    for epoch in range(1, config.max_epochs + 1):
        train_loss, train_accuracy = _training_epoch(
            model, fit_loader, criterion, optimizer, device, stop_file
        )
        validation_loss, validation_accuracy = _evaluation_epoch(
            model, validation_loader, criterion, device, stop_file
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )
        print(
            f"epoch {epoch:03d}: train_loss={train_loss:.6f} "
            f"val_loss={validation_loss:.6f} val_accuracy={validation_accuracy:.4f}",
            flush=True,
        )
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_epoch = epoch
            without_improvement = 0
        else:
            without_improvement += 1
            if without_improvement >= config.patience:
                break
    return best_epoch, history


def train_final_model(
    streams: Sequence[PacketStream],
    candidates: Sequence[SequenceCandidate],
    classes: Sequence[int],
    epochs: int,
    config: TrainingConfig,
    device: torch.device,
    stop_file: Path | str | None,
) -> tuple[PacketSequenceClassifier, list[float], list[dict[str, Any]]]:
    set_reproducible_seed(config.seed)
    model = PacketSequenceClassifier(
        len(classes), config.hidden_size, config.dropout, config.model_type
    ).to(device)
    weights = sequence_class_weights(streams, candidates, classes)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loader = _loader(streams, candidates, config, shuffle=True, seed=config.seed)
    history = []
    for epoch in range(1, epochs + 1):
        loss, accuracy = _training_epoch(
            model, loader, criterion, optimizer, device, stop_file
        )
        history.append({"epoch": epoch, "loss": loss, "accuracy": accuracy})
        print(
            f"final epoch {epoch:03d}/{epochs:03d}: loss={loss:.6f} "
            f"accuracy={accuracy:.4f}",
            flush=True,
        )
    return model, weights.tolist(), history


def select_best_epoch_cached(
    fit_dataset: Dataset,
    validation_dataset: Dataset,
    weights: np.ndarray,
    classes: Sequence[int],
    config: TrainingConfig,
    device: torch.device,
    stop_file: Path | str | None,
) -> tuple[int, list[dict[str, Any]]]:
    set_reproducible_seed(config.seed)
    model = PacketSequenceClassifier(
        len(classes), config.hidden_size, config.dropout, config.model_type
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    fit_loader = _dataset_loader(
        fit_dataset, config, shuffle=True, seed=config.seed
    )
    validation_loader = _dataset_loader(
        validation_dataset, config, shuffle=False, seed=config.seed
    )
    best_loss = math.inf
    best_epoch = 1
    without_improvement = 0
    history = []
    for epoch in range(1, config.max_epochs + 1):
        train_loss, train_accuracy = _training_epoch(
            model, fit_loader, criterion, optimizer, device, stop_file
        )
        validation_loss, validation_accuracy = _evaluation_epoch(
            model, validation_loader, criterion, device, stop_file
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )
        print(
            f"epoch {epoch:03d}: train_loss={train_loss:.6f} "
            f"val_loss={validation_loss:.6f} val_accuracy={validation_accuracy:.4f}",
            flush=True,
        )
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_epoch = epoch
            without_improvement = 0
        else:
            without_improvement += 1
            if without_improvement >= config.patience:
                break
    return best_epoch, history


def train_final_model_cached(
    dataset: Dataset,
    weights: np.ndarray,
    classes: Sequence[int],
    epochs: int,
    config: TrainingConfig,
    device: torch.device,
    stop_file: Path | str | None,
) -> tuple[PacketSequenceClassifier, list[float], list[dict[str, Any]]]:
    set_reproducible_seed(config.seed)
    model = PacketSequenceClassifier(
        len(classes), config.hidden_size, config.dropout, config.model_type
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loader = _dataset_loader(dataset, config, shuffle=True, seed=config.seed)
    history = []
    for epoch in range(1, epochs + 1):
        loss, accuracy = _training_epoch(
            model, loader, criterion, optimizer, device, stop_file
        )
        history.append({"epoch": epoch, "loss": loss, "accuracy": accuracy})
        print(
            f"final epoch {epoch:03d}/{epochs:03d}: loss={loss:.6f} "
            f"accuracy={accuracy:.4f}",
            flush=True,
        )
    return model, weights.tolist(), history


def train_components(
    streams: Sequence[PacketStream],
    classes: Sequence[int],
    config: TrainingConfig,
    device: torch.device,
    stop_file: Path | str | None,
) -> TrainedComponents:
    fit_raw, validation_raw, assignments = split_training_streams(
        streams, config.validation_fraction, config.seed
    )
    selection_scaler = fit_packet_scaler(fit_raw)
    fit = transform_streams(fit_raw, selection_scaler)
    validation = transform_streams(validation_raw, selection_scaler)
    fit_candidates, fit_counts = build_sequence_candidates(
        fit,
        config.sequence_packets,
        config.max_sequences_per_device,
        config.seed,
        "selection-fit",
    )
    validation_candidates, validation_counts = build_sequence_candidates(
        validation,
        config.sequence_packets,
        config.max_sequences_per_device,
        config.seed,
        "selection-validation",
    )
    best_epoch, validation_history = select_best_epoch(
        fit,
        fit_candidates,
        validation,
        validation_candidates,
        classes,
        config,
        device,
        stop_file,
    )
    print(f"selected epoch: {best_epoch}; refitting on all training packets", flush=True)
    final_scaler = fit_packet_scaler(streams)
    transformed = transform_streams(streams, final_scaler)
    final_candidates, final_counts = build_sequence_candidates(
        transformed,
        config.sequence_packets,
        config.max_sequences_per_device,
        config.seed,
        "final-training",
    )
    model, class_weights, final_history = train_final_model(
        transformed,
        final_candidates,
        classes,
        best_epoch,
        config,
        device,
        stop_file,
    )
    return TrainedComponents(
        scaler=final_scaler,
        model=model,
        best_epoch=best_epoch,
        validation_history=validation_history,
        validation_assignments=assignments,
        final_history=final_history,
        class_weights=class_weights,
        classes=list(classes),
        selection_sequence_counts={
            **{f"fit:{key}": value for key, value in fit_counts.items()},
            **{
                f"validation:{key}": value
                for key, value in validation_counts.items()
            },
        },
        final_sequence_counts=final_counts,
    )


def train_catalog_components(
    captures: Sequence[PacketCapture],
    classes: Sequence[int],
    config: TrainingConfig,
    device: torch.device,
    stop_file: Path | str | None,
    cache_workspace: Path,
) -> TrainedComponents:
    fit_raw, validation_raw, assignments = split_training_streams(
        captures, config.validation_fraction, config.seed
    )
    selection_scaler = fit_packet_scaler(fit_raw)
    fit_candidates, fit_counts = build_sequence_candidates(
        fit_raw,
        config.sequence_packets,
        config.max_sequences_per_device,
        config.seed,
        "selection-fit",
    )
    validation_candidates, validation_counts = build_sequence_candidates(
        validation_raw,
        config.sequence_packets,
        config.max_sequences_per_device,
        config.seed,
        "selection-validation",
    )
    selection_weights = sequence_class_weights(
        fit_raw, fit_candidates, classes
    )
    with sequence_cache(
        fit_raw,
        fit_candidates,
        selection_scaler,
        config,
        cache_workspace / "selection-fit",
    ) as fit_dataset, sequence_cache(
        validation_raw,
        validation_candidates,
        selection_scaler,
        config,
        cache_workspace / "selection-validation",
    ) as validation_dataset:
        best_epoch, validation_history = select_best_epoch_cached(
            fit_dataset,
            validation_dataset,
            selection_weights,
            classes,
            config,
            device,
            stop_file,
        )
    print(f"selected epoch: {best_epoch}; refitting on all training packets", flush=True)
    final_scaler = fit_packet_scaler(captures)
    final_candidates, final_counts = build_sequence_candidates(
        captures,
        config.sequence_packets,
        config.max_sequences_per_device,
        config.seed,
        "final-training",
    )
    final_weights = sequence_class_weights(captures, final_candidates, classes)
    with sequence_cache(
        captures,
        final_candidates,
        final_scaler,
        config,
        cache_workspace / "final-training",
    ) as final_dataset:
        model, class_weights, final_history = train_final_model_cached(
            final_dataset,
            final_weights,
            classes,
            best_epoch,
            config,
            device,
            stop_file,
        )
    return TrainedComponents(
        scaler=final_scaler,
        model=model,
        best_epoch=best_epoch,
        validation_history=validation_history,
        validation_assignments=assignments,
        final_history=final_history,
        class_weights=class_weights,
        classes=list(classes),
        selection_sequence_counts={
            **{f"fit:{key}": value for key, value in fit_counts.items()},
            **{
                f"validation:{key}": value
                for key, value in validation_counts.items()
            },
        },
        final_sequence_counts=final_counts,
    )


@torch.no_grad()
def predict_candidates(
    streams: Sequence[PacketStream],
    candidates: Sequence[SequenceCandidate],
    scaler: StandardScaler,
    model: PacketSequenceClassifier,
    config: TrainingConfig,
    device: torch.device,
    stop_file: Path | str | None = None,
) -> np.ndarray:
    transformed = transform_streams(streams, scaler)
    loader = _loader(transformed, candidates, config, shuffle=False, seed=0)
    model.eval()
    chunks = []
    for values, _labels, lengths in loader:
        common.check_stop(stop_file)
        values = values.to(device)
        lengths = lengths.to(device)
        probabilities = torch.softmax(model(values, lengths), dim=-1)
        positions = torch.arange(probabilities.shape[1], device=device)[None, :]
        mask = positions < lengths[:, None]
        summed = (probabilities * mask[:, :, None]).sum(dim=1)
        chunks.append((summed / lengths[:, None]).cpu().numpy())
    return np.concatenate(chunks).astype(np.float64, copy=False)


def sample_unseen_sequences(
    streams: Sequence[PacketStream],
    scaler: StandardScaler,
    model: PacketSequenceClassifier,
    classes: Sequence[int],
    config: TrainingConfig,
    device: torch.device,
    test_trials: int,
    stop_file: Path | str | None = None,
) -> pd.DataFrame:
    if test_trials <= 0:
        raise PipelineError("test trials must be positive")
    grouped: dict[tuple[str, str], list[int]] = {}
    for index, stream in enumerate(streams):
        grouped.setdefault((stream.dataset, stream.device_id), []).append(index)
    selections: list[tuple[int, SequenceCandidate, bool]] = []
    unique: dict[SequenceCandidate, int] = {}
    unique_candidates: list[SequenceCandidate] = []
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(config.seed + 10_000)
    for (dataset, device_id), indexes in sorted(grouped.items()):
        indexes.sort(key=lambda index: streams[index].capture_id)
        total = sum(
            stream_candidate_count(streams[index], config.sequence_packets)
            for index in indexes
        )
        replacement = total < test_trials
        chosen = rng.choice(total, size=test_trials, replace=replacement)
        for trial, ordinal in enumerate(chosen.tolist(), start=1):
            candidate = _ordinal_candidate(
                streams, indexes, config.sequence_packets, int(ordinal)
            )
            if candidate not in unique:
                unique[candidate] = len(unique_candidates)
                unique_candidates.append(candidate)
            selections.append((unique[candidate], candidate, replacement))
            stream = streams[candidate.stream_index]
            rows.append(
                {
                    "trial": trial,
                    "dataset": dataset,
                    "device_id": device_id,
                    "capture_id": stream.capture_id,
                    "start_packet_index": int(
                        stream.packet_indexes[candidate.start]
                    ),
                    "valid_length": candidate.valid_length,
                    "replacement_required": replacement,
                    "true_label": stream.label,
                }
            )
    probabilities = predict_candidates(
        streams,
        unique_candidates,
        scaler,
        model,
        config,
        device,
        stop_file,
    )
    for row, (prediction_index, _candidate, _replacement) in zip(rows, selections):
        values = probabilities[prediction_index]
        row["predicted_label"] = int(np.argmax(values))
        for index, label in enumerate(classes):
            row[f"probability_class_{label}"] = float(values[index])
    return pd.DataFrame(rows).sort_values(
        ["trial", "dataset", "device_id"], ignore_index=True
    )


@torch.no_grad()
def predict_cached_dataset(
    dataset: Dataset,
    model: PacketSequenceClassifier,
    config: TrainingConfig,
    device: torch.device,
    stop_file: Path | str | None = None,
) -> np.ndarray:
    loader = _dataset_loader(dataset, config, shuffle=False, seed=0)
    model.eval()
    chunks = []
    for values, _labels, lengths in loader:
        common.check_stop(stop_file)
        values = values.to(device)
        lengths = lengths.to(device)
        probabilities = torch.softmax(model(values, lengths), dim=-1)
        positions = torch.arange(probabilities.shape[1], device=device)[None, :]
        mask = positions < lengths[:, None]
        summed = (probabilities * mask[:, :, None]).sum(dim=1)
        chunks.append((summed / lengths[:, None]).cpu().numpy())
    return np.concatenate(chunks).astype(np.float64, copy=False)


def sample_unseen_catalog(
    captures: Sequence[PacketCapture],
    scaler: StandardScaler,
    model: PacketSequenceClassifier,
    classes: Sequence[int],
    config: TrainingConfig,
    device: torch.device,
    test_trials: int,
    cache_workspace: Path,
    phase: str,
    stop_file: Path | str | None = None,
) -> pd.DataFrame:
    if test_trials <= 0:
        raise PipelineError("test trials must be positive")
    grouped: dict[tuple[str, str], list[int]] = {}
    for index, capture in enumerate(captures):
        grouped.setdefault((capture.dataset, capture.device_id), []).append(index)
    selections: list[tuple[int, SequenceCandidate, bool]] = []
    unique: dict[SequenceCandidate, int] = {}
    unique_candidates: list[SequenceCandidate] = []
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(config.seed + 10_000)
    for (dataset, device_id), indexes in sorted(grouped.items()):
        indexes.sort(key=lambda index: captures[index].capture_id)
        total = sum(
            stream_candidate_count(captures[index], config.sequence_packets)
            for index in indexes
        )
        replacement = total < test_trials
        chosen = rng.choice(total, size=test_trials, replace=replacement)
        for trial, ordinal in enumerate(chosen.tolist(), start=1):
            candidate = _ordinal_candidate(
                captures, indexes, config.sequence_packets, int(ordinal)
            )
            if candidate not in unique:
                unique[candidate] = len(unique_candidates)
                unique_candidates.append(candidate)
            selections.append((unique[candidate], candidate, replacement))
            capture = captures[candidate.stream_index]
            rows.append(
                {
                    "trial": trial,
                    "dataset": dataset,
                    "device_id": device_id,
                    "capture_id": capture.capture_id,
                    "start_packet_index": candidate.start,
                    "valid_length": candidate.valid_length,
                    "replacement_required": replacement,
                    "true_label": capture.label,
                }
            )
    with sequence_cache(
        captures,
        unique_candidates,
        scaler,
        config,
        cache_workspace / phase,
    ) as dataset:
        probabilities = predict_cached_dataset(
            dataset, model, config, device, stop_file
        )
    for row, (prediction_index, _candidate, _replacement) in zip(rows, selections):
        values = probabilities[prediction_index]
        row["predicted_label"] = int(np.argmax(values))
        for index, label in enumerate(classes):
            row[f"probability_class_{label}"] = float(values[index])
    return pd.DataFrame(rows).sort_values(
        ["trial", "dataset", "device_id"], ignore_index=True
    )


def trial_report(
    predictions: pd.DataFrame,
    test_datasets: Sequence[str],
    unevaluable_devices: Sequence[Mapping[str, Any]],
    classes: Sequence[int],
) -> dict[str, Any]:
    reports = []
    for trial, group in predictions.groupby("trial", sort=True):
        evaluated = common.evaluate_predictions(group, test_datasets, classes)
        reports.append(
            {
                "trial": int(trial),
                "combined": evaluated["combined"],
                "by_dataset": evaluated["by_dataset"],
            }
        )
    return {
        "classes": list(classes),
        "evaluation_unit": (
            "one randomly selected packet sequence per device; device probability "
            "is the mean of valid packet probabilities"
        ),
        "trials": reports,
        "summary": {
            "combined": common.summarize_metric_dicts(
                [report["combined"] for report in reports]
            ),
            "by_dataset": {
                dataset: common.summarize_metric_dicts(
                    [report["by_dataset"][dataset] for report in reports]
                )
                for dataset in test_datasets
            },
        },
        "unevaluable_devices": list(unevaluable_devices),
    }


def _filter_removed_streams(
    streams: Sequence[PacketStream], removed: Sequence[Mapping[str, Any]]
) -> list[PacketStream]:
    remove_keys = {
        (str(entry["dataset"]), str(device_id))
        for entry in removed
        for device_id in entry["device_ids"]
    }
    result = [
        stream
        for stream in streams
        if (stream.dataset, stream.device_id) not in remove_keys
    ]
    if {stream.label for stream in result} != {stream.label for stream in streams}:
        raise PipelineError("device removal eliminated a training class")
    return result


def stream_counts(streams: Sequence[PacketStream | PacketCapture]) -> dict[str, Any]:
    devices = {(stream.dataset, stream.device_id, stream.label) for stream in streams}
    return {
        "streams": len(streams),
        "packets": int(sum(_stream_packet_count(stream) for stream in streams)),
        "devices": len(devices),
        "devices_by_class": {
            str(label): sum(1 for _dataset, _device, value in devices if value == label)
            for label in sorted({value for _dataset, _device, value in devices})
        },
    }


def _save_components(
    run_dir: Path,
    components: TrainedComponents,
    payload: Mapping[str, Any],
) -> None:
    torch.save(components.model.state_dict(), run_dir / "lstm_state.pt")
    joblib.dump(components.scaler, run_dir / "scaler.joblib")
    common.atomic_write_json(run_dir / "label_mapping.json", {"classes": components.classes})
    common.atomic_write_json(run_dir / "config.json", payload)
    pd.DataFrame(components.validation_history).to_csv(
        run_dir / "validation_history.csv", index=False
    )
    pd.DataFrame(components.final_history).to_csv(
        run_dir / "final_training_history.csv", index=False
    )
    common.atomic_write_json(
        run_dir / "validation_assignments.json",
        {"assignments": components.validation_assignments},
    )


def _load_components(
    run_dir: Path, device: torch.device
) -> tuple[StandardScaler, PacketSequenceClassifier, list[int], TrainingConfig]:
    payload = common.read_json(run_dir / "config.json")
    training = dict(payload["training"])
    config = TrainingConfig(**training)
    classes = [int(label) for label in payload["classes"]]
    if payload.get("model_feature_columns") != list(MODEL_FEATURE_COLUMNS):
        raise PipelineError("saved LSTM has an incompatible feature contract")
    scaler = joblib.load(run_dir / "scaler.joblib")
    if int(scaler.n_features_in_) != len(CONTINUOUS_FEATURE_INDEXES):
        raise PipelineError("saved LSTM scaler has an incompatible width")
    model = PacketSequenceClassifier(
        len(classes), config.hidden_size, config.dropout, config.model_type
    ).to(device)
    state = torch.load(run_dir / "lstm_state.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return scaler, model, classes, config


def _execute_model_run(
    run_entry: Mapping[str, Any],
    experiment_dir: Path,
    train_streams: Sequence[PacketCapture],
    test_streams: Sequence[PacketCapture],
    unevaluable_devices: Sequence[Mapping[str, Any]],
    train_datasets: Sequence[str],
    test_datasets: Sequence[str],
    features_root: Path,
    labels_root: Path,
    label_file: str,
    classes: Sequence[int],
    config: TrainingConfig,
    device_name: str,
    test_trials: int,
    stop_file: str | None,
    cache_root: str,
    cache_namespace: str,
    scan_batch_rows: int,
) -> dict[str, Any]:
    run_number = int(run_entry["run"])
    run_dir, final_run_dir = common.create_model_run_staging_directory(
        experiment_dir, run_number
    )
    cache_workspace = (
        Path(cache_root) / cache_namespace / f"model_run_{run_number:03d}"
    )
    shutil.rmtree(cache_workspace, ignore_errors=True)
    cache_workspace.mkdir(parents=True, exist_ok=True)
    common.atomic_write_json(run_dir / "_run_manifest.json", {"status": "running"})
    try:
        common.check_stop(stop_file)
        device = resolve_torch_device(device_name)
        reduced = _filter_removed_streams(
            train_streams, run_entry["removed_devices"]
        )
        print(
            f"model run {run_number}: training streams={len(reduced):,}; "
            f"torch_device={device}",
            flush=True,
        )
        components = train_catalog_components(
            reduced, classes, config, device, stop_file, cache_workspace
        )
        predictions = sample_unseen_catalog(
            test_streams,
            components.scaler,
            components.model,
            classes,
            config,
            device,
            test_trials,
            cache_workspace,
            "unseen-test",
            stop_file,
        )
        predictions.insert(0, "model_run", run_number)
        report = trial_report(
            predictions, test_datasets, unevaluable_devices, classes
        )
        payload = {
            "pipeline": "packet_sequence_lstm_mac_v2",
            "training": asdict(config),
            "train_datasets": list(train_datasets),
            "test_datasets": list(test_datasets),
            "features_root": str(features_root),
            "labels_root": str(labels_root),
            "label_file": label_file,
            "packet_feature_algorithm_version": PACKET_ALGORITHM_VERSION,
            "model_feature_columns": list(MODEL_FEATURE_COLUMNS),
            "auxiliary_feature_columns": list(AUXILIARY_FEATURE_COLUMNS),
            "continuous_preprocessing_columns": [
                MODEL_FEATURE_COLUMNS[index]
                for index in CONTINUOUS_FEATURE_INDEXES
            ],
            "classes": list(classes),
            "removed_devices": run_entry["removed_devices"],
            "training_counts_after_removal": stream_counts(reduced),
            "test_trials": test_trials,
            "test_sampling_seed": config.seed + 10_000,
            "test_sampling_rule": (
                "one sliding sequence per device/trial; without replacement when "
                "candidate sequences >= trials, with replacement otherwise"
            ),
            "device_probability_rule": "mean valid-packet softmax probability",
            "validation_split_policy": (
                "device holdout per dataset/class with capture-date fallback for "
                "singleton strata"
            ),
            "best_epoch": components.best_epoch,
            "class_weights": components.class_weights,
            "selection_sequence_counts": components.selection_sequence_counts,
            "final_sequence_counts": components.final_sequence_counts,
            "torch_device": str(device),
            "data_access": {
                "version": DATA_ACCESS_VERSION,
                "mode": "streamed_catalog_disk_sequence_cache",
                "scan_batch_rows": scan_batch_rows,
                "peak_rss_bytes": peak_rss_bytes(),
            },
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "pyarrow": pa.__version__,
                "scikit_learn": sklearn.__version__,
                "torch": torch.__version__,
            },
        }
        _save_components(run_dir, components, payload)
        common.atomic_write_json(
            run_dir / "removed_devices.json", run_entry["removed_devices"]
        )
        predictions.to_parquet(
            run_dir / "selected_device_predictions.parquet",
            index=False,
            compression="zstd",
        )
        common.atomic_write_json(run_dir / "metrics.json", report)
        metric_rows = []
        for trial_result in report["trials"]:
            trial = trial_result["trial"]
            metric_rows.append(
                common.flatten_metrics(
                    "combined", run_number, trial, trial_result["combined"]
                )
            )
            metric_rows.extend(
                common.flatten_metrics(
                    dataset,
                    run_number,
                    trial,
                    trial_result["by_dataset"][dataset],
                )
                for dataset in test_datasets
            )
        pd.DataFrame(metric_rows).to_parquet(
            run_dir / "trial_metrics.parquet", index=False, compression="zstd"
        )
        reloaded_scaler, reloaded_model, reloaded_classes, reloaded_config = (
            _load_components(run_dir, device)
        )
        reloaded_predictions = sample_unseen_catalog(
            test_streams,
            reloaded_scaler,
            reloaded_model,
            reloaded_classes,
            reloaded_config,
            device,
            test_trials,
            cache_workspace,
            "artifact-reload-test",
        )
        expected = predictions.drop(columns=["model_run"]).reset_index(drop=True)
        if not np.array_equal(
            expected["predicted_label"].to_numpy(),
            reloaded_predictions["predicted_label"].to_numpy(),
        ) or not np.allclose(
            expected.loc[:, probability_columns(classes)].to_numpy(),
            reloaded_predictions.loc[:, probability_columns(classes)].to_numpy(),
            rtol=0,
            atol=1e-12,
        ):
            raise PipelineError("saved LSTM artifacts do not reproduce predictions")
        common.check_stop(stop_file)
        common.atomic_write_json(
            run_dir / "_run_manifest.json",
            {
                "status": "complete",
                "run": run_number,
                "best_epoch": components.best_epoch,
                "evaluable_test_devices": int(
                    predictions[["dataset", "device_id"]].drop_duplicates().shape[0]
                ),
                "unevaluable_test_devices": len(unevaluable_devices),
                "test_trials": test_trials,
                "artifact_reload_verified": True,
                "training_counts_after_removal": stream_counts(reduced),
                "peak_rss_bytes": peak_rss_bytes(),
            },
        )
        common.check_stop(stop_file)
        common.commit_model_run(run_dir, final_run_dir)
        return {
            "run": run_number,
            "run_dir": str(final_run_dir),
            "summary": report["summary"],
            "metric_rows": metric_rows,
        }
    except common.StopRequested:
        common.discard_model_run(run_dir)
        raise
    except BaseException:
        common.discard_model_run(run_dir)
        raise
    finally:
        shutil.rmtree(cache_workspace, ignore_errors=True)


def _load_completed_result(run_dir: Path) -> dict[str, Any]:
    if not common.completed_model_run(run_dir):
        raise PipelineError(f"incomplete model-run checkpoint: {run_dir}")
    for name in (
        "lstm_state.pt",
        "scaler.joblib",
        "config.json",
        "selected_device_predictions.parquet",
        "metrics.json",
        "trial_metrics.parquet",
    ):
        if not (run_dir / name).is_file():
            raise PipelineError(f"incomplete model-run checkpoint: {run_dir / name}")
    report = common.read_json(run_dir / "metrics.json")
    return {
        "run": int(run_dir.name.rsplit("_", 1)[1]),
        "run_dir": str(run_dir),
        "summary": report["summary"],
        "metric_rows": pd.read_parquet(run_dir / "trial_metrics.parquet").to_dict(
            "records"
        ),
    }


def _capped_candidate_total(
    captures: Sequence[PacketCapture], config: TrainingConfig
) -> int:
    grouped: dict[tuple[str, str], int] = {}
    for capture in captures:
        key = (capture.dataset, capture.device_id)
        grouped[key] = grouped.get(key, 0) + stream_candidate_count(
            capture, config.sequence_packets
        )
    return sum(min(total, config.max_sequences_per_device) for total in grouped.values())


def estimated_peak_cache_bytes(
    train_captures: Sequence[PacketCapture],
    test_captures: Sequence[PacketCapture],
    config: TrainingConfig,
    test_trials: int,
    jobs: int,
) -> int:
    train_candidates = _capped_candidate_total(train_captures, config)
    test_devices = len({(item.dataset, item.device_id) for item in test_captures})
    per_run_candidates = max(train_candidates * 2, test_devices * test_trials)
    return sequence_cache_bytes(
        per_run_candidates * max(1, jobs), config.sequence_packets
    )


def execute_experiment(args: argparse.Namespace) -> Path:
    train_datasets, test_datasets = common.validate_dataset_sets(
        args.train_datasets, args.test_datasets
    )
    features_root = args.features_root.resolve()
    labels_root = args.labels_root.resolve()
    label_file = common.validate_label_file_name(args.label_file)
    config = TrainingConfig(
        seed=args.seed,
        model_type=args.model_type,
        sequence_packets=args.sequence_packets,
        validation_fraction=args.validation_fraction,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        max_sequences_per_device=args.max_sequences_per_device,
        loader_workers=args.loader_workers,
    )
    if config.sequence_packets <= 0 or config.batch_size <= 0:
        raise PipelineError("sequence packets and batch size must be positive")
    if config.max_epochs <= 0 or config.patience <= 0:
        raise PipelineError("maximum epochs and patience must be positive")
    if config.hidden_size <= 0 or config.max_sequences_per_device <= 0:
        raise PipelineError("hidden size and sequence cap must be positive")
    if not 0 <= config.dropout < 1:
        raise PipelineError("dropout must be in [0, 1)")
    if args.test_trials <= 0 or args.jobs <= 0:
        raise PipelineError("test trials and jobs must be positive")
    scan_batch_rows = int(getattr(args, "scan_batch_rows", DEFAULT_SCAN_BATCH_ROWS))
    if scan_batch_rows <= 0:
        raise PipelineError("scan batch rows must be positive")
    stop_file = args.stop_file.resolve() if args.stop_file else None
    removal_specs = common.parse_removal_specs(args.remove_devices, train_datasets)
    experiment_dir = args.output_root.resolve() / common.experiment_directory_name(
        train_datasets, test_datasets, features_root, removal_specs, config.seed
    )
    if common.stop_requested(stop_file):
        print(f"Stop requested before startup; output was not changed: {stop_file}")
        return experiment_dir
    device = resolve_torch_device(args.device)
    if args.jobs > 1 and device.type != "cpu":
        raise PipelineError("--jobs greater than 1 is supported only on CPU")
    provenance = validate_packet_feature_manifest(
        features_root,
        labels_root,
        label_file,
        [*train_datasets, *test_datasets],
    )
    print(f"selected torch device: {device} ({torch_device_description(device)})")
    print(f"streaming training catalog: {', '.join(train_datasets)}", flush=True)
    train = load_packet_catalog(
        train_datasets,
        features_root,
        labels_root,
        label_file,
        scan_batch_rows=scan_batch_rows,
        stop_file=stop_file,
    )
    common.check_stop(stop_file)
    print(f"streaming test catalog: {', '.join(test_datasets)}", flush=True)
    test = load_packet_catalog(
        test_datasets,
        features_root,
        labels_root,
        label_file,
        scan_batch_rows=scan_batch_rows,
        stop_file=stop_file,
    )
    print(
        f"catalog complete: training_rows={train.total_rows:,}; "
        f"test_rows={test.total_rows:,}; peak_rss={peak_rss_bytes() / (1024**3):.2f} GiB",
        flush=True,
    )
    classes = common.validate_class_sets(train.device_frame, test.device_frame)
    removal_plan = common.generate_removal_plan(
        train.device_frame, removal_specs, args.removal_runs, config.seed
    )
    configuration = {
        "pipeline": "packet_sequence_lstm_mac_v2",
        "feature_provenance": provenance,
        "train_datasets": train_datasets,
        "test_datasets": test_datasets,
        "label_file": label_file,
        "classes": classes,
        "removal_plan": removal_plan,
        "training": asdict(config),
        "test_trials": args.test_trials,
        "requested_device": args.device,
        "resolved_device": str(device),
    }
    fingerprint = common.stable_fingerprint(configuration)
    previous = common.prepare_resumable_output_directory(
        experiment_dir,
        resume=args.resume,
        overwrite=args.overwrite,
        fingerprint=fingerprint,
    )
    supplied_cache_root = getattr(args, "cache_root", None)
    cache_root = (
        Path(supplied_cache_root).resolve()
        if supplied_cache_root is not None
        else experiment_dir / ".sequence-cache"
    )
    cache_namespace = fingerprint[:16]
    cache_estimate = estimated_peak_cache_bytes(
        train.captures, test.captures, config, args.test_trials, args.jobs
    )
    _check_cache_space(cache_root, cache_estimate)
    manifest: dict[str, Any] = {
        "status": "running",
        "pipeline": "packet_sequence_lstm_v2",
        "started_at": (previous or {}).get("started_at", common.utc_now()),
        "updated_at": common.utc_now(),
        "train_datasets": train_datasets,
        "test_datasets": test_datasets,
        "feature_provenance": provenance,
        "model_runs": len(removal_plan),
        "completed_model_runs": 0,
        "test_trials": args.test_trials,
        "sequence_packets": config.sequence_packets,
        "model_type": config.model_type,
        "classes": classes,
        "label_file": label_file,
        "configuration_fingerprint": fingerprint,
        "data_access": {
            "version": DATA_ACCESS_VERSION,
            "mode": "streamed_catalog_disk_sequence_cache",
            "scan_batch_rows": scan_batch_rows,
            "training_packet_rows": train.total_rows,
            "test_packet_rows": test.total_rows,
            "estimated_peak_cache_bytes": cache_estimate,
            "catalog_peak_rss_bytes": peak_rss_bytes(),
        },
    }
    common.atomic_write_json(experiment_dir / "_run_manifest.json", manifest)
    common.atomic_write_json(experiment_dir / "removal_plan.json", removal_plan)
    common.atomic_write_json(
        experiment_dir / "config.json",
        {
            **configuration,
            "features_root": str(features_root),
            "labels_root": str(labels_root),
            "output_root": str(args.output_root.resolve()),
            "jobs": args.jobs,
            "stop_file": str(stop_file) if stop_file else None,
            "cache_root": str(cache_root),
            "scan_batch_rows": scan_batch_rows,
        },
    )
    results: dict[int, dict[str, Any]] = {}
    pending = []
    for entry in removal_plan:
        run_number = int(entry["run"])
        run_dir = experiment_dir / f"model_run_{run_number:03d}"
        loaded = None
        if args.resume and common.completed_model_run(run_dir):
            try:
                loaded = _load_completed_result(run_dir)
            except (OSError, ValueError, KeyError, PipelineError):
                loaded = None
        if loaded is not None:
            results[run_number] = loaded
        else:
            if run_dir.exists():
                shutil.rmtree(run_dir)
            pending.append(entry)
    manifest["completed_model_runs"] = len(results)
    common.atomic_write_json(experiment_dir / "_run_manifest.json", manifest)

    worker_args = (
        experiment_dir,
        train.captures,
        test.captures,
        test.unevaluable_devices,
        train_datasets,
        test_datasets,
        features_root,
        labels_root,
        label_file,
        classes,
        config,
        str(device),
        args.test_trials,
        str(stop_file) if stop_file else None,
        str(cache_root),
        cache_namespace,
        scan_batch_rows,
    )

    def completed(result: dict[str, Any]) -> None:
        results[int(result["run"])] = result
        manifest["completed_model_runs"] = len(results)
        manifest["updated_at"] = common.utc_now()
        common.atomic_write_json(experiment_dir / "_run_manifest.json", manifest)

    try:
        stopped = common.run_bounded_model_jobs(
            pending,
            _execute_model_run,
            worker_args,
            args.jobs,
            stop_file,
            completed,
        )
        if stopped:
            manifest.update(
                {
                    "status": "stopped",
                    "stopped_at": common.utc_now(),
                    "updated_at": common.utc_now(),
                }
            )
            common.atomic_write_json(experiment_dir / "_run_manifest.json", manifest)
            print("Stopped gracefully; remove the stop file and rerun with --resume.")
            return experiment_dir
        ordered = [results[index] for index in sorted(results)]
        if len(ordered) != len(removal_plan):
            raise PipelineError("not all LSTM model runs completed")
        metric_rows = [row for result in ordered for row in result["metric_rows"]]
        common.atomic_write_parquet(
            pd.DataFrame(metric_rows), experiment_dir / "all_trial_metrics.parquet"
        )
        model_means = [
            {
                key: value["mean"] if value is not None else None
                for key, value in result["summary"]["combined"].items()
            }
            for result in ordered
        ]
        aggregate = {
            "model_runs": [
                {"run": result["run"], "summary": result["summary"]}
                for result in ordered
            ],
            "across_model_run_means": common.summarize_metric_dicts(model_means),
            "by_dataset_across_model_run_means": {
                dataset: common.summarize_metric_dicts(
                    [
                        {
                            key: value["mean"] if value is not None else None
                            for key, value in result["summary"]["by_dataset"][
                                dataset
                            ].items()
                        }
                        for result in ordered
                    ]
                )
                for dataset in test_datasets
            },
        }
        common.atomic_write_json(experiment_dir / "aggregate_metrics.json", aggregate)
        mean_accuracy = aggregate["across_model_run_means"]["accuracy"]["mean"]
        now = common.utc_now()
        manifest.update(
            {
                "status": "complete",
                "completed_model_runs": len(ordered),
                "mean_accuracy_across_model_run_means": mean_accuracy,
                "completed_at": now,
                "updated_at": now,
            }
        )
        common.atomic_write_json(experiment_dir / "_run_manifest.json", manifest)
        print(
            f"complete: mean_model_accuracy={mean_accuracy:.4f}; "
            f"output={experiment_dir}",
            flush=True,
        )
        return experiment_dir
    except common.StopRequested:
        manifest.update(
            {"status": "stopped", "stopped_at": common.utc_now(), "updated_at": common.utc_now()}
        )
        common.atomic_write_json(experiment_dir / "_run_manifest.json", manifest)
        return experiment_dir
    except BaseException as exc:
        manifest.update(
            {"status": "failed", "error": str(exc), "updated_at": common.utc_now()}
        )
        common.atomic_write_json(experiment_dir / "_run_manifest.json", manifest)
        raise
    finally:
        shutil.rmtree(cache_root / cache_namespace, ignore_errors=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-datasets", nargs="+", required=True, choices=common.DATASETS)
    parser.add_argument("--test-datasets", nargs="+", required=True, choices=common.DATASETS)
    parser.add_argument(
        "--features-root",
        type=Path,
        default=Path("built_features_mac_v2/binary-class/packet_features"),
    )
    parser.add_argument("--labels-root", type=Path, default=Path("labels"))
    parser.add_argument("--label-file", default=common.DEFAULT_MAC_LABEL_FILE)
    parser.add_argument("--output-root", type=Path, default=Path("lstm_runs_mac_v2"))
    parser.add_argument("--model-type", choices=("lstm", "blstm"), default="blstm")
    parser.add_argument("--sequence-packets", type=int, default=100)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--max-sequences-per-device", type=int, default=5_000)
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument(
        "--scan-batch-rows",
        type=int,
        default=DEFAULT_SCAN_BATCH_ROWS,
        help="maximum packet rows retained during streamed validation",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        help="temporary disk-backed sequence cache (defaults inside the experiment)",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps", "xpu"), default="auto"
    )
    parser.add_argument("--test-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--remove-devices",
        action="append",
        nargs=3,
        metavar=("DATASET", "CLASS", "COUNT"),
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
    except (PipelineError, OSError, ValueError, pa.ArrowException) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
