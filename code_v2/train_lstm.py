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
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

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


def stream_candidate_count(stream: PacketStream, sequence_packets: int) -> int:
    size = len(stream.features)
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
    start = int(ordinal - previous) if len(stream.features) >= sequence_packets else 0
    return SequenceCandidate(
        stream_index=stream_index,
        start=start,
        valid_length=min(sequence_packets, len(stream.features) - start),
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


def fit_packet_scaler(streams: Sequence[PacketStream]) -> StandardScaler:
    scaler = StandardScaler()
    for stream in streams:
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


def stream_counts(streams: Sequence[PacketStream]) -> dict[str, Any]:
    devices = {(stream.dataset, stream.device_id, stream.label) for stream in streams}
    return {
        "streams": len(streams),
        "packets": int(sum(len(stream.features) for stream in streams)),
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
    train_streams: Sequence[PacketStream],
    test_streams: Sequence[PacketStream],
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
) -> dict[str, Any]:
    run_number = int(run_entry["run"])
    run_dir, final_run_dir = common.create_model_run_staging_directory(
        experiment_dir, run_number
    )
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
        components = train_components(reduced, classes, config, device, stop_file)
        predictions = sample_unseen_sequences(
            test_streams,
            components.scaler,
            components.model,
            classes,
            config,
            device,
            test_trials,
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
        reloaded_predictions = sample_unseen_sequences(
            test_streams,
            reloaded_scaler,
            reloaded_model,
            reloaded_classes,
            reloaded_config,
            device,
            test_trials,
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
    print(f"loading training datasets: {', '.join(train_datasets)}", flush=True)
    train = load_packet_datasets(
        train_datasets, features_root, labels_root, label_file
    )
    common.check_stop(stop_file)
    print(f"loading test datasets: {', '.join(test_datasets)}", flush=True)
    test = load_packet_datasets(test_datasets, features_root, labels_root, label_file)
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
        train.streams,
        test.streams,
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
