#!/usr/bin/env python3
"""Train and evaluate a leakage-free cross-dataset CNN-PN-RF classifier."""

from __future__ import annotations

import argparse
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
import sklearn
import torch
import experiment_common as common
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Keep PyTorch from competing with native scikit-learn/Accelerate thread pools on
# macOS. The network is too small to benefit from intra-op fan-out.
torch.set_num_threads(1)


DATASETS = common.DATASETS
FEATURE_COLUMNS = (
    "outgoing_packet_count",
    "outgoing_total_bytes",
    "incoming_packet_count",
    "incoming_total_bytes",
    "aggregated_packet_count",
    "aggregated_total_bytes",
    "idle_time",
)
IDENTITY_COLUMNS = ("dataset", "device_id", "capture_id", "window")
PREDICTION_BASE_COLUMNS = (
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


def prediction_columns(classes: Sequence[int]) -> list[str]:
    return [
        *PREDICTION_BASE_COLUMNS,
        *probability_columns(classes),
        *(f"distance_class_{label}" for label in classes),
    ]


def device_prediction_columns(classes: Sequence[int]) -> list[str]:
    return [
        *DEVICE_PREDICTION_BASE_COLUMNS[:4],
        *(f"votes_class_{label}" for label in classes),
        *(f"mean_probability_class_{label}" for label in classes),
        *DEVICE_PREDICTION_BASE_COLUMNS[4:],
    ]


PipelineError = common.PipelineError


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    validation_fraction: float = 0.2
    batch_size: int = 1024
    max_epochs: int = 100
    patience: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    rf_trees: int = 100
    loader_workers: int = 0
    rf_jobs: int = -1


@dataclass
class LoadedData:
    frame: pd.DataFrame
    unevaluable_devices: list[dict[str, Any]]


@dataclass
class TrainedComponents:
    scaler: StandardScaler
    cnn: "CompactCNN"
    prototypes: np.ndarray
    random_forest: RandomForestClassifier
    best_epoch: int
    validation_history: list[dict[str, Any]]
    validation_devices: list[dict[str, Any]]
    class_weights: list[float]
    classes: list[int]


def resume_fingerprint_configuration(
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    """Exclude Random Forest worker count without relaxing model parameters."""
    normalized = dict(configuration)
    training = dict(normalized["training"])
    training.pop("rf_jobs", None)
    normalized["training"] = training
    return normalized


def migrate_legacy_resource_fingerprint(
    experiment_dir: Path,
    resume_configuration: Mapping[str, Any],
    fingerprint: str,
    *,
    resume: bool,
) -> None:
    """Permit resume when only the legacy RF worker allocation changed."""
    if not resume or not experiment_dir.is_dir():
        return
    manifest_path = experiment_dir / "_run_manifest.json"
    config_path = experiment_dir / "config.json"
    manifest = common.read_json(manifest_path)
    if manifest.get("configuration_fingerprint") == fingerprint:
        return
    previous_config = common.read_json(config_path)
    try:
        comparable = {
            key: previous_config[key] for key in resume_configuration
        }
    except KeyError:
        return
    previous_fingerprint = common.stable_fingerprint(
        resume_fingerprint_configuration(comparable)
    )
    if previous_fingerprint != fingerprint:
        return
    manifest["configuration_fingerprint"] = fingerprint
    manifest["updated_at"] = common.utc_now()
    atomic_write_json(manifest_path, manifest)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    common.atomic_write_json(path, value)


def validate_dataset_sets(
    train_datasets: Sequence[str], test_datasets: Sequence[str]
) -> tuple[list[str], list[str]]:
    return common.validate_dataset_sets(train_datasets, test_datasets)


def load_label_map(
    labels_root: Path,
    dataset: str,
    label_file: str = common.DEFAULT_MAC_LABEL_FILE,
) -> dict[str, int]:
    return common.load_mac_label_map(labels_root, dataset, label_file)


def _validate_feature_table(
    table: pa.Table, dataset: str, device_id: str, authoritative_label: int, path: Path
) -> None:
    expected = {"capture_id", "window", "device_id", "is_iot_device", *FEATURE_COLUMNS}
    missing = expected - set(table.column_names)
    if missing:
        raise PipelineError(f"{path}: missing columns {', '.join(sorted(missing))}")
    if not table.num_rows:
        return
    parquet_ids = set(table.column("device_id").to_pylist())
    if parquet_ids != {device_id}:
        raise PipelineError(
            f"{path}: device_id values {sorted(parquet_ids)} do not match folder {device_id}"
        )
    parquet_labels = set(table.column("is_iot_device").to_pylist())
    if parquet_labels != {authoritative_label}:
        raise PipelineError(
            f"{path}: repeated labels {sorted(parquet_labels)} disagree with label JSON "
            f"value {authoritative_label}"
        )


def load_datasets(
    datasets: Sequence[str],
    features_root: Path,
    labels_root: Path,
    label_file: str = common.DEFAULT_MAC_LABEL_FILE,
) -> LoadedData:
    loaded = common.load_mac_datasets(datasets, features_root, labels_root, label_file)
    return LoadedData(loaded.frame, loaded.unevaluable_devices)


def raw_feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    return common.raw_feature_matrix(frame)


def log_features(values: np.ndarray) -> np.ndarray:
    return np.log1p(values)


def stratified_device_validation_split(
    frame: pd.DataFrame, validation_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Split by device, falling back to capture dates for singleton strata."""
    if not 0 < validation_fraction < 1:
        raise PipelineError("validation fraction must be between zero and one")
    devices = frame[["dataset", "device_id", "label"]].drop_duplicates()
    inconsistent = devices.groupby(["dataset", "device_id"])["label"].nunique()
    if (inconsistent > 1).any():
        raise PipelineError("a device has multiple class labels")
    rng = np.random.default_rng(seed)
    validation_device_keys: set[tuple[str, str]] = set()
    validation_capture_keys: set[tuple[str, str, str]] = set()
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
                    validation_device_keys.add((dataset, device_id))
            continue

        device_id = identifiers[0]
        capture_ids = sorted(
            frame.loc[
                (frame["dataset"] == dataset)
                & (frame["device_id"] == device_id)
                & (frame["label"] == label),
                "capture_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )
        if len(capture_ids) < 2:
            raise PipelineError(
                "cannot create capture-date-held-out validation for "
                f"{dataset}, class {label}, device {device_id}: only "
                f"{len(capture_ids)} capture date(s) have rows"
            )
        shuffled = np.asarray(capture_ids, dtype=object)
        rng.shuffle(shuffled)
        count = max(1, int(round(len(shuffled) * validation_fraction)))
        count = min(count, len(shuffled) - 1)
        selected = set(shuffled[:count].tolist())
        for capture_id in capture_ids:
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
                validation_capture_keys.add((dataset, device_id, capture_id))

    row_keys = zip(
        frame["dataset"], frame["device_id"], frame["capture_id"].astype(str)
    )
    validation_mask = np.fromiter(
        (
            (dataset, device_id) in validation_device_keys
            or (dataset, device_id, capture_id) in validation_capture_keys
            for dataset, device_id, capture_id in row_keys
        ),
        dtype=bool,
        count=len(frame),
    )
    fit_mask = ~validation_mask
    classes = set(frame["label"].unique())
    if set(frame.loc[fit_mask, "label"].unique()) != classes:
        raise PipelineError("CNN fit split does not contain every training class")
    if set(frame.loc[validation_mask, "label"].unique()) != classes:
        raise PipelineError(
            "CNN validation split does not contain every training class"
        )
    return fit_mask, validation_mask, assignments


def calculate_class_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels.astype(np.int64))
    if (counts == 0).any():
        raise PipelineError("all contiguous classes must be present in training data")
    return (len(labels) / (len(counts) * counts)).astype(np.float32)


def set_reproducible_seed(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # The small network does not benefit from CPU intra-op fan-out, and using a
    # single thread avoids native OpenMP/Accelerate instability on macOS while
    # also making repeated fixed-seed runs reproducible.
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

    def smoke_test(name: str) -> torch.device:
        device = torch.device(name)
        try:
            probe = torch.ones(1, device=device)
            _ = (probe + 1).cpu()
        except Exception as exc:
            raise PipelineError(
                f"{name.upper()} is reported available but failed an allocation test: {exc}"
            ) from exc
        return device

    if requested == "auto":
        failures = []
        for candidate in ("cuda", "mps", "xpu"):
            if not available(candidate):
                continue
            try:
                return smoke_test(candidate)
            except PipelineError as exc:
                failures.append(str(exc))
        if failures:
            print(
                "accelerator probe failed; falling back to CPU: " + "; ".join(failures),
                flush=True,
            )
        return smoke_test("cpu")
    if not available(requested):
        raise PipelineError(f"{requested.upper()} was requested but is unavailable")
    return smoke_test(requested)


def torch_device_description(device: torch.device) -> str:
    if device.type == "cuda":
        return str(torch.cuda.get_device_name(device))
    if device.type == "xpu":
        backend = getattr(torch, "xpu", None)
        getter = getattr(backend, "get_device_name", None)
        return str(getter(device)) if getter is not None else "Intel XPU"
    if device.type == "mps":
        return "Apple Metal Performance Shaders"
    return platform.processor() or platform.machine() or "CPU"


class CompactCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        if num_classes < 2:
            raise PipelineError("CNN requires at least two classes")
        self.num_classes = num_classes
        self.convolution = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.embedding_layer = nn.Linear(32 * 3, 32)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(32, num_classes)

    def forward(
        self, values: torch.Tensor, return_embedding: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        values = values.unsqueeze(1)
        values = self.pool(torch.relu(self.batch_norm(self.convolution(values))))
        values = torch.flatten(values, start_dim=1)
        embedding = torch.relu(self.embedding_layer(values))
        logits = self.classifier(self.dropout(embedding))
        if return_embedding:
            return logits, embedding
        return logits


def _loader(
    features: np.ndarray,
    labels: np.ndarray | None,
    batch_size: int,
    shuffle: bool,
    seed: int,
    workers: int,
) -> DataLoader:
    feature_tensor = torch.from_numpy(np.asarray(features, dtype=np.float32))
    if labels is None:
        dataset = TensorDataset(feature_tensor)
    else:
        dataset = TensorDataset(
            feature_tensor, torch.from_numpy(np.asarray(labels, dtype=np.int64))
        )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        generator=generator,
        persistent_workers=workers > 0,
    )


def _run_training_epoch(
    model: CompactCNN,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    stop_file: Path | str | None = None,
) -> tuple[float, float]:
    model.train()
    loss_total = 0.0
    correct = 0
    examples = 0
    for values, labels in loader:
        common.check_stop(stop_file)
        values, labels = values.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(values)
        assert isinstance(logits, torch.Tensor)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = labels.shape[0]
        loss_total += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        examples += batch_size
    return loss_total / examples, correct / examples


@torch.no_grad()
def _evaluate_cnn(
    model: CompactCNN,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    stop_file: Path | str | None = None,
) -> tuple[float, float]:
    model.eval()
    loss_total = 0.0
    correct = 0
    examples = 0
    for values, labels in loader:
        common.check_stop(stop_file)
        values, labels = values.to(device), labels.to(device)
        logits = model(values)
        assert isinstance(logits, torch.Tensor)
        loss = criterion(logits, labels)
        batch_size = labels.shape[0]
        loss_total += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        examples += batch_size
    return loss_total / examples, correct / examples


def select_best_epoch(
    fit_features: np.ndarray,
    fit_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    config: TrainingConfig,
    device: torch.device,
    num_classes: int,
    stop_file: Path | str | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    set_reproducible_seed(config.seed)
    model = CompactCNN(num_classes).to(device)
    weights = calculate_class_weights(fit_labels)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    fit_loader = _loader(
        fit_features,
        fit_labels,
        config.batch_size,
        True,
        config.seed,
        config.loader_workers,
    )
    validation_loader = _loader(
        validation_features,
        validation_labels,
        config.batch_size,
        False,
        config.seed,
        config.loader_workers,
    )
    history = []
    best_loss = math.inf
    best_epoch = 1
    epochs_without_improvement = 0
    for epoch in range(1, config.max_epochs + 1):
        common.check_stop(stop_file)
        train_loss, train_accuracy = _run_training_epoch(
            model, fit_loader, criterion, optimizer, device, stop_file
        )
        validation_loss, validation_accuracy = _evaluate_cnn(
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
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break
    return best_epoch, history


def train_final_cnn(
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    config: TrainingConfig,
    device: torch.device,
    num_classes: int,
    stop_file: Path | str | None = None,
) -> tuple[CompactCNN, list[float], list[dict[str, Any]]]:
    set_reproducible_seed(config.seed)
    model = CompactCNN(num_classes).to(device)
    weights = calculate_class_weights(labels)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loader = _loader(
        features,
        labels,
        config.batch_size,
        True,
        config.seed,
        config.loader_workers,
    )
    history = []
    for epoch in range(1, epochs + 1):
        common.check_stop(stop_file)
        loss, accuracy = _run_training_epoch(
            model, loader, criterion, optimizer, device, stop_file
        )
        history.append({"epoch": epoch, "loss": loss, "accuracy": accuracy})
        print(
            f"final epoch {epoch:03d}/{epochs:03d}: loss={loss:.6f} "
            f"accuracy={accuracy:.4f}",
            flush=True,
        )
    return model, weights.tolist(), history


@torch.no_grad()
def extract_embeddings(
    model: CompactCNN,
    features: np.ndarray,
    batch_size: int,
    device: torch.device,
    workers: int = 0,
    stop_file: Path | str | None = None,
) -> np.ndarray:
    model.eval()
    loader = _loader(features, None, batch_size, False, 0, workers)
    chunks = []
    for (values,) in loader:
        common.check_stop(stop_file)
        output = model(values.to(device), return_embedding=True)
        assert isinstance(output, tuple)
        chunks.append(output[1].cpu().numpy())
    return np.concatenate(chunks).astype(np.float32, copy=False)


def compute_prototypes(
    embeddings: np.ndarray, labels: np.ndarray, classes: Sequence[int]
) -> np.ndarray:
    if embeddings.ndim != 2 or len(embeddings) != len(labels):
        raise PipelineError("embedding and label shapes are incompatible")
    prototypes = []
    for label in classes:
        selected = embeddings[labels == label]
        if not len(selected):
            raise PipelineError(f"cannot compute prototype for absent class {label}")
        prototypes.append(selected.mean(axis=0))
    return np.stack(prototypes).astype(np.float32)


def prototype_distances(embeddings: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    if (
        embeddings.ndim != 2
        or prototypes.ndim != 2
        or prototypes.shape[1] != embeddings.shape[1]
    ):
        raise PipelineError("prototype dimensions do not match embeddings")
    differences = embeddings[:, None, :] - prototypes[None, :, :]
    return np.linalg.norm(differences, axis=2).astype(np.float32)


def fit_random_forest(
    training_distances: np.ndarray,
    training_labels: np.ndarray,
    trees: int,
    seed: int,
    stop_file: Path | str | None = None,
    jobs: int = -1,
) -> RandomForestClassifier:
    if len(training_distances) != len(training_labels):
        raise PipelineError(
            "Random Forest training distances and labels differ in length"
        )
    classes = sorted(int(label) for label in np.unique(training_labels))
    if training_distances.ndim != 2 or training_distances.shape[1] != len(classes):
        raise PipelineError(
            "Random Forest requires one prototype-distance feature per class"
        )
    common.check_stop(stop_file)
    classifier = RandomForestClassifier(
        n_estimators=trees,
        class_weight="balanced",
        random_state=seed,
        n_jobs=jobs,
    )
    classifier.fit(training_distances, training_labels)
    common.check_stop(stop_file)
    return classifier


def train_components(
    train_frame: pd.DataFrame,
    config: TrainingConfig,
    device: torch.device,
    classes: Sequence[int],
    stop_file: Path | str | None = None,
) -> tuple[TrainedComponents, list[dict[str, Any]]]:
    common.check_stop(stop_file)
    raw = raw_feature_matrix(train_frame)
    labels = train_frame["label"].to_numpy(dtype=np.int64)
    fit_mask, validation_mask, assignments = stratified_device_validation_split(
        train_frame, config.validation_fraction, config.seed
    )

    selection_scaler = StandardScaler().fit(log_features(raw[fit_mask]))
    fit_features = selection_scaler.transform(log_features(raw[fit_mask])).astype(
        np.float32
    )
    validation_features = selection_scaler.transform(
        log_features(raw[validation_mask])
    ).astype(np.float32)
    best_epoch, selection_history = select_best_epoch(
        fit_features,
        labels[fit_mask],
        validation_features,
        labels[validation_mask],
        config,
        device,
        len(classes),
        stop_file,
    )
    print(
        f"selected epoch: {best_epoch}; refitting CNN on all training rows", flush=True
    )

    final_scaler = StandardScaler().fit(log_features(raw))
    full_features = final_scaler.transform(log_features(raw)).astype(np.float32)
    final_cnn, class_weights, final_history = train_final_cnn(
        full_features, labels, best_epoch, config, device, len(classes), stop_file
    )
    train_embeddings = extract_embeddings(
        final_cnn,
        full_features,
        config.batch_size,
        device,
        config.loader_workers,
        stop_file,
    )
    prototypes = compute_prototypes(train_embeddings, labels, classes)
    train_distances = prototype_distances(train_embeddings, prototypes)
    # Only training distances and labels enter this fit; test data is unavailable here.
    random_forest = fit_random_forest(
        train_distances,
        labels,
        config.rf_trees,
        config.seed,
        stop_file,
        config.rf_jobs,
    )
    components = TrainedComponents(
        scaler=final_scaler,
        cnn=final_cnn,
        prototypes=prototypes,
        random_forest=random_forest,
        best_epoch=best_epoch,
        validation_history=selection_history,
        validation_devices=assignments,
        class_weights=class_weights,
        classes=list(classes),
    )
    return components, final_history


def predict_samples(
    frame: pd.DataFrame,
    components: TrainedComponents,
    config: TrainingConfig,
    device: torch.device,
    stop_file: Path | str | None = None,
) -> pd.DataFrame:
    common.check_stop(stop_file)
    raw = raw_feature_matrix(frame)
    transformed = components.scaler.transform(log_features(raw)).astype(np.float32)
    embeddings = extract_embeddings(
        components.cnn,
        transformed,
        config.batch_size,
        device,
        config.loader_workers,
        stop_file,
    )
    distances = prototype_distances(embeddings, components.prototypes)
    predictions = components.random_forest.predict(distances).astype(np.int32)
    class_indices = [int(label) for label in components.random_forest.classes_]
    if class_indices != components.classes:
        raise PipelineError(f"unexpected Random Forest classes: {class_indices}")
    probabilities = components.random_forest.predict_proba(distances)
    output = frame.loc[:, IDENTITY_COLUMNS].copy()
    output["true_label"] = frame["label"].to_numpy(dtype=np.int32)
    output["predicted_label"] = predictions
    for index, label in enumerate(components.classes):
        output[f"probability_class_{label}"] = probabilities[:, index].astype(
            np.float64
        )
        output[f"distance_class_{label}"] = distances[:, index].astype(np.float64)
    return output.loc[:, prediction_columns(components.classes)]


def aggregate_device_predictions(
    sample_predictions: pd.DataFrame, classes: Sequence[int]
) -> pd.DataFrame:
    rows = []
    for (dataset, device_id), group in sample_predictions.groupby(
        ["dataset", "device_id"], sort=True
    ):
        labels = group["true_label"].unique()
        if len(labels) != 1:
            raise PipelineError(f"{dataset}/{device_id}: inconsistent true labels")
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
            "n_samples": len(group),
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


def compute_metrics(
    true_labels: Sequence[int],
    predictions: Sequence[int],
    probabilities: Sequence[Sequence[float]] | Sequence[float] | np.ndarray,
    classes: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper around the shared multi-class metric implementation."""
    y_true = np.asarray(true_labels, dtype=np.int64)
    y_pred = np.asarray(predictions, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if classes is None:
        observed = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
        classes = list(range(max(observed, default=1) + 1))
    if scores.ndim == 1:
        if list(classes) != [0, 1]:
            raise PipelineError("one-dimensional probabilities require classes [0, 1]")
        scores = np.column_stack([1.0 - scores, scores])
    return common.compute_metrics(y_true, y_pred, scores, classes)


def metrics_report(
    sample_predictions: pd.DataFrame,
    device_predictions: pd.DataFrame,
    test_datasets: Sequence[str],
    unevaluable_devices: Sequence[Mapping[str, Any]],
    classes: Sequence[int],
) -> dict[str, Any]:
    return {
        "classes": list(classes),
        "sample_level": common.evaluate_predictions(
            sample_predictions, test_datasets, classes
        ),
        "device_level": common.evaluate_predictions(
            device_predictions.rename(
                columns={
                    f"mean_probability_class_{label}": f"probability_class_{label}"
                    for label in classes
                }
            ),
            test_datasets,
            classes,
        ),
        "unevaluable_devices": list(unevaluable_devices),
    }


def run_directory_name(
    train_datasets: Sequence[str], test_datasets: Sequence[str], seed: int
) -> str:
    return (
        f"train-{'+'.join(train_datasets)}__test-{'+'.join(test_datasets)}__seed-{seed}"
    )


def prepare_run_directory(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise PipelineError(f"run output already exists: {path}; pass --overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def save_components(
    run_dir: Path,
    components: TrainedComponents,
    config_payload: Mapping[str, Any],
    final_history: Sequence[Mapping[str, Any]],
) -> None:
    torch.save(components.cnn.state_dict(), run_dir / "cnn_state.pt")
    joblib.dump(components.scaler, run_dir / "scaler.joblib")
    np.save(run_dir / "prototypes.npy", components.prototypes)
    joblib.dump(components.random_forest, run_dir / "random_forest.joblib")
    atomic_write_json(run_dir / "label_mapping.json", {"classes": components.classes})
    atomic_write_json(run_dir / "config.json", config_payload)
    pd.DataFrame(components.validation_history).to_csv(
        run_dir / "validation_history.csv", index=False
    )
    pd.DataFrame(final_history).to_csv(
        run_dir / "final_training_history.csv", index=False
    )
    atomic_write_json(
        run_dir / "validation_devices.json",
        {"assignments": components.validation_devices},
    )


def load_components(run_dir: Path, device: torch.device) -> TrainedComponents:
    scaler = joblib.load(run_dir / "scaler.joblib")
    prototypes = np.load(run_dir / "prototypes.npy")
    random_forest = joblib.load(run_dir / "random_forest.joblib")
    config_payload = json.loads((run_dir / "config.json").read_text())
    classes = [int(label) for label in config_payload["classes"]]
    cnn = CompactCNN(len(classes)).to(device)
    state = torch.load(run_dir / "cnn_state.pt", map_location=device, weights_only=True)
    cnn.load_state_dict(state)
    cnn.eval()
    return TrainedComponents(
        scaler=scaler,
        cnn=cnn,
        prototypes=prototypes,
        random_forest=random_forest,
        best_epoch=int(config_payload["best_epoch"]),
        validation_history=[],
        validation_devices=[],
        class_weights=list(config_payload["class_weights"]),
        classes=classes,
    )


def verify_artifact_reload(
    run_dir: Path,
    test_frame: pd.DataFrame,
    original_predictions: pd.DataFrame,
    config: TrainingConfig,
    device: torch.device,
) -> None:
    reloaded = load_components(run_dir, device)
    reloaded_predictions = predict_samples(
        test_frame.reset_index(drop=True), reloaded, config, device
    )
    expected = original_predictions.reset_index(drop=True)
    if not np.array_equal(
        reloaded_predictions["predicted_label"].to_numpy(),
        expected["predicted_label"].to_numpy(),
    ) or not np.allclose(
        reloaded_predictions.loc[:, probability_columns(reloaded.classes)].to_numpy(),
        expected.loc[:, probability_columns(reloaded.classes)].to_numpy(),
        rtol=0,
        atol=1e-12,
    ):
        raise PipelineError("saved artifacts do not reproduce predictions after reload")


def sample_test_windows(
    test_frame: pd.DataFrame, test_trials: int, seed: int
) -> pd.DataFrame:
    """Choose one row per device/trial using the requested hybrid replacement rule."""
    if test_trials <= 0:
        raise PipelineError("--test-trials must be positive")
    frame = test_frame.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for (dataset, device_id), indices in frame.groupby(
        ["dataset", "device_id"], sort=True
    ).indices.items():
        candidates = np.asarray(indices, dtype=np.int64)
        replacement_required = len(candidates) < test_trials
        chosen = rng.choice(candidates, size=test_trials, replace=replacement_required)
        for trial, source_row in enumerate(chosen.tolist(), start=1):
            rows.append(
                {
                    "trial": trial,
                    "source_row": source_row,
                    "dataset": dataset,
                    "device_id": device_id,
                    "replacement_required": replacement_required,
                }
            )
    return pd.DataFrame(rows)


def _trial_report(
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
        "evaluation_unit": "one randomly selected window per device, classified by RF",
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


def _execute_cnn_model_run(
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
    config: TrainingConfig,
    device_name: str,
    test_trials: int,
    stop_file: str | None,
) -> dict[str, Any]:
    run_number = int(run_entry["run"])
    run_dir, final_run_dir = common.create_model_run_staging_directory(
        experiment_dir, run_number
    )
    manifest: dict[str, Any] = {"status": "running", "run": run_number}
    atomic_write_json(run_dir / "_run_manifest.json", manifest)
    try:
        common.check_stop(stop_file)
        device = resolve_torch_device(device_name)
        reduced_train = common.apply_removal(train_frame, run_entry["removed_devices"])
        print(
            f"model run {run_number}: training rows={len(reduced_train):,}; "
            f"torch_device={device}",
            flush=True,
        )
        components, final_history = train_components(
            reduced_train, config, device, classes, stop_file
        )

        common.check_stop(stop_file)
        selection = sample_test_windows(test_frame, test_trials, config.seed + 10_000)
        unique_rows = np.sort(selection["source_row"].unique())
        selected_frame = (
            test_frame.reset_index(drop=True).iloc[unique_rows].reset_index(drop=True)
        )
        selected_predictions = predict_samples(
            selected_frame, components, config, device, stop_file
        )
        selected_predictions.insert(0, "source_row", unique_rows)
        predictions = selection.drop(columns=["dataset", "device_id"]).merge(
            selected_predictions, on="source_row", how="left", validate="many_to_one"
        )
        predictions.insert(0, "model_run", run_number)
        predictions = predictions.sort_values(
            ["trial", "dataset", "device_id"]
        ).reset_index(drop=True)
        report = _trial_report(predictions, test_datasets, unevaluable_devices, classes)

        config_payload = {
            "training": asdict(config),
            "train_datasets": list(train_datasets),
            "test_datasets": list(test_datasets),
            "features_root": str(features_root),
            "labels_root": str(labels_root),
            "label_file": label_file,
            "feature_columns": list(FEATURE_COLUMNS),
            "classes": list(classes),
            "removed_devices": run_entry["removed_devices"],
            "training_counts_after_removal": common.removal_counts(reduced_train),
            "test_trials": test_trials,
            "test_sampling_seed": config.seed + 10_000,
            "sampling_rule": (
                "without replacement per device when windows >= trials; "
                "with replacement otherwise"
            ),
            "validation_split_policy": (
                "device holdout per dataset/class with seeded capture-date holdout "
                "for singleton strata"
            ),
            "architecture": {
                "conv_filters": 32,
                "kernel_size": 3,
                "padding": 1,
                "max_pool_size": 2,
                "embedding_dimensions": 32,
                "dropout": 0.3,
            },
            "preprocessing": ["log1p", "StandardScaler fitted on training only"],
            "prototype_weighting": "row-weighted class mean",
            "best_epoch": components.best_epoch,
            "class_weights": components.class_weights,
            "torch_device": str(device),
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
                "torch": torch.__version__,
            },
        }
        save_components(run_dir, components, config_payload, final_history)
        common.atomic_write_json(
            run_dir / "removed_devices.json", run_entry["removed_devices"]
        )
        predictions.drop(columns=["source_row"]).to_parquet(
            run_dir / "selected_device_predictions.parquet",
            index=False,
            compression="zstd",
        )
        common.atomic_write_json(run_dir / "metrics.json", report)
        metric_rows = []
        for trial_report in report["trials"]:
            trial = trial_report["trial"]
            metric_rows.append(
                common.flatten_metrics(
                    "combined", run_number, trial, trial_report["combined"]
                )
            )
            for dataset in test_datasets:
                metric_rows.append(
                    common.flatten_metrics(
                        dataset,
                        run_number,
                        trial,
                        trial_report["by_dataset"][dataset],
                    )
                )
        pd.DataFrame(metric_rows).to_parquet(
            run_dir / "trial_metrics.parquet", index=False, compression="zstd"
        )
        verify_artifact_reload(
            run_dir,
            selected_frame,
            selected_predictions.drop(columns=["source_row"]),
            config,
            device,
        )
        common.check_stop(stop_file)
        manifest.update(
            {
                "status": "complete",
                "training_counts_after_removal": common.removal_counts(reduced_train),
                "evaluable_test_devices": int(
                    predictions[["dataset", "device_id"]].drop_duplicates().shape[0]
                ),
                "unevaluable_test_devices": len(unevaluable_devices),
                "test_trials": test_trials,
                "best_epoch": components.best_epoch,
                "artifact_reload_verified": True,
            }
        )
        atomic_write_json(run_dir / "_run_manifest.json", manifest)
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
    except BaseException as exc:
        manifest.update({"status": "failed", "error": str(exc)})
        atomic_write_json(run_dir / "_run_manifest.json", manifest)
        common.discard_model_run(run_dir)
        raise


def _load_completed_cnn_result(run_dir: Path) -> dict[str, Any]:
    manifest = common.read_json(run_dir / "_run_manifest.json")
    if manifest.get("status") != "complete" or not manifest.get(
        "artifact_reload_verified"
    ):
        raise PipelineError(f"incomplete model-run checkpoint: {run_dir}")
    for name in (
        "cnn_state.pt",
        "scaler.joblib",
        "prototypes.npy",
        "random_forest.joblib",
        "config.json",
        "selected_device_predictions.parquet",
    ):
        if not (run_dir / name).is_file():
            raise PipelineError(f"incomplete model-run checkpoint: {run_dir / name}")
    report = common.read_json(run_dir / "metrics.json")
    rows = pd.read_parquet(run_dir / "trial_metrics.parquet").to_dict("records")
    return {
        "run": int(run_dir.name.rsplit("_", 1)[1]),
        "run_dir": str(run_dir),
        "summary": report["summary"],
        "metric_rows": rows,
    }


def execute_experiment(args: argparse.Namespace) -> Path:
    train_datasets, test_datasets = validate_dataset_sets(
        args.train_datasets, args.test_datasets
    )
    features_root = args.features_root.resolve()
    labels_root = args.labels_root.resolve()
    label_file = common.validate_label_file_name(
        getattr(args, "label_file", common.DEFAULT_MAC_LABEL_FILE)
    )
    removal_specs = common.parse_removal_specs(
        getattr(args, "remove_devices", None), train_datasets
    )
    config = TrainingConfig(
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        rf_trees=args.rf_trees,
        loader_workers=args.loader_workers,
        rf_jobs=int(getattr(args, "rf_jobs", -1)),
    )
    test_trials = int(getattr(args, "test_trials", 100))
    jobs = int(getattr(args, "jobs", 1))
    resume = bool(getattr(args, "resume", False))
    overwrite = bool(getattr(args, "overwrite", False))
    stop_file_arg = getattr(args, "stop_file", None)
    stop_file = Path(stop_file_arg).resolve() if stop_file_arg is not None else None
    if config.batch_size <= 0 or config.max_epochs <= 0 or config.patience <= 0:
        raise PipelineError("batch size, maximum epochs, and patience must be positive")
    if config.rf_jobs == 0 or config.rf_jobs < -1:
        raise PipelineError("--rf-jobs must be -1 or a positive integer")
    if test_trials <= 0 or jobs <= 0:
        raise PipelineError("test trials and jobs must be positive")
    experiment_dir = args.output_root.resolve() / common.experiment_directory_name(
        train_datasets, test_datasets, features_root, removal_specs, config.seed
    )
    if common.stop_requested(stop_file):
        print(f"Stop requested before startup; output was not changed: {stop_file}")
        return experiment_dir

    device = resolve_torch_device(args.device)
    device_description = torch_device_description(device)
    print(f"selected torch device: {device} ({device_description})", flush=True)
    if jobs > 1 and device.type != "cpu":
        raise PipelineError("--jobs greater than 1 is supported only with a CPU device")
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
    train_data = load_datasets(train_datasets, features_root, labels_root, label_file)
    if common.stop_requested(stop_file):
        print(f"Stop requested during setup; output was not changed: {stop_file}")
        return experiment_dir
    print(f"loading test datasets: {', '.join(test_datasets)}", flush=True)
    test_data = load_datasets(test_datasets, features_root, labels_root, label_file)
    if common.stop_requested(stop_file):
        print(f"Stop requested during setup; output was not changed: {stop_file}")
        return experiment_dir
    classes = common.validate_class_sets(train_data.frame, test_data.frame)
    removal_plan = common.generate_removal_plan(
        train_data.frame,
        removal_specs,
        int(getattr(args, "removal_runs", 5)),
        config.seed,
    )
    resume_configuration = {
        "pipeline": "cnn_pn_rf_mac_v1",
        "feature_provenance": provenance,
        "train_datasets": train_datasets,
        "test_datasets": test_datasets,
        "label_file": label_file,
        "classes": classes,
        "removal_plan": removal_plan,
        "training": asdict(config),
        "test_trials": test_trials,
        "requested_device": args.device,
        "resolved_device": str(device),
    }
    fingerprint = common.stable_fingerprint(
        resume_fingerprint_configuration(resume_configuration)
    )
    migrate_legacy_resource_fingerprint(
        experiment_dir,
        resume_configuration,
        fingerprint,
        resume=resume,
    )
    previous = common.prepare_resumable_output_directory(
        experiment_dir,
        resume=resume,
        overwrite=overwrite,
        fingerprint=fingerprint,
    )
    manifest: dict[str, Any] = {
        "status": "running",
        "pipeline": "cnn_pn_rf",
        "started_at": (previous or {}).get("started_at", common.utc_now()),
        "updated_at": common.utc_now(),
        "completed_at": None,
        "train_datasets": train_datasets,
        "test_datasets": test_datasets,
        "feature_root_tag": common.feature_root_tag(features_root),
        "feature_provenance": provenance,
        "model_runs": len(removal_plan),
        "completed_model_runs": 0,
        "test_trials": test_trials,
        "seed": config.seed,
        "label_file": label_file,
        "classes": classes,
        "requested_device": args.device,
        "resolved_device": str(device),
        "device_description": device_description,
        "configuration_fingerprint": fingerprint,
    }
    atomic_write_json(experiment_dir / "_run_manifest.json", manifest)
    common.atomic_write_json(experiment_dir / "removal_plan.json", removal_plan)
    common.atomic_write_json(
        experiment_dir / "config.json",
        {
            **resume_configuration,
            "features_root": str(features_root),
            "labels_root": str(labels_root),
            "jobs": jobs,
            "stop_file": str(stop_file) if stop_file is not None else None,
            "device_description": device_description,
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
                loaded = _load_completed_cnn_result(run_dir)
            except (OSError, ValueError, KeyError, PipelineError):
                loaded = None
        if loaded is not None:
            results_by_run[run_number] = loaded
        else:
            if run_dir.exists():
                shutil.rmtree(run_dir)
            pending.append(entry)
    manifest["completed_model_runs"] = len(results_by_run)
    atomic_write_json(experiment_dir / "_run_manifest.json", manifest)

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
            config,
            str(device),
            test_trials,
            str(stop_file) if stop_file is not None else None,
        )

        def completed(result: dict[str, Any]) -> None:
            results_by_run[int(result["run"])] = result
            manifest["completed_model_runs"] = len(results_by_run)
            manifest["updated_at"] = common.utc_now()
            atomic_write_json(experiment_dir / "_run_manifest.json", manifest)

        stopped = common.run_bounded_model_jobs(
            pending,
            _execute_cnn_model_run,
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
            atomic_write_json(experiment_dir / "_run_manifest.json", manifest)
            print("Stopped gracefully; remove the stop file and rerun with --resume.")
            return experiment_dir

        results = [results_by_run[index] for index in sorted(results_by_run)]
        if len(results) != len(removal_plan):
            raise PipelineError("not all model runs completed")
        common.check_stop(stop_file)
        all_metric_rows = [row for result in results for row in result["metric_rows"]]
        common.atomic_write_parquet(
            pd.DataFrame(all_metric_rows),
            experiment_dir / "all_trial_metrics.parquet",
        )
        model_combined_means = [
            {
                key: value["mean"] if value is not None else None
                for key, value in result["summary"]["combined"].items()
            }
            for result in results
        ]
        aggregate = {
            "model_runs": [
                {"run": result["run"], "summary": result["summary"]}
                for result in results
            ],
            "across_model_run_means": common.summarize_metric_dicts(
                model_combined_means
            ),
        }
        common.check_stop(stop_file)
        common.atomic_write_json(experiment_dir / "aggregate_metrics.json", aggregate)
        mean_accuracy = aggregate["across_model_run_means"]["accuracy"]["mean"]
        now = common.utc_now()
        manifest.update(
            {
                "status": "complete",
                "completed_model_runs": len(results),
                "mean_accuracy_across_model_run_means": mean_accuracy,
                "completed_at": now,
                "updated_at": now,
            }
        )
        atomic_write_json(experiment_dir / "_run_manifest.json", manifest)
        print(
            f"complete: mean_model_accuracy={mean_accuracy:.4f}; output={experiment_dir}",
            flush=True,
        )
        return experiment_dir
    except common.StopRequested:
        manifest.update(
            {"status": "stopped", "stopped_at": common.utc_now(), "updated_at": common.utc_now()}
        )
        atomic_write_json(experiment_dir / "_run_manifest.json", manifest)
        print("Stopped gracefully; remove the stop file and rerun with --resume.")
        return experiment_dir
    except BaseException as exc:
        manifest.update(
            {"status": "failed", "error": str(exc), "updated_at": common.utc_now()}
        )
        atomic_write_json(experiment_dir / "_run_manifest.json", manifest)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-datasets", nargs="+", required=True, choices=DATASETS)
    parser.add_argument("--test-datasets", nargs="+", required=True, choices=DATASETS)
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
        "--output-root", type=Path, default=Path("classification_runs_mac")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps", "xpu"), default="auto"
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rf-trees", type=int, default=100)
    parser.add_argument(
        "--rf-jobs",
        type=int,
        default=-1,
        help="parallel Random Forest workers; -1 uses all assigned CPUs",
    )
    parser.add_argument("--loader-workers", type=int, default=0)
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
    parser.add_argument("--test-trials", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--stop-file", type=Path)
    lifecycle = parser.add_mutually_exclusive_group()
    lifecycle.add_argument("--resume", action="store_true")
    lifecycle.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        execute_experiment(parse_args(argv))
    except (PipelineError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
