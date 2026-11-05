#!/usr/bin/env python3
"""Shared data, removal, reporting, and metric helpers for classifiers."""

from __future__ import annotations

import json
import datetime as dt
import concurrent.futures
import hashlib
import math
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from typing import Callable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)


DATASETS = ("NCSU", "UNSW1", "UNSW2", "YourThing")
DEFAULT_LABEL_FILE = "device_to_be_include.json"
DEFAULT_MAC_LABEL_FILE = "mac_mapping_binary_labels.json"
FEATURE_ALGORITHM_VERSION = 1
MAC_RE = re.compile(r"^(?:[0-9a-f]{2}:){5}[0-9a-f]{2}$")
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


class PipelineError(RuntimeError):
    """A configuration, data, training, or validation failure."""


class StopRequested(RuntimeError):
    """Cooperative, successful cancellation requested through a stop file."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def stop_requested(stop_file: Path | str | None) -> bool:
    return stop_file is not None and Path(stop_file).exists()


def check_stop(stop_file: Path | str | None) -> None:
    if stop_requested(stop_file):
        raise StopRequested(f"stop requested by {stop_file}")


def validate_label_file_name(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or Path(value).is_absolute()
        or "/" in value
        or "\\" in value
    ):
        raise PipelineError("--label-file must be a non-empty filename, not a path")
    return value


@dataclass
class LoadedData:
    frame: pd.DataFrame
    unevaluable_devices: list[dict[str, Any]]


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.parquet")
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def stable_fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PipelineError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PipelineError(f"invalid JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PipelineError(f"JSON file must contain an object: {path}")
    return value


def prepare_resumable_output_directory(
    path: Path,
    *,
    resume: bool,
    overwrite: bool,
    fingerprint: str,
) -> dict[str, Any] | None:
    """Prepare a root without mixing artifacts from incompatible invocations."""
    if resume and overwrite:
        raise PipelineError("--resume and --overwrite are mutually exclusive")
    manifest_path = path / "_run_manifest.json"
    if path.exists():
        if overwrite:
            shutil.rmtree(path)
            path.mkdir(parents=True)
            return None
        if not resume:
            raise PipelineError(f"run output already exists: {path}; pass --resume or --overwrite")
        manifest = read_json(manifest_path)
        if manifest.get("configuration_fingerprint") != fingerprint:
            raise PipelineError(
                "existing run manifest is incompatible with this --resume configuration"
            )
        for temporary in path.glob(".model_run_*.tmp-*"):
            if temporary.is_dir():
                shutil.rmtree(temporary)
        return manifest
    if resume:
        raise PipelineError(f"cannot --resume missing run output: {path}")
    path.mkdir(parents=True)
    return None


def completed_model_run(path: Path) -> bool:
    try:
        manifest = read_json(path / "_run_manifest.json")
    except PipelineError:
        return False
    return manifest.get("status") == "complete"


def create_model_run_staging_directory(
    experiment_dir: Path, run_number: int
) -> tuple[Path, Path]:
    final = experiment_dir / f"model_run_{run_number:03d}"
    if final.exists():
        raise PipelineError(f"model run output already exists: {final}")
    temporary = experiment_dir / (
        f".model_run_{run_number:03d}.tmp-{uuid.uuid4().hex}"
    )
    temporary.mkdir(parents=True, exist_ok=False)
    return temporary, final


def commit_model_run(temporary: Path, final: Path) -> None:
    if final.exists():
        raise PipelineError(f"model run output already exists: {final}")
    os.replace(temporary, final)


def discard_model_run(temporary: Path) -> None:
    if temporary.exists():
        shutil.rmtree(temporary)


def run_bounded_model_jobs(
    entries: Sequence[Mapping[str, Any]],
    worker: Callable[..., dict[str, Any]],
    worker_args: tuple[Any, ...],
    jobs: int,
    stop_file: Path | None,
    on_complete: Callable[[dict[str, Any]], None],
) -> bool:
    """Run no more than ``jobs`` processes and stop submitting cooperatively."""
    if jobs <= 0:
        raise PipelineError("--jobs must be positive")
    if jobs == 1:
        for entry in entries:
            if stop_requested(stop_file):
                return True
            try:
                result = worker(entry, *worker_args)
            except StopRequested:
                return True
            on_complete(result)
        return stop_requested(stop_file)

    pending_entries = iter(entries)
    stopped = False
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
        futures: dict[concurrent.futures.Future[dict[str, Any]], Mapping[str, Any]] = {}

        def submit_one() -> bool:
            if stop_requested(stop_file):
                return False
            try:
                entry = next(pending_entries)
            except StopIteration:
                return False
            futures[executor.submit(worker, entry, *worker_args)] = entry
            return True

        for _ in range(min(jobs, len(entries))):
            if not submit_one():
                stopped = stop_requested(stop_file)
                break
        while futures:
            if stop_requested(stop_file):
                stopped = True
            done, _ = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                futures.pop(future)
                try:
                    result = future.result()
                except StopRequested:
                    stopped = True
                else:
                    on_complete(result)
                if not stopped and not submit_one() and stop_requested(stop_file):
                    stopped = True
    return stopped or stop_requested(stop_file)


def validate_dataset_sets(
    train_datasets: Sequence[str], test_datasets: Sequence[str]
) -> tuple[list[str], list[str]]:
    train_supplied = list(dict.fromkeys(train_datasets))
    test_supplied = list(dict.fromkeys(test_datasets))
    unknown = (set(train_supplied) | set(test_supplied)) - set(DATASETS)
    if unknown:
        raise PipelineError(f"unknown datasets: {', '.join(sorted(unknown))}")
    train = [dataset for dataset in DATASETS if dataset in train_supplied]
    test = [dataset for dataset in DATASETS if dataset in test_supplied]
    if not train or not test:
        raise PipelineError("training and test dataset lists must both be non-empty")
    overlap = set(train) & set(test)
    if overlap:
        raise PipelineError(
            f"training and test datasets overlap: {', '.join(sorted(overlap))}"
        )
    return train, test


def load_label_map(
    labels_root: Path, dataset: str, label_file: str = DEFAULT_LABEL_FILE
) -> dict[str, int]:
    path = labels_root / dataset / "labels" / validate_label_file_name(label_file)
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise PipelineError(f"missing label file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PipelineError(f"invalid label JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict) or not payload:
        raise PipelineError(f"label file must contain a non-empty object: {path}")
    labels: dict[str, int] = {}
    for device_id, metadata in payload.items():
        if not isinstance(device_id, str) or not isinstance(metadata, dict):
            raise PipelineError(f"invalid device label entry in {path}")
        label = metadata.get("is_iot_device")
        if type(label) is not int or label < 0:
            raise PipelineError(
                f"{dataset}/{device_id}: is_iot_device must be a non-negative integer"
            )
        labels[device_id.lower()] = label
    return labels


def _read_mac_label_map(
    labels_root: Path, dataset: str, label_file: str
) -> tuple[dict[str, int], str]:
    """Read the feature-builder label format without normalizing identifiers."""
    path = labels_root / dataset / validate_label_file_name(label_file)
    try:
        raw_bytes = path.read_bytes()
    except FileNotFoundError as exc:
        raise PipelineError(f"missing label file: {path}") from exc
    try:
        payload = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PipelineError(f"invalid label JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict) or not payload:
        raise PipelineError(f"label file must contain a non-empty object: {path}")
    labels: dict[str, int] = {}
    for device_id, metadata in payload.items():
        if not isinstance(device_id, str) or not MAC_RE.fullmatch(device_id):
            raise PipelineError(
                f"{dataset}/{device_id}: label key must be a lowercase MAC address"
            )
        if not isinstance(metadata, dict):
            raise PipelineError(f"invalid device label entry in {path}: {device_id}")
        label = metadata.get("is_iot_device")
        if type(label) is not int or label < 0 or label > 2**31 - 1:
            raise PipelineError(
                f"{dataset}/{device_id}: is_iot_device must be a non-negative "
                "integer in the int32 range"
            )
        labels[device_id] = label
    return labels, hashlib.sha256(raw_bytes).hexdigest()


def load_mac_label_map(
    labels_root: Path,
    dataset: str,
    label_file: str = DEFAULT_MAC_LABEL_FILE,
) -> dict[str, int]:
    return _read_mac_label_map(labels_root, dataset, label_file)[0]


def validate_mac_feature_manifest(
    features_root: Path,
    labels_root: Path,
    label_file: str,
    datasets: Sequence[str],
) -> dict[str, Any]:
    """Validate that a feature root is a complete matching MAC-only build."""
    manifest_path = features_root / "_run_manifest.json"
    try:
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)
    except FileNotFoundError as exc:
        raise PipelineError(f"missing feature build manifest: {manifest_path}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PipelineError(f"invalid feature build manifest {manifest_path}: {exc}") from exc
    if manifest.get("status") != "complete":
        raise PipelineError(f"feature build manifest is not complete: {manifest_path}")
    if manifest.get("algorithm_version") != FEATURE_ALGORITHM_VERSION:
        raise PipelineError(
            f"unsupported feature algorithm version in {manifest_path}: "
            f"{manifest.get('algorithm_version')!r}"
        )
    configuration = manifest.get("configuration")
    if not isinstance(configuration, dict) or configuration.get("label_file") != label_file:
        raise PipelineError(
            f"feature build label file does not match --label-file: {manifest_path}"
        )
    manifest_datasets = manifest.get("datasets")
    if not isinstance(manifest_datasets, dict):
        raise PipelineError(f"feature build manifest has no dataset inventory: {manifest_path}")
    label_hashes: dict[str, str] = {}
    for dataset in datasets:
        entry = manifest_datasets.get(dataset)
        if not isinstance(entry, dict):
            raise PipelineError(f"feature build manifest is missing dataset {dataset}")
        if entry.get("identifier_type") != "mac":
            raise PipelineError(f"{dataset}: feature build identifier_type is not mac")
        _, label_hash = _read_mac_label_map(labels_root, dataset, label_file)
        if entry.get("label_sha256") != label_hash:
            raise PipelineError(
                f"{dataset}: feature build label hash does not match {label_file}"
            )
        label_hashes[dataset] = label_hash
    return {
        "algorithm_version": FEATURE_ALGORITHM_VERSION,
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "label_file": label_file,
        "label_sha256": label_hashes,
    }


def _validate_mac_feature_table(
    table: pa.Table,
    dataset: str,
    device_id: str,
    authoritative_label: int,
    path: Path,
) -> None:
    expected = {
        "dataset",
        "capture_id",
        "window",
        "device_id",
        "device_id_type",
        "is_iot_device",
        *FEATURE_COLUMNS,
    }
    missing = expected - set(table.column_names)
    if missing:
        raise PipelineError(f"{path}: missing columns {', '.join(sorted(missing))}")
    if not table.num_rows:
        return
    constants = {
        "dataset": dataset,
        "device_id": device_id,
        "device_id_type": "mac",
        "is_iot_device": authoritative_label,
    }
    for column, expected_value in constants.items():
        actual = set(table.column(column).to_pylist())
        if actual != {expected_value}:
            if column == "is_iot_device":
                raise PipelineError(
                    f"{path}: repeated labels {sorted(actual)} disagree with label "
                    f"JSON value {expected_value}"
                )
            raise PipelineError(
                f"{path}: {column} values {sorted(actual, key=str)} do not match "
                f"{expected_value!r}"
            )


def load_mac_datasets(
    datasets: Sequence[str],
    features_root: Path,
    labels_root: Path,
    label_file: str = DEFAULT_MAC_LABEL_FILE,
) -> LoadedData:
    """Load only builder-validated MAC identities from the new feature layout."""
    validate_mac_feature_manifest(features_root, labels_root, label_file, datasets)
    frames: list[pd.DataFrame] = []
    unevaluable: list[dict[str, Any]] = []
    read_columns = [
        "dataset",
        "capture_id",
        "window",
        "device_id",
        "device_id_type",
        "is_iot_device",
        *FEATURE_COLUMNS,
    ]
    for dataset in datasets:
        labels = load_mac_label_map(labels_root, dataset, label_file)
        dataset_dir = features_root / dataset
        if not dataset_dir.is_dir():
            raise PipelineError(f"missing feature directory: {dataset_dir}")
        unexpected = {
            path.name
            for path in dataset_dir.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        } - set(labels)
        if unexpected:
            raise PipelineError(
                f"{dataset}: feature folders absent from label JSON: {sorted(unexpected)}"
            )
        for device_id, label in labels.items():
            path = dataset_dir / device_id / "features.parquet"
            if not path.is_file():
                raise PipelineError(f"missing feature file: {path}")
            table = pq.read_table(path, columns=read_columns)
            _validate_mac_feature_table(table, dataset, device_id, label, path)
            if not table.num_rows:
                unevaluable.append(
                    {
                        "dataset": dataset,
                        "device_id": device_id,
                        "true_label": label,
                        "reason": "no feature rows",
                    }
                )
                continue
            frame = table.drop(["dataset", "device_id_type", "is_iot_device"]).to_pandas()
            frame.insert(0, "dataset", dataset)
            frame["device_id"] = frame["device_id"].astype(str)
            frame["label"] = np.int32(label)
            frames.append(frame)
    if not frames:
        raise PipelineError("selected datasets contain no feature rows")
    result = pd.concat(frames, ignore_index=True)
    result["window"] = result["window"].astype("int32")
    result["label"] = result["label"].astype("int32")
    return LoadedData(result, unevaluable)


def _validate_feature_table(
    table: pa.Table, dataset: str, device_id: str, authoritative_label: int, path: Path
) -> None:
    expected = {"capture_id", "window", "device_id", "is_iot_device", *FEATURE_COLUMNS}
    missing = expected - set(table.column_names)
    if missing:
        raise PipelineError(f"{path}: missing columns {', '.join(sorted(missing))}")
    if not table.num_rows:
        return
    parquet_ids = {
        str(value).lower() for value in table.column("device_id").to_pylist()
    }
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
    label_file: str = DEFAULT_LABEL_FILE,
) -> LoadedData:
    frames: list[pd.DataFrame] = []
    unevaluable: list[dict[str, Any]] = []
    read_columns = [
        "capture_id",
        "window",
        "device_id",
        "is_iot_device",
        *FEATURE_COLUMNS,
    ]
    for dataset in datasets:
        labels = load_label_map(labels_root, dataset, label_file)
        dataset_dir = features_root / dataset
        if not dataset_dir.is_dir():
            raise PipelineError(f"missing feature directory: {dataset_dir}")
        unexpected = {
            path.name.lower()
            for path in dataset_dir.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        } - set(labels)
        if unexpected:
            raise PipelineError(
                f"{dataset}: feature folders absent from label JSON: {sorted(unexpected)}"
            )
        for device_id, label in labels.items():
            path = dataset_dir / device_id / "features.parquet"
            if not path.is_file():
                raise PipelineError(f"missing feature file: {path}")
            table = pq.read_table(path, columns=read_columns)
            _validate_feature_table(table, dataset, device_id, label, path)
            if not table.num_rows:
                unevaluable.append(
                    {
                        "dataset": dataset,
                        "device_id": device_id,
                        "true_label": label,
                        "reason": "no feature rows",
                    }
                )
                continue
            frame = table.drop(["is_iot_device"]).to_pandas()
            frame.insert(0, "dataset", dataset)
            frame["device_id"] = frame["device_id"].astype(str).str.lower()
            frame["label"] = np.int32(label)
            frames.append(frame)
    if not frames:
        raise PipelineError("selected datasets contain no feature rows")
    result = pd.concat(frames, ignore_index=True)
    result["window"] = result["window"].astype("int32")
    result["label"] = result["label"].astype("int32")
    return LoadedData(result, unevaluable)


def validate_class_sets(
    train_frame: pd.DataFrame, test_frame: pd.DataFrame
) -> list[int]:
    """Validate the global class contract and return the ordered training classes."""
    classes = sorted(int(label) for label in train_frame["label"].unique())
    if len(classes) < 2:
        raise PipelineError(
            f"training data must contain at least two classes; found {classes}"
        )
    expected = list(range(classes[-1] + 1))
    if classes != expected:
        raise PipelineError(
            f"training classes must be contiguous and start at 0; found {classes}"
        )
    test_classes = sorted(int(label) for label in test_frame["label"].unique())
    unknown = sorted(set(test_classes) - set(classes))
    if unknown:
        raise PipelineError(
            f"test data contains classes absent from training: {unknown}"
        )
    return classes


def raw_feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    missing = set(FEATURE_COLUMNS) - set(frame.columns)
    if missing:
        raise PipelineError(f"missing feature columns: {', '.join(sorted(missing))}")
    values = frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(values).all():
        raise PipelineError("feature matrix contains NaN or infinite values")
    if (values < 0).any():
        raise PipelineError("features must be nonnegative")
    return values


def parse_removal_specs(
    values: Sequence[Sequence[str]] | None, train_datasets: Sequence[str]
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for triplet in values or []:
        if len(triplet) != 3:
            raise PipelineError("--remove-devices requires DATASET CLASS COUNT")
        dataset, raw_class, raw_count = triplet
        if dataset not in DATASETS:
            raise PipelineError(f"unknown removal dataset: {dataset}")
        if dataset not in train_datasets:
            raise PipelineError(
                f"removal dataset is not selected for training: {dataset}"
            )
        try:
            label = int(raw_class)
            count = int(raw_count)
        except ValueError as exc:
            raise PipelineError("removal class and count must be integers") from exc
        if label < 0:
            raise PipelineError("removal class must be non-negative")
        if count <= 0:
            raise PipelineError("removal count must be positive")
        key = (dataset, label)
        if key in seen:
            raise PipelineError(
                f"duplicate --remove-devices entry for {dataset}, class {label}"
            )
        seen.add(key)
        specs.append({"dataset": dataset, "class": label, "count": count})
    specs.sort(key=lambda spec: (spec["dataset"], spec["class"]))
    return specs


def generate_removal_plan(
    train_frame: pd.DataFrame,
    removal_specs: Sequence[Mapping[str, Any]],
    removal_runs: int,
    seed: int,
) -> list[dict[str, Any]]:
    if not removal_specs:
        return [{"run": 1, "removed_devices": []}]
    if removal_runs <= 0:
        raise PipelineError("--removal-runs must be positive")
    selections: dict[tuple[str, int], list[list[str]]] = {}
    for spec in removal_specs:
        dataset = str(spec["dataset"])
        label = int(spec["class"])
        count = int(spec["count"])
        devices = sorted(
            train_frame.loc[
                (train_frame["dataset"] == dataset) & (train_frame["label"] == label),
                "device_id",
            ].unique()
        )
        if count > len(devices):
            raise PipelineError(
                f"cannot remove {count} devices from {dataset}, class {label}: only "
                f"{len(devices)} evaluable devices are available"
            )
        if math.comb(len(devices), count) < removal_runs:
            raise PipelineError(
                f"cannot produce {removal_runs} distinct {dataset}, class {label} "
                f"removal sets of size {count} from {len(devices)} devices"
            )
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, DATASETS.index(dataset), label])
        )
        unique: set[tuple[str, ...]] = set()
        dataset_runs: list[list[str]] = []
        while len(dataset_runs) < removal_runs:
            chosen = tuple(
                sorted(rng.choice(devices, size=count, replace=False).tolist())
            )
            if chosen not in unique:
                unique.add(chosen)
                dataset_runs.append(list(chosen))
        selections[(dataset, label)] = dataset_runs
    plan = []
    for run_index in range(removal_runs):
        removed_devices = [
            {
                "dataset": str(spec["dataset"]),
                "class": int(spec["class"]),
                "device_ids": selections[(str(spec["dataset"]), int(spec["class"]))][
                    run_index
                ],
            }
            for spec in removal_specs
        ]
        apply_removal(train_frame, removed_devices)
        plan.append({"run": run_index + 1, "removed_devices": removed_devices})
    return plan


def apply_removal(
    train_frame: pd.DataFrame, removed_devices: Sequence[Mapping[str, Any]]
) -> pd.DataFrame:
    device_labels = {
        (str(row.dataset), str(row.device_id).lower()): int(row.label)
        for row in train_frame[["dataset", "device_id", "label"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    remove_keys: set[tuple[str, str]] = set()
    for entry in removed_devices:
        dataset = str(entry["dataset"])
        label = int(entry["class"])
        for raw_device_id in entry["device_ids"]:
            key = (dataset, str(raw_device_id).lower())
            actual_label = device_labels.get(key)
            if actual_label is None:
                raise PipelineError(
                    f"cannot remove unknown training device {dataset}/{key[1]}"
                )
            if actual_label != label:
                raise PipelineError(
                    f"cannot remove {dataset}/{key[1]} as class {label}: "
                    f"authoritative class is {actual_label}"
                )
            remove_keys.add(key)
    keys = list(
        zip(train_frame["dataset"], train_frame["device_id"].astype(str).str.lower())
    )
    keep = np.fromiter((key not in remove_keys for key in keys), bool, count=len(keys))
    reduced = train_frame.loc[keep].reset_index(drop=True)
    original_labels = sorted(int(label) for label in train_frame["label"].unique())
    remaining_labels = sorted(int(label) for label in reduced["label"].unique())
    if remaining_labels != original_labels:
        raise PipelineError(
            "device removal must not eliminate a training class; "
            f"before={original_labels}, after={remaining_labels}"
        )
    return reduced


def removal_counts(frame: pd.DataFrame) -> dict[str, Any]:
    devices = frame[["dataset", "device_id", "label"]].drop_duplicates()
    return {
        "rows": int(len(frame)),
        "devices": int(len(devices)),
        "rows_by_class": {
            str(label): int(count)
            for label, count in frame["label"].value_counts().sort_index().items()
        },
        "devices_by_class": {
            str(label): int(count)
            for label, count in devices["label"].value_counts().sort_index().items()
        },
    }


def feature_root_tag(features_root: Path) -> str:
    name = features_root.resolve().name or "features"
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-._") or "features"


def experiment_directory_name(
    train_datasets: Sequence[str],
    test_datasets: Sequence[str],
    features_root: Path,
    removal_specs: Sequence[Mapping[str, Any]],
    seed: int,
) -> str:
    parts = [
        f"train-{'+'.join(train_datasets)}",
        f"test-{'+'.join(test_datasets)}",
        f"features-{feature_root_tag(features_root)}",
    ]
    if removal_specs:
        removal = "+".join(
            f"{spec['dataset']}-class{spec['class']}-{spec['count']}"
            for spec in sorted(
                removal_specs,
                key=lambda item: (str(item["dataset"]), int(item["class"])),
            )
        )
        parts.append(f"remove-devices-{removal}")
    parts.append(f"seed-{seed}")
    return "__".join(parts)


def prepare_output_directory(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise PipelineError(f"run output already exists: {path}; pass --overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _macro(values: Sequence[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    return float(np.mean(defined)) if defined else None


def compute_metrics(
    true_labels: Sequence[int],
    predictions: Sequence[int],
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    classes: Sequence[int],
) -> dict[str, Any]:
    y_true = np.asarray(true_labels, dtype=np.int64)
    y_pred = np.asarray(predictions, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    ordered_classes = [int(label) for label in classes]
    if ordered_classes != list(range(len(ordered_classes))):
        raise PipelineError(
            f"metric classes must be contiguous and start at 0; found {ordered_classes}"
        )
    if y_true.shape != y_pred.shape:
        raise PipelineError("true labels and predictions differ in shape")
    if not len(y_true):
        return {
            "n_samples": 0,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "specificity": None,
            "balanced_accuracy": None,
            "roc_auc": None,
            "pr_auc": None,
            "per_class": {},
            "confusion_matrix": {"labels": ordered_classes, "matrix": []},
            "warnings": ["no evaluable observations"],
        }
    if scores.shape != (len(y_true), len(ordered_classes)):
        raise PipelineError(
            "probability matrix shape does not match observations and classes: "
            f"{scores.shape} != {(len(y_true), len(ordered_classes))}"
        )
    if not np.isfinite(scores).all():
        raise PipelineError("probability matrix contains NaN or infinite values")
    allowed = set(ordered_classes)
    unexpected = sorted((set(y_true.tolist()) | set(y_pred.tolist())) - allowed)
    if unexpected:
        raise PipelineError(f"predictions contain unknown classes: {unexpected}")
    matrix = confusion_matrix(y_true, y_pred, labels=ordered_classes)
    warnings: list[str] = []
    per_class: dict[str, dict[str, Any]] = {}
    for index, label in enumerate(ordered_classes):
        tp = int(matrix[index, index])
        fn = int(matrix[index, :].sum() - tp)
        fp = int(matrix[:, index].sum() - tp)
        tn = int(matrix.sum() - tp - fn - fp)
        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, tp + fn)
        specificity = _safe_ratio(tn, tn + fp)
        f1 = (
            _safe_ratio(2 * precision * recall, precision + recall)
            if precision is not None and recall is not None
            else None
        )
        binary_truth = (y_true == label).astype(np.int8)
        has_positive = bool(binary_truth.any())
        has_negative = bool((binary_truth == 0).any())
        roc_auc = (
            float(roc_auc_score(binary_truth, scores[:, index]))
            if has_positive and has_negative
            else None
        )
        pr_auc = (
            float(average_precision_score(binary_truth, scores[:, index]))
            if has_positive and has_negative
            else None
        )
        class_metrics = {
            "support": int(tp + fn),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        }
        per_class[str(label)] = class_metrics
        for metric_name, value in class_metrics.items():
            if metric_name != "support" and value is None:
                warnings.append(f"class {label} {metric_name} undefined")

    precision = _macro([item["precision"] for item in per_class.values()])
    recall = _macro([item["recall"] for item in per_class.values()])
    f1 = _macro([item["f1"] for item in per_class.values()])
    specificity = _macro([item["specificity"] for item in per_class.values()])
    roc_auc = _macro([item["roc_auc"] for item in per_class.values()])
    pr_auc = _macro([item["pr_auc"] for item in per_class.values()])
    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_accuracy": recall,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "per_class": per_class,
        "confusion_matrix": {
            "labels": ordered_classes,
            "matrix": matrix.astype(int).tolist(),
        },
        "warnings": warnings,
    }


def evaluate_predictions(
    frame: pd.DataFrame,
    test_datasets: Sequence[str],
    classes: Sequence[int],
) -> dict[str, Any]:
    probability_columns = [f"probability_class_{label}" for label in classes]

    def evaluate(subset: pd.DataFrame) -> dict[str, Any]:
        return compute_metrics(
            subset["true_label"],
            subset["predicted_label"],
            subset.loc[:, probability_columns].to_numpy(dtype=np.float64),
            classes,
        )

    return {
        "combined": evaluate(frame),
        "by_dataset": {
            dataset: evaluate(frame[frame["dataset"] == dataset])
            for dataset in test_datasets
        },
    }


def summarize_metric_dicts(metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc",
    )
    summary: dict[str, Any] = {}
    for key in keys:
        values = np.asarray(
            [metric[key] for metric in metrics if metric.get(key) is not None],
            dtype=np.float64,
        )
        summary[key] = (
            {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "max": float(values.max()),
                "observations": int(len(values)),
            }
            if len(values)
            else None
        )
    return summary


def flatten_metrics(
    scope: str, run: int, trial: int | None, metrics: Mapping[str, Any]
) -> dict[str, Any]:
    row: dict[str, Any] = {"scope": scope, "run": run, "trial": trial}
    for key, value in metrics.items():
        if key == "confusion_matrix":
            labels = value["labels"]
            matrix = value["matrix"]
            if not matrix:
                continue
            for true_index, true_label in enumerate(labels):
                for predicted_index, predicted_label in enumerate(labels):
                    row[f"true_{true_label}_pred_{predicted_label}"] = matrix[
                        true_index
                    ][predicted_index]
        elif key == "per_class":
            for label, class_metrics in value.items():
                for metric_name, metric_value in class_metrics.items():
                    row[f"{metric_name}_class_{label}"] = metric_value
        elif key == "warnings":
            row["warnings"] = json.dumps(value)
        else:
            row[key] = value
    return row
