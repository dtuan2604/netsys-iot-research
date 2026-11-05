#!/usr/bin/env python3
"""Build MAC-addressed fixed-window device features from packet Parquet files."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - exercised by the CLI environment
    raise SystemExit(
        "pyarrow is required. Install code/requirements-feature-construction.txt "
        "with the Python interpreter used to run this script."
    ) from exc


DATASETS = ("NCSU", "UNSW1", "UNSW2", "YourThing")
DEFAULT_LABEL_FILE = "mac_mapping_binary_labels.json"
ALGORITHM_VERSION = 1
EXTRACT_SCHEMA_VERSION = 1
MAC_RE = re.compile(r"^(?:[0-9a-f]{2}:){5}[0-9a-f]{2}$")
PACKET_COLUMNS = ("timestamp_utc", "frame_len", "eth_src", "eth_dst")
NANOSECONDS_PER_SECOND = 1_000_000_000

FEATURE_SCHEMA = pa.schema(
    [
        pa.field("dataset", pa.string(), nullable=False),
        pa.field("capture_id", pa.string(), nullable=False),
        pa.field("window", pa.int32(), nullable=False),
        pa.field("device_id", pa.string(), nullable=False),
        pa.field("device_id_type", pa.string(), nullable=False),
        pa.field("device_name", pa.string(), nullable=False),
        pa.field("is_iot_device", pa.int32(), nullable=False),
        pa.field("device_type", pa.string(), nullable=True),
        pa.field("vendor", pa.string(), nullable=True),
        pa.field("outgoing_packet_count", pa.int64(), nullable=False),
        pa.field("outgoing_total_bytes", pa.int64(), nullable=False),
        pa.field("incoming_packet_count", pa.int64(), nullable=False),
        pa.field("incoming_total_bytes", pa.int64(), nullable=False),
        pa.field("aggregated_packet_count", pa.int64(), nullable=False),
        pa.field("aggregated_total_bytes", pa.int64(), nullable=False),
        pa.field("idle_time", pa.float64(), nullable=False),
    ]
)


class PipelineError(RuntimeError):
    """An invalid configuration, source, or output condition."""


@dataclass
class DirectionStats:
    packet_count: int = 0
    total_bytes: int = 0
    first_timestamp_ns: int | None = None
    last_timestamp_ns: int | None = None

    def merge(
        self, packet_count: int, total_bytes: int, first_ns: int, last_ns: int
    ) -> None:
        self.packet_count += packet_count
        self.total_bytes += total_bytes
        if self.first_timestamp_ns is None or first_ns < self.first_timestamp_ns:
            self.first_timestamp_ns = first_ns
        if self.last_timestamp_ns is None or last_ns > self.last_timestamp_ns:
            self.last_timestamp_ns = last_ns


@dataclass
class WindowStats:
    outgoing: DirectionStats = field(default_factory=DirectionStats)
    incoming: DirectionStats = field(default_factory=DirectionStats)


@dataclass(frozen=True)
class DateJob:
    dataset: str
    capture_id: str
    labels: dict[str, dict[str, Any]]
    label_sha256: str
    files: tuple[str, ...]
    source_inventory: tuple[dict[str, Any], ...]
    capture_start_ns: int
    window_seconds: int
    batch_rows: int
    stage_root: str
    resume: bool
    stop_file: str | None


@dataclass(frozen=True)
class ConsolidationJob:
    window_root: str
    dataset: str
    capture_ids: tuple[str, ...]
    device_id: str
    metadata: dict[str, Any]
    expected_rows: int
    label_sha256: str
    stop_file: str | None


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


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


def stop_requested(stop_file: str | Path | None) -> bool:
    return stop_file is not None and Path(stop_file).is_file()


def load_labels(
    labels_root: Path, dataset: str, label_file: str = DEFAULT_LABEL_FILE
) -> tuple[dict[str, dict[str, Any]], str]:
    path = labels_root / dataset / validate_label_file_name(label_file)
    try:
        raw_bytes = path.read_bytes()
        raw = json.loads(raw_bytes)
    except FileNotFoundError as exc:
        raise PipelineError(f"{dataset}: missing label file {path}") from exc
    except json.JSONDecodeError as exc:
        raise PipelineError(f"{dataset}: invalid JSON in {path}: {exc}") from exc
    if not isinstance(raw, dict) or not raw:
        raise PipelineError(f"{dataset}: label file must contain a non-empty object")

    labels: dict[str, dict[str, Any]] = {}
    for original_key, metadata in raw.items():
        if not isinstance(original_key, str) or not isinstance(metadata, dict):
            raise PipelineError(
                f"{dataset}: every label must map a string MAC key to an object"
            )
        if original_key != original_key.lower() or not MAC_RE.fullmatch(original_key):
            raise PipelineError(
                f"{dataset}: invalid lowercase MAC device key: {original_key!r}"
            )
        if original_key in labels:
            raise PipelineError(f"{dataset}: duplicate MAC device key {original_key}")

        device_name = metadata.get("device_name")
        is_iot_device = metadata.get("is_iot_device")
        if not isinstance(device_name, str) or not device_name:
            raise PipelineError(
                f"{dataset}/{original_key}: device_name must be a non-empty string"
            )
        if (
            type(is_iot_device) is not int
            or is_iot_device < 0
            or is_iot_device > 2**31 - 1
        ):
            raise PipelineError(
                f"{dataset}/{original_key}: is_iot_device must be a non-negative "
                "integer in the int32 range"
            )
        for nullable_field in ("device_type", "vendor"):
            value = metadata.get(nullable_field)
            if value is not None and not isinstance(value, str):
                raise PipelineError(
                    f"{dataset}/{original_key}: {nullable_field} must be a string or null"
                )
        labels[original_key] = {
            "device_name": device_name,
            "is_iot_device": is_iot_device,
            "device_type": metadata.get("device_type"),
            "vendor": metadata.get("vendor"),
        }
    return labels, hashlib.sha256(raw_bytes).hexdigest()


def load_extract_manifest(metadata_root: Path) -> dict[str, Any]:
    path = metadata_root / "_extract_manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PipelineError(f"missing extraction manifest: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PipelineError(f"invalid extraction manifest {path}: {exc}") from exc
    if manifest.get("status") != "complete":
        raise PipelineError(f"extraction manifest is not complete: {path}")
    if manifest.get("schema_version") != EXTRACT_SCHEMA_VERSION:
        raise PipelineError(
            f"unsupported extraction schema version in {path}: "
            f"{manifest.get('schema_version')!r}"
        )
    if not isinstance(manifest.get("captures"), dict):
        raise PipelineError(f"extraction manifest has no capture inventory: {path}")
    return manifest


def _safe_metadata_path(metadata_root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise PipelineError("extraction manifest contains an invalid output path")
    candidate = (metadata_root / relative).resolve()
    try:
        candidate.relative_to(metadata_root.resolve())
    except ValueError as exc:
        raise PipelineError(
            f"extraction manifest output escapes metadata root: {relative}"
        ) from exc
    return candidate


def create_jobs(
    metadata_root: Path,
    labels_root: Path,
    window_root: Path,
    datasets: Sequence[str],
    selected_dates: set[str] | None,
    label_file: str,
    window_seconds: int,
    batch_rows: int,
    resume: bool,
    stop_file: Path | None,
) -> tuple[
    list[DateJob],
    dict[str, dict[str, dict[str, Any]]],
    dict[str, str],
    dict[str, list[str]],
]:
    extract_manifest = load_extract_manifest(metadata_root)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for source_key, entry in extract_manifest["captures"].items():
        if not isinstance(entry, dict) or entry.get("status") != "complete":
            raise PipelineError(
                f"incomplete capture entry in extraction manifest: {source_key}"
            )
        dataset = entry.get("dataset")
        capture_id = entry.get("capture_date")
        if dataset not in datasets:
            continue
        if not isinstance(capture_id, str) or not capture_id:
            raise PipelineError(f"capture entry has an invalid date: {source_key}")
        if selected_dates is not None and capture_id not in selected_dates:
            continue
        grouped.setdefault((dataset, capture_id), []).append(entry)

    labels_by_dataset: dict[str, dict[str, dict[str, Any]]] = {}
    label_hashes: dict[str, str] = {}
    captures_by_dataset: dict[str, list[str]] = {}
    jobs: list[DateJob] = []
    for dataset in datasets:
        labels, label_hash = load_labels(labels_root, dataset, label_file)
        labels_by_dataset[dataset] = labels
        label_hashes[dataset] = label_hash
        available_dates = sorted(
            capture_id
            for candidate_dataset, capture_id in grouped
            if candidate_dataset == dataset
        )
        if selected_dates is not None:
            missing = selected_dates - set(available_dates)
            if missing:
                raise PipelineError(
                    f"{dataset}: requested metadata dates not found: "
                    f"{', '.join(sorted(missing))}"
                )
        if not available_dates:
            raise PipelineError(f"{dataset}: no metadata dates selected")
        captures_by_dataset[dataset] = available_dates

        for capture_id in available_dates:
            entries = sorted(
                grouped[(dataset, capture_id)], key=lambda item: item["output"]
            )
            files: list[str] = []
            inventory: list[dict[str, Any]] = []
            starts: list[int] = []
            for entry in entries:
                relative = entry.get("output")
                path = _safe_metadata_path(metadata_root, relative)
                if not path.is_file():
                    raise PipelineError(f"missing packet metadata file: {path}")
                stat = path.stat()
                expected_size = entry.get("output_size")
                if type(expected_size) is int and stat.st_size != expected_size:
                    raise PipelineError(
                        f"packet metadata size differs from manifest for {path}: "
                        f"expected {expected_size}, got {stat.st_size}"
                    )
                row_count = entry.get("row_count")
                if type(row_count) is not int or row_count < 0:
                    raise PipelineError(
                        f"invalid row count in extraction manifest for {path}"
                    )
                first_timestamp_ns = entry.get("first_timestamp_ns")
                if first_timestamp_ns is not None:
                    if type(first_timestamp_ns) is not int:
                        raise PipelineError(
                            f"invalid first timestamp in extraction manifest for {path}"
                        )
                    starts.append(first_timestamp_ns)
                files.append(str(path))
                inventory.append(
                    {
                        "path": str(relative),
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        "row_count": row_count,
                    }
                )
            if not starts:
                raise PipelineError(
                    f"{dataset}/{capture_id}: all metadata files are empty"
                )
            jobs.append(
                DateJob(
                    dataset=dataset,
                    capture_id=capture_id,
                    labels=labels,
                    label_sha256=label_hash,
                    files=tuple(files),
                    source_inventory=tuple(inventory),
                    capture_start_ns=min(starts),
                    window_seconds=window_seconds,
                    batch_rows=batch_rows,
                    stage_root=str((window_root / ".staging").resolve()),
                    resume=resume,
                    stop_file=str(stop_file) if stop_file is not None else None,
                )
            )
    jobs.sort(key=lambda item: (DATASETS.index(item.dataset), item.capture_id))
    return jobs, labels_by_dataset, label_hashes, captures_by_dataset


def _validate_packet_file(path: Path) -> pq.ParquetFile:
    try:
        parquet = pq.ParquetFile(path)
    except (OSError, pa.ArrowException) as exc:
        raise PipelineError(f"cannot read packet metadata {path}: {exc}") from exc
    missing = set(PACKET_COLUMNS) - set(parquet.schema_arrow.names)
    if missing:
        raise PipelineError(
            f"packet metadata {path} is missing columns: {', '.join(sorted(missing))}"
        )
    schema = parquet.schema_arrow
    if not pa.types.is_timestamp(schema.field("timestamp_utc").type):
        raise PipelineError(f"packet metadata {path}: timestamp_utc is not a timestamp")
    if not pa.types.is_integer(schema.field("frame_len").type):
        raise PipelineError(f"packet metadata {path}: frame_len is not an integer")
    for field_name in ("eth_src", "eth_dst"):
        if not pa.types.is_string(schema.field(field_name).type):
            raise PipelineError(f"packet metadata {path}: {field_name} is not a string")
    return parquet


def _merge_grouped_direction(
    aggregates: dict[tuple[str, int], WindowStats],
    timestamps: pa.Array,
    lengths: pa.Array,
    addresses: pa.Array,
    device_ids: pa.Array,
    capture_start_ns: int,
    window_ns: int,
    outgoing: bool,
) -> None:
    normalized = pc.utf8_lower(addresses)
    mask = pc.is_in(normalized, value_set=device_ids)
    if not pc.any(mask).as_py():
        return
    selected_timestamps = pc.filter(timestamps, mask)
    selected_lengths = pc.filter(lengths, mask)
    selected_devices = pc.filter(normalized, mask)
    timestamp_ns = pc.cast(selected_timestamps, pa.int64())
    preceding = pc.less(timestamp_ns, pa.scalar(capture_start_ns, pa.int64()))
    if pc.any(preceding).as_py():
        raise PipelineError("packet timestamp precedes the capture-date start")
    windows = pc.divide(
        pc.subtract(timestamp_ns, pa.scalar(capture_start_ns, pa.int64())),
        pa.scalar(window_ns, pa.int64()),
    )
    if pc.any(pc.greater(windows, pa.scalar(2**31 - 1, pa.int64()))).as_py():
        raise PipelineError("window index exceeds the int32 output range")
    grouped = (
        pa.table(
            {
                "device_id": selected_devices,
                "window": windows,
                "frame_len": selected_lengths,
                "timestamp_ns": timestamp_ns,
            }
        )
        .group_by(["device_id", "window"])
        .aggregate(
            [
                ("frame_len", "count"),
                ("frame_len", "sum"),
                ("timestamp_ns", "min"),
                ("timestamp_ns", "max"),
            ]
        )
    )
    values = grouped.to_pydict()
    for index, device_id in enumerate(values["device_id"]):
        window = int(values["window"][index])
        stats = aggregates.setdefault((device_id, window), WindowStats())
        direction = stats.outgoing if outgoing else stats.incoming
        direction.merge(
            int(values["frame_len_count"][index]),
            int(values["frame_len_sum"][index]),
            int(values["timestamp_ns_min"][index]),
            int(values["timestamp_ns_max"][index]),
        )


def update_aggregates_from_batch(
    aggregates: dict[tuple[str, int], WindowStats],
    batch: pa.RecordBatch,
    device_ids: pa.Array,
    capture_start_ns: int,
    window_seconds: int,
) -> None:
    if window_seconds <= 0:
        raise ValueError("window_seconds must be greater than zero")
    timestamps = batch.column(batch.schema.get_field_index("timestamp_utc"))
    lengths = batch.column(batch.schema.get_field_index("frame_len"))
    if timestamps.null_count or lengths.null_count:
        raise PipelineError("packet timestamps and frame lengths must not be null")
    if pc.any(pc.less(lengths, pa.scalar(0, lengths.type))).as_py():
        raise PipelineError("packet frame lengths must not be negative")
    window_ns = window_seconds * NANOSECONDS_PER_SECOND
    _merge_grouped_direction(
        aggregates,
        timestamps,
        lengths,
        batch.column(batch.schema.get_field_index("eth_src")),
        device_ids,
        capture_start_ns,
        window_ns,
        outgoing=True,
    )
    _merge_grouped_direction(
        aggregates,
        timestamps,
        lengths,
        batch.column(batch.schema.get_field_index("eth_dst")),
        device_ids,
        capture_start_ns,
        window_ns,
        outgoing=False,
    )


def rows_from_aggregates(
    aggregates: Mapping[tuple[str, int], WindowStats],
    labels: Mapping[str, Mapping[str, Any]],
    dataset: str,
    capture_id: str,
    capture_start_ns: int,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {device_id: [] for device_id in labels}
    windows_by_device: dict[str, list[tuple[int, WindowStats]]] = {
        device_id: [] for device_id in labels
    }
    for (device_id, window), stats in aggregates.items():
        windows_by_device[device_id].append((window, stats))

    for device_id, windows in windows_by_device.items():
        previous_last_ns = capture_start_ns
        metadata = labels[device_id]
        for window, stats in sorted(windows, key=lambda item: item[0]):
            first_candidates = [
                value
                for value in (
                    stats.outgoing.first_timestamp_ns,
                    stats.incoming.first_timestamp_ns,
                )
                if value is not None
            ]
            last_candidates = [
                value
                for value in (
                    stats.outgoing.last_timestamp_ns,
                    stats.incoming.last_timestamp_ns,
                )
                if value is not None
            ]
            first_active_ns = min(first_candidates)
            last_active_ns = max(last_candidates)
            idle_time = (first_active_ns - previous_last_ns) / NANOSECONDS_PER_SECOND
            if idle_time < 0 or not math.isfinite(idle_time):
                raise PipelineError(
                    f"{dataset}/{capture_id}/{device_id}: invalid idle time"
                )
            outgoing_count = stats.outgoing.packet_count
            outgoing_bytes = stats.outgoing.total_bytes
            incoming_count = stats.incoming.packet_count
            incoming_bytes = stats.incoming.total_bytes
            result[device_id].append(
                {
                    "dataset": dataset,
                    "capture_id": capture_id,
                    "window": window,
                    "device_id": device_id,
                    "device_id_type": "mac",
                    "device_name": metadata["device_name"],
                    "is_iot_device": metadata["is_iot_device"],
                    "device_type": metadata.get("device_type"),
                    "vendor": metadata.get("vendor"),
                    "outgoing_packet_count": outgoing_count,
                    "outgoing_total_bytes": outgoing_bytes,
                    "incoming_packet_count": incoming_count,
                    "incoming_total_bytes": incoming_bytes,
                    "aggregated_packet_count": outgoing_count + incoming_count,
                    "aggregated_total_bytes": outgoing_bytes + incoming_bytes,
                    "idle_time": float(idle_time),
                }
            )
            previous_last_ns = last_active_ns
    return result


def table_from_rows(rows: Sequence[Mapping[str, Any]]) -> pa.Table:
    return pa.Table.from_pylist(list(rows), schema=FEATURE_SCHEMA)


def date_stage_dir(stage_root: Path, dataset: str, capture_id: str) -> Path:
    return stage_root / dataset / capture_id


def _stage_fingerprint(job: DateJob) -> dict[str, Any]:
    return {
        "algorithm_version": ALGORITHM_VERSION,
        "dataset": job.dataset,
        "capture_id": job.capture_id,
        "identifier_type": "mac",
        "window_seconds": job.window_seconds,
        "label_sha256": job.label_sha256,
        "capture_start_ns": job.capture_start_ns,
        "source_inventory": list(job.source_inventory),
    }


def valid_completed_stage(job: DateJob) -> dict[str, Any] | None:
    directory = date_stage_dir(Path(job.stage_root), job.dataset, job.capture_id)
    manifest_path = directory / "_complete.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if any(
        manifest.get(key) != value for key, value in _stage_fingerprint(job).items()
    ):
        return None
    row_counts = manifest.get("row_counts")
    if not isinstance(row_counts, dict) or set(row_counts) != set(job.labels):
        return None
    for device_id, expected_rows in row_counts.items():
        path = directory / f"{device_id}.parquet"
        try:
            parquet = pq.ParquetFile(path)
        except (OSError, pa.ArrowException):
            return None
        if parquet.metadata.num_rows != expected_rows:
            return None
        if not parquet.schema_arrow.equals(FEATURE_SCHEMA, check_metadata=True):
            return None
    return manifest


def _replace_directory(temporary: Path, target: Path) -> None:
    backup: Path | None = None
    if target.exists():
        backup = target.parent / f".{target.name}.{uuid.uuid4().hex}.old"
        os.replace(target, backup)
    try:
        os.replace(temporary, target)
    except BaseException:
        if backup is not None and backup.exists() and not target.exists():
            os.replace(backup, target)
        raise
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)


def process_date(job: DateJob) -> dict[str, Any]:
    if job.resume and (existing := valid_completed_stage(job)) is not None:
        return {**existing, "status": "complete", "resumed": True}
    if stop_requested(job.stop_file):
        return {"status": "stopped"}

    device_ids = pa.array(sorted(job.labels), type=pa.string())
    aggregates: dict[tuple[str, int], WindowStats] = {}
    for raw_path in job.files:
        if stop_requested(job.stop_file):
            return {"status": "stopped"}
        path = Path(raw_path)
        parquet = _validate_packet_file(path)
        try:
            for batch in parquet.iter_batches(
                batch_size=job.batch_rows,
                columns=list(PACKET_COLUMNS),
                use_threads=True,
            ):
                if stop_requested(job.stop_file):
                    return {"status": "stopped"}
                update_aggregates_from_batch(
                    aggregates,
                    batch,
                    device_ids,
                    job.capture_start_ns,
                    job.window_seconds,
                )
        except (OSError, pa.ArrowException) as exc:
            raise PipelineError(f"cannot scan packet metadata {path}: {exc}") from exc

    rows_by_device = rows_from_aggregates(
        aggregates,
        job.labels,
        job.dataset,
        job.capture_id,
        job.capture_start_ns,
    )
    target = date_stage_dir(Path(job.stage_root), job.dataset, job.capture_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{job.capture_id}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        row_counts: dict[str, int] = {}
        for device_id, rows in rows_by_device.items():
            if stop_requested(job.stop_file):
                shutil.rmtree(temporary, ignore_errors=True)
                return {"status": "stopped"}
            row_counts[device_id] = len(rows)
            pq.write_table(
                table_from_rows(rows),
                temporary / f"{device_id}.parquet",
                compression="zstd",
            )
        manifest = {
            **_stage_fingerprint(job),
            "row_counts": row_counts,
            "completed_at": utc_now(),
        }
        atomic_write_json(temporary / "_complete.json", manifest)
        if stop_requested(job.stop_file):
            shutil.rmtree(temporary, ignore_errors=True)
            return {"status": "stopped"}
        _replace_directory(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {**manifest, "status": "complete", "resumed": False}


def validate_device_output(
    path: Path,
    dataset: str,
    device_id: str,
    metadata: Mapping[str, Any],
    expected_rows: int,
) -> None:
    table = pq.read_table(path)
    if not table.schema.equals(FEATURE_SCHEMA, check_metadata=True):
        raise PipelineError(f"validation failed for {path}: unexpected schema")
    if table.num_rows != expected_rows:
        raise PipelineError(
            f"validation failed for {path}: expected {expected_rows} rows, "
            f"got {table.num_rows}"
        )
    if not table.num_rows:
        return
    values = table.to_pydict()
    constants = {
        "dataset": dataset,
        "device_id": device_id,
        "device_id_type": "mac",
        "device_name": metadata["device_name"],
        "is_iot_device": metadata["is_iot_device"],
        "device_type": metadata.get("device_type"),
        "vendor": metadata.get("vendor"),
    }
    for column, expected in constants.items():
        if any(value != expected for value in values[column]):
            raise PipelineError(f"validation failed for {path}: inconsistent {column}")

    numeric_columns = (
        "outgoing_packet_count",
        "outgoing_total_bytes",
        "incoming_packet_count",
        "incoming_total_bytes",
        "aggregated_packet_count",
        "aggregated_total_bytes",
        "idle_time",
    )
    for column in numeric_columns:
        if pc.any(pc.less(table[column], 0)).as_py():
            raise PipelineError(f"validation failed for {path}: negative {column}")
    if pc.any(pc.invert(pc.is_finite(table["idle_time"]))).as_py():
        raise PipelineError(f"validation failed for {path}: non-finite idle_time")

    seen: set[tuple[str, int]] = set()
    previous: tuple[str, int] | None = None
    for index, key in enumerate(zip(values["capture_id"], values["window"])):
        if key in seen or (previous is not None and key < previous):
            raise PipelineError(
                f"validation failed for {path}: rows are duplicated or unsorted"
            )
        seen.add(key)
        previous = key
        if values["aggregated_packet_count"][index] != (
            values["outgoing_packet_count"][index]
            + values["incoming_packet_count"][index]
        ):
            raise PipelineError(f"validation failed for {path}: packet total mismatch")
        if values["aggregated_total_bytes"][index] != (
            values["outgoing_total_bytes"][index]
            + values["incoming_total_bytes"][index]
        ):
            raise PipelineError(f"validation failed for {path}: byte total mismatch")


def consolidate_device(job: ConsolidationJob) -> dict[str, Any]:
    if stop_requested(job.stop_file):
        return {"status": "stopped"}
    window_root = Path(job.window_root)
    output_dir = window_root / job.dataset / job.device_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "features.parquet"
    temporary = output_dir / f".features.{uuid.uuid4().hex}.tmp.parquet"
    writer = pq.ParquetWriter(temporary, FEATURE_SCHEMA, compression="zstd")
    total_rows = 0
    stopped = False
    try:
        for capture_id in job.capture_ids:
            if stop_requested(job.stop_file):
                stopped = True
                break
            fragment = (
                window_root
                / ".staging"
                / job.dataset
                / capture_id
                / f"{job.device_id}.parquet"
            )
            table = pq.read_table(fragment)
            if not table.schema.equals(FEATURE_SCHEMA, check_metadata=True):
                raise PipelineError(f"staged feature schema mismatch: {fragment}")
            if table.num_rows:
                writer.write_table(table)
                total_rows += table.num_rows
        writer.close()
        if stopped or stop_requested(job.stop_file):
            temporary.unlink(missing_ok=True)
            return {"status": "stopped"}
        if total_rows != job.expected_rows:
            raise PipelineError(
                f"{job.dataset}/{job.device_id}: expected {job.expected_rows} staged "
                f"rows, got {total_rows}"
            )
        os.replace(temporary, output_path)
    except BaseException:
        writer.close()
        temporary.unlink(missing_ok=True)
        raise
    validate_device_output(
        output_path,
        job.dataset,
        job.device_id,
        job.metadata,
        job.expected_rows,
    )
    return {
        "status": "complete",
        "dataset": job.dataset,
        "device_id": job.device_id,
        "row_count": total_rows,
        "output": str(Path(job.dataset) / job.device_id / "features.parquet"),
        "algorithm_version": ALGORITHM_VERSION,
        "label_sha256": job.label_sha256,
        "completed_at": utc_now(),
    }


def valid_completed_device(
    window_root: Path,
    entry: Mapping[str, Any] | None,
    dataset: str,
    device_id: str,
    metadata: Mapping[str, Any],
    expected_rows: int,
    label_sha256: str,
) -> bool:
    if not entry or entry.get("status") != "complete":
        return False
    expected = {
        "dataset": dataset,
        "device_id": device_id,
        "row_count": expected_rows,
        "algorithm_version": ALGORITHM_VERSION,
        "label_sha256": label_sha256,
    }
    if any(entry.get(key) != value for key, value in expected.items()):
        return False
    path = window_root / dataset / device_id / "features.parquet"
    try:
        validate_device_output(path, dataset, device_id, metadata, expected_rows)
    except (OSError, pa.ArrowException, PipelineError):
        return False
    return True


def prepare_output_root(window_root: Path, resume: bool, overwrite: bool) -> None:
    if window_root.exists() and overwrite:
        shutil.rmtree(window_root)
    elif window_root.exists() and not resume:
        raise PipelineError(
            f"output already exists: {window_root}; pass --resume or --overwrite"
        )
    window_root.mkdir(parents=True, exist_ok=True)


def _run_bounded_jobs(
    jobs: Sequence[Any],
    worker: Callable[[Any], dict[str, Any]],
    workers: int,
    stop_file: Path | None,
    on_complete: Callable[[Any, dict[str, Any], int, int], None],
) -> bool:
    iterator = iter(jobs)
    in_flight: dict[concurrent.futures.Future[dict[str, Any]], Any] = {}
    stopped = False
    completed_count = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        while len(in_flight) < workers and not stop_requested(stop_file):
            try:
                job = next(iterator)
            except StopIteration:
                break
            in_flight[executor.submit(worker, job)] = job
        if stop_requested(stop_file):
            stopped = True

        while in_flight:
            done, _ = concurrent.futures.wait(
                in_flight, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                job = in_flight.pop(future)
                result = future.result()
                if result.get("status") == "stopped":
                    stopped = True
                    continue
                completed_count += 1
                on_complete(job, result, completed_count, len(jobs))
            if stop_requested(stop_file):
                stopped = True
            while len(in_flight) < workers and not stopped:
                try:
                    job = next(iterator)
                except StopIteration:
                    break
                in_flight[executor.submit(worker, job)] = job
    return stopped


def _resume_configuration(manifest: Mapping[str, Any]) -> dict[str, Any]:
    configuration = manifest.get("configuration", {})
    keys = (
        "metadata_root",
        "labels_root",
        "label_file",
        "output_root",
        "window_seconds",
        "datasets",
        "dates",
        "label_sha256",
    )
    return {key: configuration.get(key) for key in keys}


def _mark_stopped(manifest: dict[str, Any], manifest_path: Path) -> None:
    manifest["status"] = "stopped"
    manifest["stopped_at"] = utc_now()
    manifest["updated_at"] = manifest["stopped_at"]
    atomic_write_json(manifest_path, manifest)


def run_pipeline(args: argparse.Namespace) -> Path | None:
    if args.window_seconds <= 0:
        raise PipelineError("--window-seconds must be greater than zero")
    if args.workers <= 0:
        raise PipelineError("--workers must be greater than zero")
    if args.batch_rows <= 0:
        raise PipelineError("--batch-rows must be greater than zero")
    if len(args.datasets) != len(set(args.datasets)):
        raise PipelineError("--datasets must not contain duplicates")
    if args.dates and len(args.dates) != len(set(args.dates)):
        raise PipelineError("--dates must not contain duplicates")

    metadata_root = args.metadata_root.resolve()
    labels_root = args.labels_root.resolve()
    output_root = args.output_root.resolve()
    stop_file = args.stop_file.resolve() if args.stop_file is not None else None
    label_file = validate_label_file_name(args.label_file)
    window_root = output_root / f"window_{args.window_seconds}s"
    if stop_requested(stop_file):
        print(
            f"Stop requested before startup; output was not changed: {stop_file}",
            flush=True,
        )
        return None

    prepare_output_root(window_root, args.resume, args.overwrite)
    jobs, labels_by_dataset, label_hashes, captures_by_dataset = create_jobs(
        metadata_root,
        labels_root,
        window_root,
        args.datasets,
        set(args.dates) if args.dates else None,
        label_file,
        args.window_seconds,
        args.batch_rows,
        args.resume,
        stop_file,
    )

    manifest_path = window_root / "_run_manifest.json"
    previous: dict[str, Any] = {}
    if args.resume and manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise PipelineError(
                f"invalid existing run manifest: {manifest_path}"
            ) from exc

    configuration = {
        "metadata_root": str(metadata_root),
        "labels_root": str(labels_root),
        "label_file": label_file,
        "output_root": str(output_root),
        "window_seconds": args.window_seconds,
        "workers": args.workers,
        "batch_rows": args.batch_rows,
        "datasets": list(args.datasets),
        "dates": list(args.dates) if args.dates else None,
        "stop_file": str(stop_file) if stop_file is not None else None,
        "label_sha256": label_hashes,
    }
    candidate = {"configuration": configuration}
    if previous and previous.get("algorithm_version") != ALGORITHM_VERSION:
        raise PipelineError(
            "existing run manifest uses an incompatible feature algorithm version"
        )
    if previous and _resume_configuration(previous) != _resume_configuration(candidate):
        raise PipelineError(
            "existing run manifest is incompatible with this --resume configuration"
        )

    previous_datasets = previous.get("datasets", {}) if previous else {}
    manifest: dict[str, Any] = {
        "status": "running",
        "algorithm_version": ALGORITHM_VERSION,
        "started_at": previous.get("started_at", utc_now()),
        "updated_at": utc_now(),
        "completed_at": None,
        "configuration": configuration,
        "datasets": {},
    }
    for dataset in args.datasets:
        old_dataset = previous_datasets.get(dataset, {})
        manifest["datasets"][dataset] = {
            "identifier_type": "mac",
            "device_count": len(labels_by_dataset[dataset]),
            "label_sha256": label_hashes[dataset],
            "dates": dict(old_dataset.get("dates", {})),
            "devices": dict(old_dataset.get("devices", {})),
        }
    atomic_write_json(manifest_path, manifest)

    print(
        f"Processing {len(jobs)} date partitions with {args.workers} worker(s); "
        f"output: {window_root}",
        flush=True,
    )
    date_results: dict[tuple[str, str], dict[str, Any]] = {}

    def date_complete(
        job: DateJob, result: dict[str, Any], index: int, total: int
    ) -> None:
        date_results[(job.dataset, job.capture_id)] = result
        manifest["datasets"][job.dataset]["dates"][job.capture_id] = {
            "status": "complete",
            "file_count": len(job.files),
            "total_input_rows": sum(item["row_count"] for item in job.source_inventory),
            "total_input_bytes": sum(item["size"] for item in job.source_inventory),
            "source_inventory": list(job.source_inventory),
            "capture_start_ns": result["capture_start_ns"],
            "row_counts": result["row_counts"],
            "resumed": result["resumed"],
        }
        manifest["updated_at"] = utc_now()
        atomic_write_json(manifest_path, manifest)
        mode = "resumed" if result["resumed"] else "processed"
        rows = sum(result["row_counts"].values())
        print(
            f"[{index}/{total}] {mode} {job.dataset}/{job.capture_id}: {rows:,} rows",
            flush=True,
        )

    try:
        stopped = _run_bounded_jobs(
            jobs, process_date, args.workers, stop_file, date_complete
        )
        if stopped:
            _mark_stopped(manifest, manifest_path)
            print(
                "Stopped gracefully; rerun with --resume after removing the stop file."
            )
            return window_root

        for job in jobs:
            key = (job.dataset, job.capture_id)
            if key not in date_results:
                existing = valid_completed_stage(job)
                if existing is None:
                    raise PipelineError(
                        f"missing completed stage for {job.dataset}/{job.capture_id}"
                    )
                date_results[key] = {**existing, "status": "complete", "resumed": True}

        consolidation_jobs: list[ConsolidationJob] = []
        for dataset in args.datasets:
            capture_ids = captures_by_dataset[dataset]
            for device_id, metadata in labels_by_dataset[dataset].items():
                expected_rows = sum(
                    date_results[(dataset, capture_id)]["row_counts"][device_id]
                    for capture_id in capture_ids
                )
                existing_entry = manifest["datasets"][dataset]["devices"].get(device_id)
                if args.resume and valid_completed_device(
                    window_root,
                    existing_entry,
                    dataset,
                    device_id,
                    metadata,
                    expected_rows,
                    label_hashes[dataset],
                ):
                    continue
                consolidation_jobs.append(
                    ConsolidationJob(
                        window_root=str(window_root),
                        dataset=dataset,
                        capture_ids=tuple(capture_ids),
                        device_id=device_id,
                        metadata=metadata,
                        expected_rows=expected_rows,
                        label_sha256=label_hashes[dataset],
                        stop_file=str(stop_file) if stop_file is not None else None,
                    )
                )

        print(
            f"Consolidating and validating {len(consolidation_jobs)} device output(s)...",
            flush=True,
        )

        def device_complete(
            job: ConsolidationJob,
            result: dict[str, Any],
            index: int,
            total: int,
        ) -> None:
            manifest["datasets"][job.dataset]["devices"][job.device_id] = result
            manifest["updated_at"] = utc_now()
            atomic_write_json(manifest_path, manifest)
            print(
                f"[{index}/{total}] {job.dataset}/{job.device_id}: "
                f"{result['row_count']:,} rows",
                flush=True,
            )

        stopped = _run_bounded_jobs(
            consolidation_jobs,
            consolidate_device,
            args.workers,
            stop_file,
            device_complete,
        )
        if stopped:
            _mark_stopped(manifest, manifest_path)
            print(
                "Stopped gracefully; rerun with --resume after removing the stop file."
            )
            return window_root

        total_rows = 0
        for dataset in args.datasets:
            for device_id, metadata in labels_by_dataset[dataset].items():
                entry = manifest["datasets"][dataset]["devices"].get(device_id)
                expected_rows = sum(
                    date_results[(dataset, capture_id)]["row_counts"][device_id]
                    for capture_id in captures_by_dataset[dataset]
                )
                if not valid_completed_device(
                    window_root,
                    entry,
                    dataset,
                    device_id,
                    metadata,
                    expected_rows,
                    label_hashes[dataset],
                ):
                    raise PipelineError(
                        f"final output validation failed for {dataset}/{device_id}"
                    )
                total_rows += expected_rows

        manifest["status"] = "complete"
        manifest["completed_at"] = utc_now()
        manifest["updated_at"] = manifest["completed_at"]
        manifest["total_rows"] = total_rows
        atomic_write_json(manifest_path, manifest)
        print(f"Complete: {total_rows:,} rows written and validated.", flush=True)
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["failed_at"] = utc_now()
        manifest["updated_at"] = manifest["failed_at"]
        manifest["error"] = str(exc)
        atomic_write_json(manifest_path, manifest)
        raise
    return window_root


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  # Binary labels, with cooperative stop-file support
  python code/build_features.py \\
    --output-root built_features_mac/binary-class \\
    --stop-file /tmp/build_features_binary.stop

  # Resume after removing a requested stop
  rm /tmp/build_features_binary.stop
  python code/build_features.py \\
    --output-root built_features_mac/binary-class \\
    --stop-file /tmp/build_features_binary.stop --resume

  # Multiclass labels use a separate output root
  python code/build_features.py \\
    --label-file mac_mapping_multi_labels.json \\
    --output-root built_features_mac/multiclass
""",
    )
    parser.add_argument(
        "--metadata-root", type=Path, default=Path("extracted_metadata_dataset")
    )
    parser.add_argument("--labels-root", type=Path, default=Path("labels"))
    parser.add_argument("--label-file", default=DEFAULT_LABEL_FILE)
    parser.add_argument("--output-root", type=Path, default=Path("built_features_mac"))
    parser.add_argument("--window-seconds", type=int, default=60)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="number of date/device jobs to process concurrently",
    )
    parser.add_argument(
        "--batch-rows",
        type=int,
        default=100_000,
        help="maximum packet rows per in-memory Arrow batch",
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(DATASETS)
    )
    parser.add_argument(
        "--dates",
        nargs="+",
        help="optional capture-date partition names (best used with one dataset)",
    )
    parser.add_argument(
        "--stop-file",
        type=Path,
        help="optional externally managed file whose presence requests a graceful stop",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run_pipeline(parse_args(argv))
    except (PipelineError, OSError, pa.ArrowException) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
