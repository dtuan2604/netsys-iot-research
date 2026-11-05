#!/usr/bin/env python3
"""Build ordered per-packet Layer-2 features for a future LSTM trainer.

The builder follows the MAC-addressed v2 dataset organization. It groups source
Parquet files by dataset and capture date, attributes each observed frame to its
labeled Ethernet endpoints, and writes one ordered packet stream per device.
Sequence lengths are deliberately not baked into this artifact. Within each
``capture_id``, a future loader must order by ``packet_index`` before forming
20-, 40-, 100-, or other packet-count sequences.

Five per-packet feature groups produce seven numeric model columns:
  1. frame_size                         (Ethernet frame_len)
  2. direction                          (1=outgoing, 0=incoming)
  3. inter_arrival_time                 (seconds since previous device packet)
  4. destination type                   (3 one-hot columns)
  5. same_direction_inter_arrival_time  (seconds since previous same direction)

Input Parquet files must contain timestamp_utc, frame_len, eth_src, and eth_dst.
Only these Layer-2 fields are read. Incoming multicast/broadcast frames cannot
be assigned to individual receivers from eth_dst, so they are not copied to all
possible devices.

Memory-bounded scanning uses Arrow batches and a disk-backed SQLite sort per
capture date. Sorting uses timestamp and deterministic insertion order so equal
timestamps still receive stable packet indices.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - exercised by the CLI environment
    raise SystemExit(
        "pyarrow is required. Install code_v2/requirements-feature-construction.txt "
        "with the Python interpreter used to run this script."
    ) from exc

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPOSITORY_ROOT.parent
DEFAULT_METADATA_ROOT = WORKSPACE_ROOT / "extracted_metadata_dataset"
DEFAULT_LABELS_ROOT = REPOSITORY_ROOT / "labels"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "built_features_mac_v2" / "binary-class"

DATASETS = ("NCSU", "UNSW1", "UNSW2", "YourThing")
DEFAULT_LABEL_FILE = "mac_mapping_binary_labels.json"
EXTRACT_SCHEMA_VERSION = 1
ALGORITHM_VERSION = 1
MAC_RE = re.compile(r"^(?:[0-9a-f]{2}:){5}[0-9a-f]{2}$")
REQUIRED_PACKET_COLUMNS = ("timestamp_utc", "frame_len", "eth_src", "eth_dst")
NS_PER_SECOND = 1_000_000_000

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

PACKET_SCHEMA = pa.schema(
    [
        pa.field("dataset", pa.string(), nullable=False),
        pa.field("capture_id", pa.string(), nullable=False),
        pa.field("device_id", pa.string(), nullable=False),
        pa.field("device_id_type", pa.string(), nullable=False),
        pa.field("device_name", pa.string(), nullable=False),
        pa.field("is_iot_device", pa.int32(), nullable=False),
        pa.field("device_type", pa.string(), nullable=True),
        pa.field("vendor", pa.string(), nullable=True),
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


class PipelineError(RuntimeError):
    pass


class StopRequested(RuntimeError):
    pass


@dataclass(frozen=True)
class CaptureJob:
    dataset: str
    capture_id: str
    files: tuple[str, ...]
    inventory: tuple[dict[str, Any], ...]
    labels: dict[str, dict[str, Any]]
    label_sha256: str
    stage_root: str
    batch_rows: int
    write_rows: int
    resume: bool
    stop_file: str | None


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def check_stop(path: Path | str | None) -> None:
    if path is not None and Path(path).is_file():
        raise StopRequested("Stop file detected")


def safe_label_name(value: str) -> str:
    if (
        not value
        or value in (".", "..")
        or Path(value).is_absolute()
        or "/" in value
        or "\\" in value
    ):
        raise PipelineError("--label-file must be a filename, not a path")
    return value


def load_labels(
    root: Path, dataset: str, label_file: str
) -> tuple[dict[str, dict[str, Any]], str]:
    path = root / dataset / safe_label_name(label_file)
    raw_bytes = path.read_bytes()
    raw = json.loads(raw_bytes)
    if not isinstance(raw, dict) or not raw:
        raise PipelineError(f"{path}: labels must be a nonempty JSON object")
    result: dict[str, dict[str, Any]] = {}
    for device_id, meta in raw.items():
        if (
            not isinstance(device_id, str)
            or not MAC_RE.fullmatch(device_id)
            or device_id != device_id.lower()
        ):
            raise PipelineError(f"{path}: invalid lowercase MAC key {device_id!r}")
        if int(device_id[:2], 16) & 1:
            raise PipelineError(f"{path}: group MAC cannot identify an individual device: {device_id}")
        if (
            not isinstance(meta, dict)
            or not isinstance(meta.get("device_name"), str)
            or not meta["device_name"]
        ):
            raise PipelineError(f"{path}: {device_id} needs device_name")
        label = meta.get("is_iot_device")
        if type(label) is not int or not 0 <= label < 2**31:
            raise PipelineError(f"{path}: {device_id} has an invalid is_iot_device")
        if any(
            meta.get(key) is not None and not isinstance(meta.get(key), str)
            for key in ("vendor", "device_type")
        ):
            raise PipelineError(f"{path}: {device_id} has invalid vendor/device_type")
        result[device_id] = {
            "device_name": meta["device_name"],
            "is_iot_device": label,
            "device_type": meta.get("device_type"),
            "vendor": meta.get("vendor"),
        }
    return result, hashlib.sha256(raw_bytes).hexdigest()


def safe_source_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise PipelineError("Extraction manifest contains an invalid output path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise PipelineError(f"Source path escapes metadata root: {relative}")
    return path


def create_jobs(args: argparse.Namespace, stage_root: Path, stop_file: Path | None):
    metadata_root = args.metadata_root.resolve()
    labels_root = args.labels_root.resolve()
    path = metadata_root / "_extract_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "complete"
        or manifest.get("schema_version") != EXTRACT_SCHEMA_VERSION
    ):
        raise PipelineError(f"Extraction manifest incomplete or unsupported: {path}")
    captures = manifest.get("captures")
    if not isinstance(captures, dict):
        raise PipelineError(f"Extraction manifest missing captures: {path}")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for key, item in captures.items():
        if not isinstance(item, dict) or item.get("status") != "complete":
            raise PipelineError(f"Incomplete extraction entry: {key}")
        dataset = item.get("dataset")
        capture_id = item.get("capture_date")
        if dataset not in args.datasets:
            continue
        if not isinstance(capture_id, str) or not capture_id:
            raise PipelineError(f"Invalid capture_date in manifest: {key}")
        if args.dates and capture_id not in args.dates:
            continue
        grouped.setdefault((dataset, capture_id), []).append(item)

    jobs = []
    labels_by_dataset = {}
    hashes_by_dataset = {}
    capture_ids_by_dataset = {}
    for dataset in args.datasets:
        labels, label_hash = load_labels(labels_root, dataset, args.label_file)
        labels_by_dataset[dataset] = labels
        hashes_by_dataset[dataset] = label_hash
        dates = sorted(date for ds, date in grouped if ds == dataset)
        if args.dates and set(args.dates) - set(dates):
            missing = sorted(set(args.dates) - set(dates))
            raise PipelineError(
                f"{dataset}: missing selected capture dates: {missing}"
            )
        if not dates:
            raise PipelineError(f"{dataset}: no selected capture dates")
        capture_ids_by_dataset[dataset] = dates
        for capture_id in dates:
            files, inventory = [], []
            for entry in sorted(
                grouped[(dataset, capture_id)], key=lambda item: item["output"]
            ):
                relative = entry.get("output")
                src = safe_source_path(metadata_root, relative)
                if not src.is_file():
                    raise PipelineError(f"Missing source file: {src}")
                stat = src.stat()
                if (
                    type(entry.get("output_size")) is int
                    and stat.st_size != entry["output_size"]
                ):
                    raise PipelineError(f"Source changed size since extraction: {src}")
                if type(entry.get("row_count")) is not int or entry["row_count"] < 0:
                    raise PipelineError(f"Invalid extraction row count: {src}")
                files.append(str(src))
                inventory.append(
                    {
                        "path": relative,
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        "row_count": entry["row_count"],
                    }
                )
            jobs.append(
                CaptureJob(
                    dataset=dataset,
                    capture_id=capture_id,
                    files=tuple(files),
                    inventory=tuple(inventory),
                    labels=labels,
                    label_sha256=label_hash,
                    stage_root=str(stage_root),
                    batch_rows=args.batch_rows,
                    write_rows=args.write_rows,
                    resume=args.resume,
                    stop_file=str(stop_file) if stop_file else None,
                )
            )
    jobs.sort(key=lambda item: (DATASETS.index(item.dataset), item.capture_id))
    return jobs, labels_by_dataset, hashes_by_dataset, capture_ids_by_dataset


def stage_fingerprint(job: CaptureJob) -> str:
    return stable_hash(
        {
            "version": ALGORITHM_VERSION,
            "dataset": job.dataset,
            "capture_id": job.capture_id,
            "labels": job.label_sha256,
            "inventory": job.inventory,
        }
    )


def stage_directory(job: CaptureJob) -> Path:
    return Path(job.stage_root) / job.dataset / job.capture_id


def valid_stage(job: CaptureJob) -> dict[str, Any] | None:
    folder = stage_directory(job)
    path = folder / "_complete.json"
    if not path.is_file():
        return None
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
        if info.get("fingerprint") != stage_fingerprint(job):
            return None
        counts = info["row_counts"]
        if set(counts) != set(job.labels):
            return None
        for mac, count in counts.items():
            source = pq.ParquetFile(folder / f"{mac}.parquet")
            if source.metadata.num_rows != count or not source.schema_arrow.equals(
                PACKET_SCHEMA, check_metadata=True
            ):
                return None
        return info
    except (OSError, ValueError, KeyError, pa.ArrowException):
        return None


def mac_destination_type(value: str) -> int:
    """Return 0=unicast, 1=multicast, 2=broadcast (Ethernet MAC semantics)."""
    if not MAC_RE.fullmatch(value):
        raise PipelineError(f"Invalid Ethernet destination MAC: {value!r}")
    if value == "ff:ff:ff:ff:ff:ff":
        return 2
    return 1 if int(value[:2], 16) & 1 else 0


def parquet_source(path: Path) -> pq.ParquetFile:
    source = pq.ParquetFile(path)
    fields = source.schema_arrow
    missing = set(REQUIRED_PACKET_COLUMNS) - set(fields.names)
    if missing:
        raise PipelineError(f"{path}: missing columns {sorted(missing)}")
    if not pa.types.is_timestamp(fields.field("timestamp_utc").type):
        raise PipelineError(f"{path}: timestamp_utc must have a timestamp type")
    if not pa.types.is_integer(fields.field("frame_len").type):
        raise PipelineError(f"{path}: frame_len must be integer")
    for column in ("eth_src", "eth_dst"):
        if not pa.types.is_string(fields.field(column).type):
            raise PipelineError(f"{path}: {column} must have string type")
    return source


def _scan_source_into_sqlite(
    connection: sqlite3.Connection,
    source_path: Path,
    job: CaptureJob,
) -> int:
    """Insert endpoint-attributed packet observations and return their count."""
    source = parquet_source(source_path)
    n_device_rows = 0
    for batch in source.iter_batches(
        batch_size=job.batch_rows,
        columns=list(REQUIRED_PACKET_COLUMNS),
        use_threads=True,
    ):
        check_stop(job.stop_file)

        def col(name: str) -> pa.Array:
            return batch.column(batch.schema.get_field_index(name))

        ts = col("timestamp_utc")
        sizes = col("frame_len")
        if ts.null_count or sizes.null_count:
            raise PipelineError(f"{source_path}: NULL timestamps or frame lengths")
        ns = pc.cast(
            pc.cast(ts, pa.timestamp("ns", tz=ts.type.tz)), pa.int64()
        ).to_pylist()
        frame_sizes = sizes.to_pylist()
        sources = col("eth_src").to_pylist()
        destinations = col("eth_dst").to_pylist()
        rows = []
        for timestamp_ns, size, raw_src, raw_dst in zip(
            ns, frame_sizes, sources, destinations
        ):
            if size is None or size <= 0:
                raise PipelineError(f"{source_path}: frame_len must be positive")
            src = raw_src.lower() if isinstance(raw_src, str) else None
            dst = raw_dst.lower() if isinstance(raw_dst, str) else None
            outgoing = src in job.labels
            incoming = dst in job.labels and src != dst
            if not outgoing and not incoming:
                continue
            if dst is None:
                raise PipelineError(f"{source_path}: labeled frame has NULL eth_dst")
            dest_type = mac_destination_type(dst)
            if outgoing:
                rows.append((src, timestamp_ns, int(size), 1, dest_type))
                n_device_rows += 1
            if incoming:
                rows.append((dst, timestamp_ns, int(size), 0, dest_type))
                n_device_rows += 1
        connection.executemany(
            "INSERT INTO packets "
            "(device_id, ts_ns, frame_size, direction, dst_type) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        connection.commit()
    return n_device_rows


def make_packet_row(
    job: CaptureJob,
    device_id: str,
    meta: Mapping[str, Any],
    packet_index: int,
    timestamp_ns: int,
    frame_size: int,
    direction: int,
    destination_type: int,
    prev_ns: int | None,
    prev_direction_ns: int | None,
) -> dict[str, Any]:
    iat_ns = 0 if prev_ns is None else timestamp_ns - prev_ns
    directional_iat_ns = (
        0 if prev_direction_ns is None else timestamp_ns - prev_direction_ns
    )
    if iat_ns < 0 or directional_iat_ns < 0:
        raise PipelineError(
            f"{job.dataset}/{job.capture_id}/{device_id}: timestamps out of order"
        )
    return {
        "dataset": job.dataset,
        "capture_id": job.capture_id,
        "device_id": device_id,
        "device_id_type": "mac",
        "device_name": meta["device_name"],
        "is_iot_device": meta["is_iot_device"],
        "device_type": meta["device_type"],
        "vendor": meta["vendor"],
        "packet_index": packet_index,
        "timestamp_ns": timestamp_ns,
        "frame_size": frame_size,
        "direction": direction,
        "inter_arrival_time": iat_ns / NS_PER_SECOND,
        "inter_arrival_valid": prev_ns is not None,
        "dst_unicast": int(destination_type == 0),
        "dst_multicast": int(destination_type == 1),
        "dst_broadcast": int(destination_type == 2),
        "same_direction_inter_arrival_time": directional_iat_ns / NS_PER_SECOND,
        "same_direction_iat_valid": prev_direction_ns is not None,
    }


def export_device(
    connection: sqlite3.Connection,
    job: CaptureJob,
    device_id: str,
    target: Path,
) -> int:
    writer = pq.ParquetWriter(target, PACKET_SCHEMA, compression="zstd")
    count = 0
    output_buffer: list[dict[str, Any]] = []

    def flush() -> None:
        if output_buffer:
            writer.write_table(
                pa.Table.from_pylist(output_buffer, schema=PACKET_SCHEMA)
            )
            output_buffer.clear()

    previous_ns = None
    previous_same_direction: dict[int, int] = {}
    packet_index = 0
    try:
        cursor = connection.execute(
            "SELECT ts_ns, frame_size, direction, dst_type "
            "FROM packets WHERE device_id = ? ORDER BY ts_ns, rowid",
            (device_id,),
        )
        while batch := cursor.fetchmany(job.write_rows):
            check_stop(job.stop_file)
            for ns, size, direction, dst_type in batch:
                output_buffer.append(
                    make_packet_row(
                        job,
                        device_id,
                        job.labels[device_id],
                        packet_index,
                        ns,
                        size,
                        direction,
                        dst_type,
                        previous_ns,
                        previous_same_direction.get(direction),
                    )
                )
                packet_index += 1
                count += 1
                previous_ns = ns
                previous_same_direction[direction] = ns
                if len(output_buffer) >= job.write_rows:
                    flush()
        flush()
    finally:
        writer.close()
    return count


def process_capture(job: CaptureJob) -> dict[str, Any]:
    if job.resume and (cached := valid_stage(job)) is not None:
        return {**cached, "status": "complete", "resumed": True}
    final = stage_directory(job)
    temporary: Path | None = None
    connection = None
    try:
        check_stop(job.stop_file)
        final.parent.mkdir(parents=True, exist_ok=True)
        temporary = final.parent / f".{job.capture_id}.{uuid.uuid4().hex}.tmp"
        temporary.mkdir()
        db = temporary / "sort.sqlite3"
        connection = sqlite3.connect(db)
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute(
            "CREATE TABLE packets ("
            "device_id TEXT NOT NULL, "
            "ts_ns INTEGER NOT NULL, "
            "frame_size INTEGER NOT NULL, "
            "direction INTEGER NOT NULL, "
            "dst_type INTEGER NOT NULL)"
        )
        total_input_device_rows = 0
        for filename in job.files:
            check_stop(job.stop_file)
            total_input_device_rows += _scan_source_into_sqlite(
                connection, Path(filename), job
            )
        check_stop(job.stop_file)
        connection.execute("CREATE INDEX packet_order ON packets(device_id, ts_ns)")
        connection.commit()
        counts = {}
        for device_id in sorted(job.labels):
            check_stop(job.stop_file)
            counts[device_id] = export_device(
                connection, job, device_id, temporary / f"{device_id}.parquet"
            )
        connection.close()
        connection = None
        db.unlink(missing_ok=True)
        info = {
            "status": "complete",
            "fingerprint": stage_fingerprint(job),
            "row_counts": counts,
            "observed_device_packet_rows": total_input_device_rows,
            "completed_at": now_utc(),
        }
        atomic_json(temporary / "_complete.json", info)
        check_stop(job.stop_file)
        # Preserve the previous completed stage until replacement is ready.
        backup = None
        if final.exists():
            backup = final.parent / f".{job.capture_id}.{uuid.uuid4().hex}.old"
            os.replace(final, backup)
        try:
            os.replace(temporary, final)
        except BaseException:
            if backup is not None and not final.exists():
                os.replace(backup, final)
            raise
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)
        return {**info, "resumed": False}
    except StopRequested:
        return {"status": "stopped"}
    finally:
        if connection is not None:
            connection.close()
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def device_sources_fingerprint(
    dataset: str,
    device_id: str,
    dates: Sequence[str],
    completed: Mapping[tuple[str, str], dict],
) -> str:
    return stable_hash(
        {
            "dataset": dataset,
            "device_id": device_id,
            "stages": [
                (
                    date,
                    completed[(dataset, date)]["fingerprint"],
                    completed[(dataset, date)]["row_counts"][device_id],
                )
                for date in dates
            ],
        }
    )


def valid_device(
    path: Path,
    expected: int,
    stored: Mapping[str, Any] | None,
    source_fingerprint: str,
) -> bool:
    if (
        not stored
        or stored.get("status") != "complete"
        or stored.get("row_count") != expected
        or stored.get("sources_fingerprint") != source_fingerprint
    ):
        return False
    try:
        parquet = pq.ParquetFile(path)
        return parquet.metadata.num_rows == expected and parquet.schema_arrow.equals(
            PACKET_SCHEMA, check_metadata=True
        )
    except (OSError, pa.ArrowException):
        return False


def consolidate_device(
    root: Path,
    stage_root: Path,
    dataset: str,
    device_id: str,
    dates: Sequence[str],
    completed: Mapping[tuple[str, str], dict],
    source_fingerprint: str,
    stop_file: Path | None,
    batch_rows: int,
) -> dict[str, Any]:
    output_dir = root / dataset / device_id
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "features.parquet"
    tmp = output_dir / f".features.{uuid.uuid4().hex}.tmp.parquet"
    count = 0
    writer = pq.ParquetWriter(tmp, PACKET_SCHEMA, compression="zstd")
    try:
        for date in dates:
            check_stop(stop_file)
            part = stage_root / dataset / date / f"{device_id}.parquet"
            source = pq.ParquetFile(part)
            if not source.schema_arrow.equals(PACKET_SCHEMA, check_metadata=True):
                raise PipelineError(f"Staged schema mismatch: {part}")
            for batch in source.iter_batches(batch_size=batch_rows):
                check_stop(stop_file)
                writer.write_batch(batch)
                count += batch.num_rows
        writer.close()
        expected = sum(
            completed[(dataset, date)]["row_counts"][device_id] for date in dates
        )
        if count != expected:
            raise PipelineError(
                f"{dataset}/{device_id}: output rows {count} != expected {expected}"
            )
        check_stop(stop_file)
        os.replace(tmp, path)
        check = pq.ParquetFile(path)
        if check.metadata.num_rows != count or not check.schema_arrow.equals(
            PACKET_SCHEMA, check_metadata=True
        ):
            raise PipelineError(f"Output validation failed: {path}")
        return {
            "status": "complete",
            "row_count": count,
            "sources_fingerprint": source_fingerprint,
            "output": str(Path(dataset) / device_id / "features.parquet"),
            "completed_at": now_utc(),
        }
    except BaseException:
        writer.close()
        raise
    finally:
        tmp.unlink(missing_ok=True)


def run_jobs(
    jobs: Sequence[CaptureJob], workers: int, stop_file: Path | None, callback
) -> bool:
    iterator = iter(jobs)
    inflight: dict[concurrent.futures.Future, CaptureJob] = {}
    stopped = False
    done_count = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        while len(inflight) < workers:
            try:
                job = next(iterator)
            except StopIteration:
                break
            inflight[pool.submit(process_capture, job)] = job
        while inflight:
            done, _ = concurrent.futures.wait(
                inflight, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                job = inflight.pop(future)
                result = future.result()
                if result.get("status") == "stopped":
                    stopped = True
                else:
                    done_count += 1
                    callback(job, result, done_count, len(jobs))
            if stop_file and stop_file.is_file():
                stopped = True
            while not stopped and len(inflight) < workers:
                try:
                    job = next(iterator)
                except StopIteration:
                    break
                inflight[pool.submit(process_capture, job)] = job
    return stopped


def run_pipeline(args: argparse.Namespace) -> Path | None:
    if len(args.datasets) != len(set(args.datasets)) or (
        args.dates and len(args.dates) != len(set(args.dates))
    ):
        raise PipelineError("Duplicate datasets/dates are not permitted")
    for name in ("workers", "batch_rows", "write_rows"):
        if getattr(args, name) <= 0:
            raise PipelineError(f"--{name.replace('_', '-')} must be positive")
    if args.resume and args.overwrite:
        raise PipelineError("--resume and --overwrite cannot be combined")

    root = args.output_root.resolve() / "packet_features"
    stage_root = root / ".staging"
    stop_file = args.stop_file.resolve() if args.stop_file else None
    if stop_file and stop_file.is_file():
        print(f"Stop requested before starting; no output changed: {stop_file}")
        return None
    if root.exists():
        if args.overwrite:
            shutil.rmtree(root)
        elif not args.resume:
            raise PipelineError(f"Output exists: {root} (use --resume or --overwrite)")
    root.mkdir(parents=True, exist_ok=True)
    jobs, labels_by_dataset, label_hashes, capture_dates = create_jobs(
        args, stage_root, stop_file
    )
    configuration = {
        "metadata_root": str(args.metadata_root.resolve()),
        "labels_root": str(args.labels_root.resolve()),
        "label_file": args.label_file,
        "label_sha256": label_hashes,
        "output_root": str(args.output_root.resolve()),
        "datasets": list(args.datasets),
        "dates": list(args.dates) if args.dates else None,
        "model_feature_columns": list(MODEL_FEATURE_COLUMNS),
        "auxiliary_feature_columns": list(AUXILIARY_FEATURE_COLUMNS),
    }
    configuration_fingerprint = stable_hash(configuration)
    path = root / "_run_manifest.json"
    previous = (
        json.loads(path.read_text(encoding="utf-8"))
        if args.resume and path.exists()
        else {}
    )
    if (
        previous
        and previous.get("configuration_fingerprint")
        != configuration_fingerprint
    ):
        raise PipelineError(
            "Cannot resume: feature-construction configuration or labels changed"
        )
    old_datasets = previous.get("datasets", {})
    manifest = {
        "status": "running",
        "algorithm_version": ALGORITHM_VERSION,
        "configuration": configuration,
        "configuration_fingerprint": configuration_fingerprint,
        "model_feature_columns": list(MODEL_FEATURE_COLUMNS),
        "auxiliary_feature_columns": list(AUXILIARY_FEATURE_COLUMNS),
        "model_input_width": len(MODEL_FEATURE_COLUMNS),
        "started_at": previous.get("started_at", now_utc()),
        "updated_at": now_utc(),
        "datasets": {},
    }
    for dataset in args.datasets:
        old = old_datasets.get(dataset, {})
        manifest["datasets"][dataset] = {
            "identifier_type": "mac",
            "device_count": len(labels_by_dataset[dataset]),
            "label_sha256": label_hashes[dataset],
            "dates": dict(old.get("dates", {})),
            "devices": dict(old.get("devices", {})),
        }
    atomic_json(path, manifest)
    completed: dict[tuple[str, str], dict[str, Any]] = {}

    def capture_done(
        job: CaptureJob, result: dict[str, Any], n: int, total: int
    ) -> None:
        completed[(job.dataset, job.capture_id)] = result
        manifest["datasets"][job.dataset]["dates"][job.capture_id] = {
            "status": "complete",
            "row_counts": result["row_counts"],
            "observed_device_packet_rows": result["observed_device_packet_rows"],
            "fingerprint": result["fingerprint"],
            "resumed": result["resumed"],
        }
        manifest["updated_at"] = now_utc()
        atomic_json(path, manifest)
        print(
            f"[{n}/{total}] {job.dataset}/{job.capture_id}: "
            f"{sum(result['row_counts'].values()):,} packet rows "
            f"({'cached' if result['resumed'] else 'processed'})",
            flush=True,
        )

    try:
        print(
            f"Processing {len(jobs)} capture-date partitions with "
            f"{args.workers} worker(s).",
            flush=True,
        )
        stopped = run_jobs(jobs, args.workers, stop_file, capture_done)
        if stopped:
            manifest["status"] = "stopped"
            manifest["updated_at"] = now_utc()
            atomic_json(path, manifest)
            print("Stopped; remove the stop file and rerun with --resume")
            return root

        total_rows = 0
        for dataset in args.datasets:
            for device_id in sorted(labels_by_dataset[dataset]):
                check_stop(stop_file)
                dates = capture_dates[dataset]
                expected = sum(
                    completed[(dataset, date)]["row_counts"][device_id]
                    for date in dates
                )
                signature = device_sources_fingerprint(
                    dataset, device_id, dates, completed
                )
                target = root / dataset / device_id / "features.parquet"
                existing = manifest["datasets"][dataset]["devices"].get(device_id)
                if args.resume and valid_device(
                    target, expected, existing, signature
                ):
                    total_rows += expected
                    continue
                record = consolidate_device(
                    root,
                    stage_root,
                    dataset,
                    device_id,
                    dates,
                    completed,
                    signature,
                    stop_file,
                    args.write_rows,
                )
                manifest["datasets"][dataset]["devices"][device_id] = record
                total_rows += record["row_count"]
                manifest["updated_at"] = now_utc()
                atomic_json(path, manifest)
                print(
                    f"  {dataset}/{device_id}: {record['row_count']:,} rows",
                    flush=True,
                )
        manifest["status"] = "complete"
        manifest["total_rows"] = total_rows
        manifest["completed_at"] = now_utc()
        manifest["updated_at"] = manifest["completed_at"]
        atomic_json(path, manifest)
        observed = sum(
            value["observed_device_packet_rows"] for value in completed.values()
        )
        print(
            f"Complete: {total_rows:,} rows from {observed:,} attributed "
            f"packet observations. Output: {root}",
            flush=True,
        )
        return root
    except StopRequested:
        manifest["status"] = "stopped"
        manifest["updated_at"] = now_utc()
        atomic_json(path, manifest)
        print("Stopped; remove the stop file and rerun with --resume")
        return root
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        manifest["updated_at"] = now_utc()
        atomic_json(path, manifest)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  # Binary labels
  python code_v2/build_lstm_packet_features.py \\
    --output-root built_features_mac_v2/binary-class

  # Multiclass labels use a separate output root
  python code_v2/build_lstm_packet_features.py \\
    --label-file mac_mapping_multi_labels.json \\
    --output-root built_features_mac_v2/multiclass
""",
    )
    p.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    p.add_argument("--labels-root", type=Path, default=DEFAULT_LABELS_ROOT)
    p.add_argument("--label-file", default=DEFAULT_LABEL_FILE)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--dates", nargs="+", help="optional capture-date partitions")
    p.add_argument(
        "--batch-rows",
        type=int,
        default=100_000,
        help="max packet metadata rows loaded per Arrow batch",
    )
    p.add_argument(
        "--write-rows",
        type=int,
        default=10_000,
        help="Parquet output buffer size and SQLite fetch size",
    )
    p.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument("--stop-file", type=Path)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run_pipeline(parse_args(argv))
        return 0
    except (
        PipelineError,
        OSError,
        ValueError,
        sqlite3.Error,
        pa.ArrowException,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
