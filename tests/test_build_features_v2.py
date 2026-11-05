from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code_v2 import build_features as bf
from code_v2 import experiment_common as common


MAC_A = "aa:aa:aa:aa:aa:aa"
MAC_B = "bb:bb:bb:bb:bb:bb"
MAC_C = "cc:cc:cc:cc:cc:cc"
PACKET_SCHEMA = pa.schema(
    [
        pa.field("timestamp_utc", pa.timestamp("ns", tz="UTC"), nullable=False),
        pa.field("frame_len", pa.uint32(), nullable=False),
        pa.field("eth_src", pa.string()),
        pa.field("eth_dst", pa.string()),
        pa.field("src_device_id", pa.string()),
        pa.field("dst_device_id", pa.string()),
    ]
)
LABELS = {
    MAC_A: {
        "device_name": "Device A",
        "is_iot_device": 1,
        "device_type": "sensor",
        "vendor": "Example",
    },
    MAC_B: {
        "device_name": "Device B",
        "is_iot_device": 0,
        "device_type": "phone",
        "vendor": None,
    },
}


def ns(seconds: float) -> int:
    return round(seconds * bf.NANOSECONDS_PER_SECOND)


def packet_table(rows: list[tuple[float, int, str | None, str | None]]) -> pa.Table:
    return pa.Table.from_pydict(
        {
            "timestamp_utc": [ns(row[0]) for row in rows],
            "frame_len": [row[1] for row in rows],
            "eth_src": [row[2] for row in rows],
            "eth_dst": [row[3] for row in rows],
            # Deliberately stale mixed-ID fields: the builder must ignore them.
            "src_device_id": ["192.0.2.1"] * len(rows),
            "dst_device_id": ["192.0.2.2"] * len(rows),
        },
        schema=PACKET_SCHEMA,
    )


def write_labels(
    root: Path,
    dataset: str = "NCSU",
    labels: dict = LABELS,
    filename: str = bf.DEFAULT_LABEL_FILE,
) -> Path:
    directory = root / dataset
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(json.dumps(labels), encoding="utf-8")
    return path


def write_metadata(
    root: Path,
    captures: list[
        tuple[str, str, str, list[tuple[float, int, str | None, str | None]]]
    ],
) -> dict:
    entries = {}
    for dataset, capture_date, filename, rows in captures:
        relative = Path(f"dataset={dataset}") / f"date={capture_date}" / filename
        output = root / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        table = packet_table(rows)
        pq.write_table(table, output, compression="zstd")
        entries[
            f"{dataset}/data/{capture_date}/{filename.removesuffix('.parquet')}"
        ] = {
            "dataset": dataset,
            "capture_date": capture_date,
            "output": str(relative),
            "output_size": output.stat().st_size,
            "row_count": table.num_rows,
            "first_timestamp_ns": min((ns(row[0]) for row in rows), default=None),
            "last_timestamp_ns": max((ns(row[0]) for row in rows), default=None),
            "schema_version": bf.EXTRACT_SCHEMA_VERSION,
            "status": "complete",
        }
    manifest = {
        "status": "complete",
        "schema_version": bf.EXTRACT_SCHEMA_VERSION,
        "captures": entries,
    }
    (root / "_extract_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def make_job(
    metadata_file: Path,
    stage_root: Path,
    *,
    capture_id: str = "24-01-01",
    resume: bool = False,
    stop_file: Path | None = None,
    inventory_mtime: int | None = None,
) -> bf.DateJob:
    stat = metadata_file.stat()
    return bf.DateJob(
        dataset="NCSU",
        capture_id=capture_id,
        labels=LABELS,
        label_sha256="fixture-label-hash",
        files=(str(metadata_file),),
        source_inventory=(
            {
                "path": f"dataset=NCSU/date={capture_id}/{metadata_file.name}",
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns
                if inventory_mtime is None
                else inventory_mtime,
                "row_count": pq.ParquetFile(metadata_file).metadata.num_rows,
            },
        ),
        capture_start_ns=ns(1000),
        window_seconds=60,
        batch_rows=2,
        stage_root=str(stage_root),
        resume=resume,
        stop_file=str(stop_file) if stop_file is not None else None,
    )


def aggregate(
    rows: list[tuple[float, int, str | None, str | None]],
    *,
    capture_start: float = 1000,
    capture_id: str = "24-01-01",
) -> dict[str, list[dict]]:
    state: dict[tuple[str, int], bf.WindowStats] = {}
    table = packet_table(rows).select(list(bf.PACKET_COLUMNS))
    for batch in table.to_batches(max_chunksize=2):
        bf.update_aggregates_from_batch(
            state,
            batch,
            pa.array(sorted(LABELS)),
            ns(capture_start),
            60,
        )
    return bf.rows_from_aggregates(state, LABELS, "NCSU", capture_id, ns(capture_start))


def test_vectorized_window_features_idle_time_and_l2_traffic():
    rows = aggregate(
        [
            # These frames need no IPv4 columns: ARP/IPv6 is included at L2.
            (1000, 10, MAC_A.upper(), MAC_C),
            (1010, 20, MAC_C, MAC_A),
            (1060, 30, MAC_A, MAC_B),
            (1185, 40, MAC_C, MAC_A),
            (1190, 50, MAC_C, MAC_C),  # neither endpoint is mapped
            (1191, 60, None, None),
        ]
    )
    device_a = rows[MAC_A]
    assert [row["window"] for row in device_a] == [0, 1, 3]
    assert device_a[0]["outgoing_packet_count"] == 1
    assert device_a[0]["incoming_packet_count"] == 1
    assert device_a[0]["aggregated_total_bytes"] == 30
    assert device_a[0]["idle_time"] == 0.0
    assert device_a[1]["idle_time"] == 50.0
    assert device_a[2]["idle_time"] == 125.0
    assert all(row["device_id_type"] == "mac" for row in device_a)

    device_b = rows[MAC_B]
    assert len(device_b) == 1
    assert (
        device_b[0]["outgoing_packet_count"],
        device_b[0]["incoming_packet_count"],
    ) == (
        0,
        1,
    )
    assert device_b[0]["idle_time"] == 60.0


def test_frame_between_mapped_macs_counts_both_devices():
    rows = aggregate([(1001, 100, MAC_A, MAC_B)])
    a = rows[MAC_A][0]
    b = rows[MAC_B][0]
    assert (a["outgoing_packet_count"], a["incoming_packet_count"]) == (1, 0)
    assert (b["outgoing_packet_count"], b["incoming_packet_count"]) == (0, 1)


def test_expanded_frame_size_and_timing_features():
    rows = aggregate(
        [
            (1000, 100, MAC_A, MAC_C),
            (1001, 200, MAC_A, MAC_C),
            (1003, 1200, MAC_C, MAC_A),
        ]
    )
    row = rows[MAC_A][0]
    assert row["outgoing_mean_frame_size"] == 150.0
    assert row["outgoing_std_frame_size"] == 50.0
    assert row["outgoing_small_frame_ratio"] == 0.5
    assert row["incoming_mean_frame_size"] == 1200.0
    assert row["incoming_std_frame_size"] is None
    assert row["incoming_large_frame_ratio"] == 1.0
    assert row["active_span"] == 3.0
    assert row["inter_arrival_cv"] == pytest.approx(1.0 / 3.0)
    assert row["max_inter_arrival_time"] == 2.0


def test_self_addressed_frame_is_not_duplicated_in_timing_features():
    row = aggregate(
        [(1000, 100, MAC_A, MAC_A), (1001, 100, MAC_A, MAC_C)]
    )[MAC_A][0]
    assert row["aggregated_packet_count"] == 3
    assert row["max_inter_arrival_time"] == 1.0
    assert row["inter_arrival_cv"] is None


def test_timestamp_units_are_normalized_to_nanoseconds():
    schema = pa.schema(
        [
            pa.field("timestamp_utc", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("frame_len", pa.uint32(), nullable=False),
            pa.field("eth_src", pa.string()),
            pa.field("eth_dst", pa.string()),
        ]
    )
    batch = pa.RecordBatch.from_pydict(
        {
            "timestamp_utc": [1_000_000],
            "frame_len": [100],
            "eth_src": [MAC_A],
            "eth_dst": [MAC_C],
        },
        schema=schema,
    )
    state = {}
    bf.update_aggregates_from_batch(
        state, batch, pa.array([MAC_A]), bf.NANOSECONDS_PER_SECOND, 60
    )
    rows = bf.rows_from_aggregates(
        state, LABELS, "NCSU", "24-01-01", bf.NANOSECONDS_PER_SECOND
    )
    assert rows[MAC_A][0]["window"] == 0
    assert rows[MAC_A][0]["idle_time"] == 0.0


def test_capture_dates_reset_window_anchor_and_idle_time():
    first = aggregate([(1061, 1, MAC_A, MAC_C)], capture_id="24-01-01")
    second = aggregate(
        [(2005, 1, MAC_A, MAC_C)],
        capture_start=2000,
        capture_id="24-01-02",
    )
    assert first[MAC_A][0]["idle_time"] == 61.0
    assert second[MAC_A][0]["idle_time"] == 5.0
    assert second[MAC_A][0]["window"] == 0


def test_invalid_window_and_preceding_timestamp_are_rejected():
    batch = (
        packet_table([(999, 1, MAC_A, MAC_C)])
        .select(list(bf.PACKET_COLUMNS))
        .to_batches()[0]
    )
    with pytest.raises(ValueError, match="greater than zero"):
        bf.update_aggregates_from_batch({}, batch, pa.array([MAC_A]), ns(1000), 0)
    with pytest.raises(bf.PipelineError, match="precedes"):
        bf.update_aggregates_from_batch({}, batch, pa.array([MAC_A]), ns(1000), 60)


def test_mac_label_validation_and_distinct_gateway_interfaces(tmp_path):
    gateways = {
        "00:01:c0:18:7f:9b": {
            "device_name": "Gateway",
            "is_iot_device": 0,
            "device_type": "router",
            "vendor": "Unknown",
        },
        "7c:c3:a1:88:b6:8d": {
            "device_name": "Gateway",
            "is_iot_device": 0,
            "device_type": "router",
            "vendor": "Unknown",
        },
    }
    write_labels(tmp_path, "YourThing", gateways)
    loaded, digest = bf.load_labels(tmp_path, "YourThing")
    assert list(loaded) == list(gateways)
    assert len(loaded) == 2
    assert len(digest) == 64

    bad = {MAC_A.upper(): LABELS[MAC_A]}
    write_labels(tmp_path, "NCSU", bad)
    with pytest.raises(bf.PipelineError, match="lowercase MAC"):
        bf.load_labels(tmp_path, "NCSU")


@pytest.mark.parametrize("label_file", ["", "../labels.json", "labels/file.json"])
def test_label_file_must_be_a_basename(label_file, tmp_path):
    with pytest.raises(bf.PipelineError, match="non-empty filename"):
        bf.load_labels(tmp_path, "NCSU", label_file)


def test_multiclass_labels_and_int32_schema(tmp_path):
    labels = {MAC_A: {**LABELS[MAC_A], "is_iot_device": 2}}
    write_labels(tmp_path, labels=labels, filename="mac_mapping_multi_labels.json")
    loaded, _ = bf.load_labels(tmp_path, "NCSU", "mac_mapping_multi_labels.json")
    assert loaded[MAC_A]["is_iot_device"] == 2
    assert bf.FEATURE_SCHEMA.field("is_iot_device").type == pa.int32()

    labels[MAC_A]["is_iot_device"] = 1.5
    write_labels(tmp_path, labels=labels, filename="mac_mapping_multi_labels.json")
    with pytest.raises(bf.PipelineError, match="non-negative integer"):
        bf.load_labels(tmp_path, "NCSU", "mac_mapping_multi_labels.json")


def test_extract_manifest_must_be_complete(tmp_path):
    (tmp_path / "_extract_manifest.json").write_text(
        json.dumps({"status": "running", "schema_version": 1, "captures": {}}),
        encoding="utf-8",
    )
    with pytest.raises(bf.PipelineError, match="not complete"):
        bf.load_extract_manifest(tmp_path)


def test_process_date_writes_empty_device_and_reuses_checkpoint(tmp_path, monkeypatch):
    metadata = tmp_path / "capture.parquet"
    pq.write_table(packet_table([(1001, 42, MAC_A, MAC_C)]), metadata)
    stage = tmp_path / "stage"
    first = bf.process_date(make_job(metadata, stage))
    assert first["status"] == "complete"
    assert first["resumed"] is False
    target = stage / "NCSU" / "24-01-01"
    assert pq.read_table(target / f"{MAC_A}.parquet").num_rows == 1
    assert pq.read_table(target / f"{MAC_B}.parquet").num_rows == 0

    monkeypatch.setattr(
        bf,
        "_validate_packet_file",
        lambda *args: pytest.fail("completed checkpoint should be reused"),
    )
    second = bf.process_date(make_job(metadata, stage, resume=True))
    assert second["resumed"] is True


def test_resume_fingerprint_invalidates_changed_inventory(tmp_path, monkeypatch):
    metadata = tmp_path / "capture.parquet"
    pq.write_table(packet_table([(1001, 42, MAC_A, MAC_C)]), metadata)
    stage = tmp_path / "stage"
    original = make_job(metadata, stage)
    bf.process_date(original)

    calls = 0
    real_validate = bf._validate_packet_file

    def counted(path):
        nonlocal calls
        calls += 1
        return real_validate(path)

    monkeypatch.setattr(bf, "_validate_packet_file", counted)
    changed = make_job(
        metadata,
        stage,
        resume=True,
        inventory_mtime=original.source_inventory[0]["mtime_ns"] + 1,
    )
    result = bf.process_date(changed)
    assert result["resumed"] is False
    assert calls == 1


def test_process_date_stop_discards_partial_work_then_resumes(tmp_path, monkeypatch):
    metadata = tmp_path / "capture.parquet"
    pq.write_table(
        packet_table([(1001, 10, MAC_A, MAC_C), (1002, 20, MAC_A, MAC_C)]),
        metadata,
        row_group_size=1,
    )
    stage = tmp_path / "stage"
    job = make_job(metadata, stage)
    checks = 0

    def stop_during_batches(_path):
        nonlocal checks
        checks += 1
        return checks >= 3

    monkeypatch.setattr(bf, "stop_requested", stop_during_batches)
    assert bf.process_date(job)["status"] == "stopped"
    assert not (stage / "NCSU" / "24-01-01").exists()
    assert not list((stage / "NCSU").glob(".*.tmp"))

    monkeypatch.setattr(bf, "stop_requested", lambda _path: False)
    completed = bf.process_date(dataclasses.replace(job, resume=True))
    assert completed["status"] == "complete"


def test_safe_output_modes(tmp_path):
    root = tmp_path / "window_60s"
    bf.prepare_output_root(root, resume=False, overwrite=False)
    marker = root / "marker"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(bf.PipelineError, match="already exists"):
        bf.prepare_output_root(root, resume=False, overwrite=False)
    bf.prepare_output_root(root, resume=True, overwrite=False)
    assert marker.exists()
    bf.prepare_output_root(root, resume=False, overwrite=True)
    assert root.exists() and not marker.exists()


def test_preexisting_stop_file_exits_zero_without_overwrite(tmp_path):
    output = tmp_path / "output"
    window = output / "window_60s"
    window.mkdir(parents=True)
    marker = window / "keep"
    marker.write_text("preserved", encoding="utf-8")
    stop = tmp_path / "stop"
    stop.touch()

    result = bf.main(
        [
            "--output-root",
            str(output),
            "--stop-file",
            str(stop),
            "--overwrite",
        ]
    )
    assert result == 0
    assert marker.read_text(encoding="utf-8") == "preserved"


def test_pipeline_parallel_output_resume_and_configuration_guard(tmp_path):
    metadata_root = tmp_path / "metadata"
    labels_root = tmp_path / "labels"
    output_root = tmp_path / "features"
    metadata_root.mkdir()
    write_labels(labels_root)
    write_metadata(
        metadata_root,
        [
            ("NCSU", "24-01-01", "one.parquet", [(1000, 10, MAC_A, MAC_C)]),
            (
                "NCSU",
                "24-01-02",
                "two.parquet",
                [(2005, 20, MAC_C, MAC_A), (2065, 30, MAC_A, MAC_B)],
            ),
        ],
    )
    arguments = [
        "--metadata-root",
        str(metadata_root),
        "--labels-root",
        str(labels_root),
        "--output-root",
        str(output_root),
        "--datasets",
        "NCSU",
        "--workers",
        "2",
        "--batch-rows",
        "1",
    ]
    window_root = bf.run_pipeline(bf.parse_args(arguments))
    assert window_root == output_root / "window_60s"
    table_a = pq.read_table(window_root / "NCSU" / MAC_A / "features.parquet")
    assert table_a.num_rows == 3
    assert table_a.column("capture_id").to_pylist() == [
        "24-01-01",
        "24-01-02",
        "24-01-02",
    ]
    table_b = pq.read_table(window_root / "NCSU" / MAC_B / "features.parquet")
    assert table_b.num_rows == 1
    run_manifest = json.loads((window_root / "_run_manifest.json").read_text())
    assert run_manifest["status"] == "complete"
    assert run_manifest["algorithm_version"] == 2
    assert run_manifest["configuration"]["model_feature_columns"] == list(
        bf.MODEL_FEATURE_COLUMNS
    )
    assert run_manifest["configuration"]["nullable_feature_columns"] == list(
        bf.NULLABLE_FEATURE_COLUMNS
    )
    assert run_manifest["configuration"]["missing_indicator_columns"] == list(
        bf.MISSING_INDICATOR_COLUMNS
    )
    assert run_manifest["datasets"]["NCSU"]["identifier_type"] == "mac"
    loaded = common.load_mac_datasets(["NCSU"], window_root, labels_root)
    assert common.raw_feature_matrix(loaded.frame).shape == (4, 22)

    resumed = bf.run_pipeline(bf.parse_args([*arguments, "--resume"]))
    assert resumed == window_root
    resumed_manifest = json.loads((window_root / "_run_manifest.json").read_text())
    assert all(
        entry["resumed"]
        for entry in resumed_manifest["datasets"]["NCSU"]["dates"].values()
    )

    alternate = "mac_mapping_multi_labels.json"
    write_labels(labels_root, filename=alternate)
    with pytest.raises(bf.PipelineError, match="incompatible"):
        bf.run_pipeline(
            bf.parse_args([*arguments, "--resume", "--label-file", alternate])
        )


def test_active_stop_records_stopped_manifest(tmp_path, monkeypatch):
    metadata_root = tmp_path / "metadata"
    labels_root = tmp_path / "labels"
    output_root = tmp_path / "features"
    metadata_root.mkdir()
    write_labels(labels_root)
    write_metadata(
        metadata_root,
        [("NCSU", "24-01-01", "one.parquet", [(1000, 10, MAC_A, MAC_C)])],
    )
    monkeypatch.setattr(bf, "_run_bounded_jobs", lambda *args, **kwargs: True)
    result = bf.run_pipeline(
        bf.parse_args(
            [
                "--metadata-root",
                str(metadata_root),
                "--labels-root",
                str(labels_root),
                "--output-root",
                str(output_root),
                "--datasets",
                "NCSU",
            ]
        )
    )
    assert result == output_root / "window_60s"
    manifest = json.loads((result / "_run_manifest.json").read_text())
    assert manifest["status"] == "stopped"
    assert "stopped_at" in manifest


def test_cli_defaults_and_stop_disabled_by_default():
    args = bf.parse_args([])
    assert args.metadata_root == Path("extracted_metadata_dataset")
    assert args.labels_root == Path("labels")
    assert args.label_file == "mac_mapping_binary_labels.json"
    assert args.output_root == Path("built_features_mac_v2")
    assert args.batch_rows == 100_000
    assert args.stop_file is None
