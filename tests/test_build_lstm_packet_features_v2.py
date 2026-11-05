from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code_v2 import build_lstm_packet_features as builder


MAC_A = "02:00:00:00:00:01"
MAC_B = "02:00:00:00:00:02"
MAC_C = "02:00:00:00:00:03"
MULTICAST = "01:00:5e:00:00:01"
BROADCAST = "ff:ff:ff:ff:ff:ff"

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
    return round(seconds * builder.NS_PER_SECOND)


def packet_table(
    rows: list[tuple[float, int, str | None, str | None]],
    *,
    include_ethertype: bool = True,
) -> pa.Table:
    values = {
        "timestamp_utc": [ns(row[0]) for row in rows],
        "frame_len": [row[1] for row in rows],
        "eth_src": [row[2] for row in rows],
        "eth_dst": [row[3] for row in rows],
    }
    fields = [
        pa.field("timestamp_utc", pa.timestamp("ns", tz="UTC"), nullable=False),
        pa.field("frame_len", pa.uint32(), nullable=False),
        pa.field("eth_src", pa.string()),
        pa.field("eth_dst", pa.string()),
    ]
    if include_ethertype:
        # The LSTM builder must ignore this field even when the extractor emits it.
        values["eth_type"] = [0x0800] * len(rows)
        fields.append(pa.field("eth_type", pa.uint16()))
    return pa.Table.from_pydict(values, schema=pa.schema(fields))


def write_labels(
    root: Path,
    *,
    dataset: str = "NCSU",
    labels: dict = LABELS,
    filename: str = builder.DEFAULT_LABEL_FILE,
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
) -> None:
    entries = {}
    for dataset, capture_date, filename, rows in captures:
        relative = Path(f"dataset={dataset}") / f"date={capture_date}" / filename
        output = root / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        table = packet_table(rows)
        pq.write_table(table, output, compression="zstd")
        entries[f"{dataset}/data/{capture_date}/{filename}"] = {
            "dataset": dataset,
            "capture_date": capture_date,
            "output": str(relative),
            "output_size": output.stat().st_size,
            "row_count": table.num_rows,
            "schema_version": builder.EXTRACT_SCHEMA_VERSION,
            "status": "complete",
        }
    manifest = {
        "status": "complete",
        "schema_version": builder.EXTRACT_SCHEMA_VERSION,
        "captures": entries,
    }
    (root / "_extract_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def make_job(
    files: list[Path],
    stage_root: Path,
    *,
    capture_id: str = "24-01-01",
    labels: dict = LABELS,
    resume: bool = False,
    stop_file: Path | None = None,
    inventory_mtime_delta: int = 0,
) -> builder.CaptureJob:
    inventory = []
    for path in files:
        stat = path.stat()
        inventory.append(
            {
                "path": f"dataset=NCSU/date={capture_id}/{path.name}",
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns + inventory_mtime_delta,
                "row_count": pq.ParquetFile(path).metadata.num_rows,
            }
        )
    return builder.CaptureJob(
        dataset="NCSU",
        capture_id=capture_id,
        files=tuple(str(path) for path in files),
        inventory=tuple(inventory),
        labels=labels,
        label_sha256="fixture-label-hash",
        stage_root=str(stage_root),
        batch_rows=2,
        write_rows=2,
        resume=resume,
        stop_file=str(stop_file) if stop_file is not None else None,
    )


def read_staged(stage_root: Path, device_id: str, date: str = "24-01-01"):
    return pq.read_table(stage_root / "NCSU" / date / f"{device_id}.parquet")


def test_feature_contract_has_only_five_l2_groups():
    assert builder.MODEL_FEATURE_COLUMNS == (
        "frame_size",
        "direction",
        "inter_arrival_time",
        "dst_unicast",
        "dst_multicast",
        "dst_broadcast",
        "same_direction_inter_arrival_time",
    )
    assert builder.AUXILIARY_FEATURE_COLUMNS == (
        "inter_arrival_valid",
        "same_direction_iat_valid",
    )
    assert "eth_type" not in builder.PACKET_SCHEMA.names
    assert not any("ether" in name for name in builder.PACKET_SCHEMA.names)
    assert not any("sequence" in name for name in builder.PACKET_SCHEMA.names)


def test_orders_packets_and_calculates_all_features(tmp_path):
    source = tmp_path / "packets.parquet"
    pq.write_table(
        packet_table(
            [
                (3, 30, MAC_A, MULTICAST),
                (1, 10, MAC_A.upper(), MAC_C),
                (4, 40, MAC_A, BROADCAST),
                (2, 20, MAC_C, MAC_A),
            ],
            include_ethertype=False,
        ),
        source,
    )
    stage = tmp_path / "stage"
    result = builder.process_capture(make_job([source], stage))

    assert result["status"] == "complete"
    table = read_staged(stage, MAC_A)
    assert table.column("packet_index").to_pylist() == [0, 1, 2, 3]
    assert table.column("timestamp_ns").to_pylist() == [ns(1), ns(2), ns(3), ns(4)]
    assert table.column("frame_size").to_pylist() == [10, 20, 30, 40]
    assert table.column("direction").to_pylist() == [1, 0, 1, 1]
    assert table.column("inter_arrival_time").to_pylist() == [0.0, 1.0, 1.0, 1.0]
    assert table.column("inter_arrival_valid").to_pylist() == [False, True, True, True]
    assert table.column("same_direction_inter_arrival_time").to_pylist() == [
        0.0,
        0.0,
        2.0,
        1.0,
    ]
    assert table.column("same_direction_iat_valid").to_pylist() == [
        False,
        False,
        True,
        True,
    ]
    assert table.column("dst_unicast").to_pylist() == [1, 1, 0, 0]
    assert table.column("dst_multicast").to_pylist() == [0, 0, 1, 0]
    assert table.column("dst_broadcast").to_pylist() == [0, 0, 0, 1]


def test_dual_endpoints_self_frames_and_unrelated_frames(tmp_path):
    source = tmp_path / "packets.parquet"
    pq.write_table(
        packet_table(
            [
                (1, 10, MAC_A, MAC_B),
                (2, 20, MAC_A, MAC_A),
                (3, 30, MAC_C, BROADCAST),
            ]
        ),
        source,
    )
    stage = tmp_path / "stage"
    result = builder.process_capture(make_job([source], stage))

    assert result["observed_device_packet_rows"] == 3
    a = read_staged(stage, MAC_A)
    b = read_staged(stage, MAC_B)
    assert a.column("direction").to_pylist() == [1, 1]
    assert b.column("direction").to_pylist() == [0]
    assert b.column("frame_size").to_pylist() == [10]


def test_multiple_files_and_equal_timestamps_have_deterministic_order(tmp_path):
    first = tmp_path / "a.parquet"
    second = tmp_path / "b.parquet"
    pq.write_table(
        packet_table([(2, 30, MAC_A, MAC_C), (1, 10, MAC_A, MAC_C)]), first
    )
    pq.write_table(packet_table([(1, 20, MAC_A, MAC_C)]), second)
    stage = tmp_path / "stage"

    builder.process_capture(make_job([first, second], stage))
    table = read_staged(stage, MAC_A)
    assert table.column("packet_index").to_pylist() == [0, 1, 2]
    assert table.column("timestamp_ns").to_pylist() == [ns(1), ns(1), ns(2)]
    assert table.column("frame_size").to_pylist() == [10, 20, 30]
    assert table.column("inter_arrival_time").to_pylist() == [0.0, 0.0, 1.0]


def test_empty_device_output_and_source_schema_validation(tmp_path):
    source = tmp_path / "packets.parquet"
    pq.write_table(packet_table([(1, 10, MAC_A, MAC_C)]), source)
    stage = tmp_path / "stage"
    builder.process_capture(make_job([source], stage))
    assert read_staged(stage, MAC_B).num_rows == 0

    invalid = tmp_path / "invalid.parquet"
    pq.write_table(packet_table([(1, 10, MAC_A, MAC_C)]).drop(["eth_dst"]), invalid)
    with pytest.raises(builder.PipelineError, match="missing columns"):
        builder.parquet_source(invalid)


def test_capture_checkpoint_resume_stop_and_inventory_change(tmp_path, monkeypatch):
    source = tmp_path / "packets.parquet"
    pq.write_table(packet_table([(1, 10, MAC_A, MAC_C)]), source)
    stage = tmp_path / "stage"
    original = make_job([source], stage)
    builder.process_capture(original)

    monkeypatch.setattr(
        builder,
        "parquet_source",
        lambda _path: pytest.fail("valid checkpoint should not rescan source"),
    )
    resumed = builder.process_capture(dataclasses.replace(original, resume=True))
    assert resumed["resumed"] is True

    monkeypatch.undo()
    changed = make_job([source], stage, resume=True, inventory_mtime_delta=1)
    assert builder.process_capture(changed)["resumed"] is False

    stop = tmp_path / "stop"
    stop.touch()
    stopped_stage = tmp_path / "stopped-stage"
    stopped = builder.process_capture(
        make_job([source], stopped_stage, stop_file=stop)
    )
    assert stopped["status"] == "stopped"
    assert not (stopped_stage / "NCSU" / "24-01-01").exists()


def test_pipeline_consolidates_dates_and_writes_manifest_contract(tmp_path):
    metadata = tmp_path / "metadata"
    labels = tmp_path / "labels"
    output = tmp_path / "binary-class"
    metadata.mkdir()
    write_labels(labels)
    write_metadata(
        metadata,
        [
            ("NCSU", "24-01-01", "b.parquet", [(2, 20, MAC_A, MAC_C)]),
            ("NCSU", "24-01-01", "a.parquet", [(1, 10, MAC_A, MAC_C)]),
            ("NCSU", "24-01-02", "c.parquet", [(3, 30, MAC_A, MAC_C)]),
        ],
    )
    arguments = [
        "--metadata-root",
        str(metadata),
        "--labels-root",
        str(labels),
        "--output-root",
        str(output),
        "--datasets",
        "NCSU",
        "--workers",
        "1",
        "--batch-rows",
        "1",
        "--write-rows",
        "1",
    ]
    root = builder.run_pipeline(builder.parse_args(arguments))

    assert root == output / "packet_features"
    table = pq.read_table(root / "NCSU" / MAC_A / "features.parquet")
    assert table.column("capture_id").to_pylist() == [
        "24-01-01",
        "24-01-01",
        "24-01-02",
    ]
    assert table.column("packet_index").to_pylist() == [0, 1, 0]
    assert table.column("inter_arrival_valid").to_pylist() == [False, True, False]
    assert table.column("same_direction_iat_valid").to_pylist() == [
        False,
        True,
        False,
    ]
    manifest = json.loads((root / "_run_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["algorithm_version"] == 1
    assert manifest["model_feature_columns"] == list(builder.MODEL_FEATURE_COLUMNS)
    assert manifest["auxiliary_feature_columns"] == list(
        builder.AUXILIARY_FEATURE_COLUMNS
    )
    assert manifest["model_input_width"] == 7
    assert manifest["datasets"]["NCSU"]["identifier_type"] == "mac"
    assert "known_ethertype_rows" not in manifest["datasets"]["NCSU"]["dates"][
        "24-01-01"
    ]

    resumed = builder.run_pipeline(builder.parse_args([*arguments, "--resume"]))
    resumed_manifest = json.loads((resumed / "_run_manifest.json").read_text())
    assert all(
        value["resumed"]
        for value in resumed_manifest["datasets"]["NCSU"]["dates"].values()
    )


def test_binary_and_multiclass_label_maps_use_manual_output_roots(tmp_path):
    metadata = tmp_path / "metadata"
    labels = tmp_path / "labels"
    metadata.mkdir()
    write_labels(labels)
    multiclass = {MAC_A: {**LABELS[MAC_A], "is_iot_device": 2}, MAC_B: LABELS[MAC_B]}
    write_labels(
        labels,
        labels=multiclass,
        filename="mac_mapping_multi_labels.json",
    )
    write_metadata(
        metadata,
        [("NCSU", "24-01-01", "one.parquet", [(1, 10, MAC_A, MAC_C)])],
    )

    common = [
        "--metadata-root",
        str(metadata),
        "--labels-root",
        str(labels),
        "--datasets",
        "NCSU",
        "--workers",
        "1",
    ]
    binary_root = builder.run_pipeline(
        builder.parse_args([*common, "--output-root", str(tmp_path / "binary-class")])
    )
    multiclass_root = builder.run_pipeline(
        builder.parse_args(
            [
                *common,
                "--label-file",
                "mac_mapping_multi_labels.json",
                "--output-root",
                str(tmp_path / "multiclass"),
            ]
        )
    )
    binary = pq.read_table(binary_root / "NCSU" / MAC_A / "features.parquet")
    multi = pq.read_table(multiclass_root / "NCSU" / MAC_A / "features.parquet")
    assert set(binary.column("is_iot_device").to_pylist()) == {1}
    assert set(multi.column("is_iot_device").to_pylist()) == {2}


def test_preexisting_stop_preserves_output_even_with_overwrite(tmp_path):
    output = tmp_path / "binary-class"
    root = output / "packet_features"
    root.mkdir(parents=True)
    marker = root / "keep"
    marker.write_text("preserved", encoding="utf-8")
    stop = tmp_path / "stop"
    stop.touch()

    result = builder.main(
        ["--output-root", str(output), "--stop-file", str(stop), "--overwrite"]
    )
    assert result == 0
    assert marker.read_text(encoding="utf-8") == "preserved"


def test_existing_output_requires_resume_or_overwrite(tmp_path):
    output = tmp_path / "binary-class"
    (output / "packet_features").mkdir(parents=True)

    with pytest.raises(builder.PipelineError, match="Output exists"):
        builder.run_pipeline(
            builder.parse_args(["--output-root", str(output)])
        )


def test_cli_defaults_are_repository_relative():
    args = builder.parse_args([])
    assert args.metadata_root == builder.DEFAULT_METADATA_ROOT
    assert args.labels_root == builder.DEFAULT_LABELS_ROOT
    assert args.output_root == builder.DEFAULT_OUTPUT_ROOT
    assert args.label_file == "mac_mapping_binary_labels.json"
    assert args.datasets == list(builder.DATASETS)
    assert not hasattr(args, "sequence_packets")
    assert not hasattr(args, "require_ethertype")

