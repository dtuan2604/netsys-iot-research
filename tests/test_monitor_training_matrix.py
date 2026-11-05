from __future__ import annotations

import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1] / "code_v2"
sys.path.insert(0, str(CODE_ROOT))

import monitor_training_matrix as monitor  # noqa: E402


def write_manifest(path: Path, status: str = "running") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": status,
                "started_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T01:00:00+00:00",
                "runtime_allocation": {
                    "concurrent_experiments": 4,
                    "workers_per_experiment": 3,
                    "cpu_affinity": "unrestricted",
                    "memory_limit": "unrestricted",
                },
                "jobs": {
                    "a": {"status": "complete"},
                    "b": {"status": "running"},
                    "c": {},
                },
            }
        ),
        encoding="utf-8",
    )


def test_snapshot_and_message_report_matrix_progress(tmp_path):
    manifest = tmp_path / "_matrix_manifest.json"
    write_manifest(manifest)

    snapshot = monitor.load_snapshot(manifest)

    assert snapshot is not None
    assert snapshot.status == "running"
    assert snapshot.total == 3
    assert snapshot.complete == 1
    assert snapshot.counts == {"complete": 1, "running": 1, "pending": 1}
    message = monitor.format_message(
        snapshot, host="server", label="Decision Tree · Ubuntu Server"
    )
    assert "1/3" in message
    assert "Decision Tree · Ubuntu Server update" in message
    assert "Experiment concurrency: 4" in message
    assert "Trainer workers per experiment: 3" in message
    assert "Host: `server`" in message


def test_once_sends_notification_and_persists_deduplication_state(
    tmp_path, monkeypatch
):
    manifest = tmp_path / "_matrix_manifest.json"
    state = tmp_path / "state.json"
    write_manifest(manifest, status="complete")
    messages = []
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.example/webhook")
    monkeypatch.setattr(
        monitor,
        "send_discord_message",
        lambda url, message, **kwargs: messages.append((url, message)),
    )

    arguments = monitor.parse_args(
        [
            "--manifest",
            str(manifest),
            "--state-file",
            str(state),
            "--once",
        ]
    )
    assert monitor.monitor(arguments) == 0
    assert len(messages) == 1
    assert state.is_file()

    assert monitor.monitor(arguments) == 0
    assert len(messages) == 1


def test_webhook_url_file_takes_precedence(tmp_path, monkeypatch):
    url_file = tmp_path / "webhook"
    url_file.write_text("https://discord.example/from-file\n", encoding="utf-8")
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.example/from-env")

    assert (
        monitor.load_webhook_url("DISCORD_WEBHOOK_URL", url_file)
        == "https://discord.example/from-file"
    )
