#!/usr/bin/env python3
"""Monitor a training-matrix manifest and send progress to Discord."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


TERMINAL_STATUSES = {"complete", "failed", "stopped"}


class MonitorError(RuntimeError):
    """Invalid monitor configuration or data."""


@dataclass(frozen=True)
class MatrixSnapshot:
    status: str
    total: int
    counts: Mapping[str, int]
    updated_at: str | None
    started_at: str | None
    error: str | None
    concurrency: int | None
    workers_per_experiment: int | None

    @property
    def complete(self) -> int:
        return int(self.counts.get("complete", 0))

    @property
    def signature(self) -> str:
        value = {
            "status": self.status,
            "total": self.total,
            "counts": dict(sorted(self.counts.items())),
            "error": self.error,
            "started_at": self.started_at,
            "concurrency": self.concurrency,
            "workers_per_experiment": self.workers_per_experiment,
        }
        return json.dumps(value, sort_keys=True, separators=(",", ":"))


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MonitorError(f"cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MonitorError(f"expected a JSON object in {path}")
    return value


def load_snapshot(manifest_path: Path) -> MatrixSnapshot | None:
    if not manifest_path.is_file():
        return None
    manifest = read_json(manifest_path)
    jobs = manifest.get("jobs")
    if not isinstance(jobs, dict):
        raise MonitorError(f"matrix manifest has no jobs object: {manifest_path}")
    counts: collections.Counter[str] = collections.Counter()
    for entry in jobs.values():
        status = (
            entry.get("status", "pending") if isinstance(entry, dict) else "invalid"
        )
        counts[str(status)] += 1
    runtime = manifest.get("runtime_allocation")
    concurrency = None
    workers_per_experiment = None
    if isinstance(runtime, dict) and isinstance(
        runtime.get("concurrent_experiments"), int
    ):
        concurrency = int(runtime["concurrent_experiments"])
    if isinstance(runtime, dict) and isinstance(
        runtime.get("workers_per_experiment"), int
    ):
        workers_per_experiment = int(runtime["workers_per_experiment"])
    return MatrixSnapshot(
        status=str(manifest.get("status", "unknown")),
        total=len(jobs),
        counts=dict(counts),
        updated_at=_optional_string(manifest.get("updated_at")),
        started_at=_optional_string(manifest.get("started_at")),
        error=_optional_string(manifest.get("error")),
        concurrency=concurrency,
        workers_per_experiment=workers_per_experiment,
    )


def _optional_string(value: Any) -> str | None:
    return str(value) if value is not None else None


def format_duration(started_at: str | None) -> str | None:
    if not started_at:
        return None
    try:
        started = dt.datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        if started.tzinfo is None:
            started = started.replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None
    seconds = max(0, int((dt.datetime.now(dt.timezone.utc) - started).total_seconds()))
    days, remainder = divmod(seconds, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, _ = divmod(remainder, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def format_message(
    snapshot: MatrixSnapshot | None,
    *,
    host: str,
    label: str = "MAC training matrix",
    heartbeat: bool = False,
    runner_exited: bool = False,
) -> str:
    if snapshot is None:
        return (
            f"⏳ **{label}**\n"
            "Waiting for the matrix manifest to be created.\n"
            f"Host: `{host}`"
        )
    emoji = {
        "running": "🔵",
        "complete": "✅",
        "failed": "❌",
        "stopped": "⏹️",
    }.get(snapshot.status, "ℹ️")
    title = f"{label} heartbeat" if heartbeat else f"{label} update"
    percentage = 100.0 * snapshot.complete / snapshot.total if snapshot.total else 0.0
    counts = snapshot.counts
    lines = [
        f"{emoji} **{title}**",
        f"Status: **{snapshot.status}**",
        f"Completed: **{snapshot.complete}/{snapshot.total}** ({percentage:.1f}%)",
        (
            f"Running: {counts.get('running', 0)} | "
            f"Pending: {counts.get('pending', 0)} | "
            f"Failed: {counts.get('failed', 0)} | "
            f"Stopped: {counts.get('stopped', 0)}"
        ),
    ]
    if snapshot.concurrency is not None:
        lines.append(f"Experiment concurrency: {snapshot.concurrency}")
    if snapshot.workers_per_experiment is not None:
        lines.append(
            f"Trainer workers per experiment: {snapshot.workers_per_experiment}"
        )
    duration = format_duration(snapshot.started_at)
    if duration:
        lines.append(f"Elapsed: {duration}")
    if snapshot.updated_at:
        lines.append(f"Manifest updated: `{snapshot.updated_at}`")
    if snapshot.error:
        lines.append(f"Error: `{snapshot.error[:500]}`")
    if runner_exited:
        lines.append("⚠️ The recorded runner process is no longer active.")
    lines.append(f"Host: `{host}`")
    return "\n".join(lines)[:2000]


def load_webhook_url(env_name: str, url_file: Path | None) -> str:
    if url_file is not None:
        try:
            url = url_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise MonitorError(
                f"cannot read webhook URL file {url_file}: {exc}"
            ) from exc
    else:
        url = os.environ.get(env_name, "").strip()
    if not url:
        source = (
            str(url_file)
            if url_file is not None
            else f"environment variable {env_name}"
        )
        raise MonitorError(f"Discord webhook URL is missing from {source}")
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise MonitorError("Discord webhook URL must be a valid HTTPS URL")
    return url


def send_discord_message(
    webhook_url: str,
    message: str,
    *,
    username: str,
    attempts: int = 5,
) -> None:
    payload = json.dumps(
        {"content": message, "username": username}, ensure_ascii=False
    ).encode("utf-8")
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(
            webhook_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "mac-training-matrix-monitor/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                if 200 <= response.status < 300:
                    return
                raise MonitorError(f"Discord returned HTTP {response.status}")
        except urllib.error.HTTPError as exc:
            delay = min(60.0, float(2 ** (attempt - 1)))
            if exc.code == 429:
                try:
                    body = json.loads(exc.read().decode("utf-8"))
                    delay = max(0.1, float(body.get("retry_after", delay)))
                except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                    pass
            if attempt == attempts:
                raise MonitorError(
                    f"Discord notification failed with HTTP {exc.code}"
                ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            delay = min(60.0, float(2 ** (attempt - 1)))
            if attempt == attempts:
                raise MonitorError("Discord notification failed after retries") from exc
        time.sleep(delay)


def read_runner_pid(pid_file: Path | None) -> int | None:
    if pid_file is None or not pid_file.is_file():
        return None
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as exc:
        raise MonitorError(f"invalid runner PID file {pid_file}") from exc
    if pid <= 0:
        raise MonitorError(f"invalid runner PID in {pid_file}")
    return pid


def process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def monitor(args: argparse.Namespace) -> int:
    manifest_path = args.manifest.resolve()
    state_path = (
        args.state_file.resolve()
        if args.state_file is not None
        else manifest_path.parent / ".discord-monitor-state.json"
    )
    webhook_url = load_webhook_url(args.webhook_env, args.webhook_url_file)
    state = read_json(state_path) if state_path.is_file() else {}
    host = socket.gethostname()
    while True:
        snapshot = load_snapshot(manifest_path)
        runner_pid = read_runner_pid(args.pid_file)
        runner_exited = bool(
            runner_pid is not None
            and not process_is_running(runner_pid)
            and (snapshot is None or snapshot.status not in TERMINAL_STATUSES)
        )
        snapshot_signature = (
            snapshot.signature if snapshot is not None else "manifest-missing"
        )
        signature = f"{args.label}:{snapshot_signature}"
        if runner_exited:
            signature += ":runner-exited"
        now = time.time()
        last_sent = float(state.get("last_sent_epoch", 0.0))
        heartbeat_due = bool(
            args.heartbeat_seconds > 0
            and last_sent
            and now - last_sent >= args.heartbeat_seconds
        )
        changed = signature != state.get("last_signature")
        should_send = changed or heartbeat_due or not last_sent
        if should_send:
            message = format_message(
                snapshot,
                host=host,
                label=args.label,
                heartbeat=heartbeat_due and not changed,
                runner_exited=runner_exited,
            )
            try:
                send_discord_message(
                    webhook_url,
                    message,
                    username=args.username,
                    attempts=args.send_attempts,
                )
            except MonitorError as exc:
                print(
                    f"[{utc_now()}] warning: {exc}; will retry",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                state = {
                    "last_signature": signature,
                    "last_sent_at": utc_now(),
                    "last_sent_epoch": now,
                }
                atomic_write_json(state_path, state)
                print(f"[{utc_now()}] Discord notification sent", flush=True)
        terminal = snapshot is not None and snapshot.status in TERMINAL_STATUSES
        delivered = state.get("last_signature") == signature
        if args.once:
            return 0 if delivered else 1
        if terminal and delivered:
            return 1 if snapshot.status == "failed" else 0
        if runner_exited and delivered:
            return 1
        time.sleep(args.interval_seconds)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    default_repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=default_repo / "output-performance-folder-mac-v2/_matrix_manifest.json",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--webhook-url-file",
        type=Path,
        help="file containing only the Discord webhook URL",
    )
    source.add_argument(
        "--webhook-env",
        default="DISCORD_WEBHOOK_URL",
        help="environment variable containing the webhook URL (default: %(default)s)",
    )
    parser.add_argument(
        "--pid-file", type=Path, help="optional training-runner PID file"
    )
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--interval-seconds", type=float, default=300.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=3600.0)
    parser.add_argument("--send-attempts", type=int, default=5)
    parser.add_argument("--username", default="MAC Training Monitor")
    parser.add_argument("--label", default="MAC training matrix")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")
    if args.heartbeat_seconds < 0:
        parser.error("--heartbeat-seconds must be non-negative")
    if args.send_attempts <= 0:
        parser.error("--send-attempts must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return monitor(parse_args(argv))
    except MonitorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("monitor interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
