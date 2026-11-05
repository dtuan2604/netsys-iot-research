#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 {cnn|decision-tree} [monitor options]" >&2
  exit 2
fi

ROLE="$1"
shift
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

case "$ROLE" in
  cnn)
    OUTPUT_ROOT="${OUTPUT_ROOT:-output-performance-cnn-mac}"
    PID_FILE="${PID_FILE:-training-matrix-cnn.pid}"
    LABEL="CNN-PN-RF · WSL GPU"
    USERNAME="CNN-PN-RF Monitor"
    ;;
  decision-tree)
    OUTPUT_ROOT="${OUTPUT_ROOT:-output-performance-decision-tree-mac}"
    PID_FILE="${PID_FILE:-training-matrix-decision-tree.pid}"
    LABEL="Decision Tree · Ubuntu Server"
    USERNAME="Decision Tree Monitor"
    ;;
  *)
    echo "error: role must be 'cnn' or 'decision-tree'" >&2
    exit 2
    ;;
esac

if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="$PYTHON_BIN"
else
  for candidate in python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      python_bin="$candidate"
      break
    fi
  done
fi
if [[ -z "${python_bin:-}" ]]; then
  echo "error: no Python interpreter found" >&2
  exit 127
fi

webhook_args=()
if [[ -n "${WEBHOOK_URL_FILE:-}" ]]; then
  webhook_args=(--webhook-url-file "$WEBHOOK_URL_FILE")
elif [[ -f .discord-webhook-url ]]; then
  webhook_args=(--webhook-url-file .discord-webhook-url)
elif [[ -z "${DISCORD_WEBHOOK_URL:-}" ]]; then
  echo "error: create .discord-webhook-url, set WEBHOOK_URL_FILE, or set DISCORD_WEBHOOK_URL" >&2
  exit 1
else
  webhook_args=(--webhook-env DISCORD_WEBHOOK_URL)
fi

exec "$python_bin" code/monitor_training_matrix.py \
  --manifest "$OUTPUT_ROOT/_matrix_manifest.json" \
  --pid-file "$PID_FILE" \
  --state-file "$OUTPUT_ROOT/.discord-monitor-state.json" \
  --username "$USERNAME" \
  --label "$LABEL" \
  "${webhook_args[@]}" \
  "$@"
