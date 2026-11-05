#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 {cnn|decision-tree} [runner options]" >&2
  exit 2
fi

ROLE="$1"
shift
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

case "$ROLE" in
  cnn)
    export OUTPUT_ROOT="${OUTPUT_ROOT:-smoke-output-performance-cnn-mac}"
    export STOP_FILE="${STOP_FILE:-training-matrix-smoke-cnn.stop}"
    export PID_FILE="${PID_FILE:-training-matrix-smoke-cnn.pid}"
    exec "$REPO_ROOT/run-cnn-wsl.sh" --smoke --overwrite "$@"
    ;;
  decision-tree)
    export OUTPUT_ROOT="${OUTPUT_ROOT:-smoke-output-performance-decision-tree-mac}"
    export STOP_FILE="${STOP_FILE:-training-matrix-smoke-decision-tree.stop}"
    export PID_FILE="${PID_FILE:-training-matrix-smoke-decision-tree.pid}"
    exec "$REPO_ROOT/run-decision-tree-server.sh" --smoke --overwrite "$@"
    ;;
  *)
    echo "error: role must be 'cnn' or 'decision-tree'" >&2
    exit 2
    ;;
esac
