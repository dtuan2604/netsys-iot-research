#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

OUTPUT_ROOT="${OUTPUT_ROOT:-output-performance-decision-tree-mac-v2}"
STOP_FILE="${STOP_FILE:-training-matrix-decision-tree-v2.stop}"
PID_FILE="${PID_FILE:-training-matrix-decision-tree-v2.pid}"

if [[ -e "$STOP_FILE" ]]; then
  echo "error: stop file already exists: $STOP_FILE" >&2
  echo "remove it before starting or resuming the decision-tree matrix" >&2
  exit 1
fi
if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(<"$PID_FILE")"
  if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "error: decision-tree matrix appears to be running with PID $existing_pid" >&2
    exit 1
  fi
fi

if [[ -n "${CLASSIFICATION_VENV:-}" ]]; then
  if [[ ! -f "$CLASSIFICATION_VENV/bin/activate" ]]; then
    echo "error: CLASSIFICATION_VENV is invalid: $CLASSIFICATION_VENV" >&2
    exit 1
  fi
  source "$CLASSIFICATION_VENV/bin/activate"
fi
if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="$PYTHON_BIN"
else
  for candidate in python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && \
      "$candidate" -c 'import joblib, numpy, pandas, pyarrow, sklearn' \
        >/dev/null 2>&1; then
      python_bin="$candidate"
      break
    fi
  done
fi
if [[ -z "${python_bin:-}" ]]; then
  echo "error: no Python interpreter with the classification dependencies was found" >&2
  echo "set CLASSIFICATION_VENV or PYTHON_BIN to the server environment" >&2
  exit 127
fi

"$python_bin" -c 'import joblib, numpy, pandas, pyarrow, sklearn'
export PYTHONUNBUFFERED=1

printf '%s\n' "$$" > "$PID_FILE"
exec "$python_bin" code_v2/run_training_matrix.py \
  --models decision-tree \
  --max-concurrent-experiments 4 \
  --decision-tree-jobs 3 \
  --output-root "$OUTPUT_ROOT" \
  --stop-file "$STOP_FILE" \
  "$@"
