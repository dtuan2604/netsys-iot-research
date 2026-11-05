#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

ROCM_VENV="${IOT_ROCM_VENV:-${HOME}/venvs/iot-rocm}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output-performance-cnn-mac-v2}"
STOP_FILE="${STOP_FILE:-training-matrix-cnn-v2.stop}"
PID_FILE="${PID_FILE:-training-matrix-cnn-v2.pid}"

if [[ -e "$STOP_FILE" ]]; then
  echo "error: stop file already exists: $STOP_FILE" >&2
  echo "remove it before starting or resuming the CNN matrix" >&2
  exit 1
fi
if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(<"$PID_FILE")"
  if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "error: CNN matrix appears to be running with PID $existing_pid" >&2
    exit 1
  fi
fi
if [[ ! -f "$ROCM_VENV/bin/activate" ]]; then
  echo "error: ROCm virtual environment not found: $ROCM_VENV" >&2
  echo "set IOT_ROCM_VENV if it is installed elsewhere" >&2
  exit 1
fi

# Non-interactive/nohup shells do not always load ~/.bashrc.
export HSA_ENABLE_DXG_DETECTION="${HSA_ENABLE_DXG_DETECTION:-1}"
export LD_LIBRARY_PATH="/opt/rocm-7.2.1/lib:/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
source "$ROCM_VENV/bin/activate"
export PYTHONUNBUFFERED=1

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("ROCm GPU is unavailable; CNN matrix was not started")
probe = torch.ones(1, device="cuda") + 1
torch.cuda.synchronize()
print(f"GPU preflight passed: {torch.cuda.get_device_name(0)} ({probe.item():.0f})")
PY

printf '%s\n' "$$" > "$PID_FILE"
exec python code_v2/run_training_matrix.py \
  --models cnn-pn-rf \
  --cnn-device cuda \
  --max-concurrent-experiments 1 \
  --output-root "$OUTPUT_ROOT" \
  --stop-file "$STOP_FILE" \
  "$@"
