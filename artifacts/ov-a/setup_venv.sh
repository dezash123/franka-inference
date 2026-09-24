#!/usr/bin/env bash
# Create the Python environment this line needs, inside the tree.
#   ./setup_venv.sh            # offline, from the bundled wheels/
#   PI05_ONLINE=1 ./setup_venv.sh   # from PyPI (same pinned versions)
# Requires python3.12 (the OpenVINO wheel pinned here is cp312).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PI05_PYTHON_BIN:-python3}"
"$PY" -c 'import sys; assert sys.version_info[:2]==(3,12), sys.version' \
  || { echo "need python3.12 (set PI05_PYTHON_BIN)"; exit 1; }
rm -rf "$ROOT/venv"
"$PY" -m venv "$ROOT/venv"
PIN=(numpy==2.2.6 openvino==2026.3.1 safetensors==0.8.0)
if [ "${PI05_ONLINE:-0}" = "1" ]; then
  "$ROOT/venv/bin/pip" install --disable-pip-version-check -q "${PIN[@]}"
else
  "$ROOT/venv/bin/pip" install --disable-pip-version-check -q --no-index \
    --find-links "$ROOT/wheels" "${PIN[@]}"
fi
"$ROOT/venv/bin/python" - <<'PY'
import numpy, openvino, safetensors
print("numpy", numpy.__version__)
print("openvino", openvino.get_version())
print("safetensors", safetensors.__version__)
print("devices", openvino.Core().available_devices)
PY
