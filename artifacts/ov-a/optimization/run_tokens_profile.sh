#!/usr/bin/env bash
set -euo pipefail
T=$1; ITER=${2:-10}
ROOT=/workflow/tokens-$T
PLUGIN=custom-plugin-tokens-r11
cd /workflow
source /workflow/framework-env.sh
export OV_DEVICE=GPU.0
export LD_LIBRARY_PATH="/workflow/$PLUGIN:/workflow/level-zero/usr/lib/x86_64-linux-gnu:/workflow/venv/lib/python3.12/site-packages/openvino/libs:${LD_LIBRARY_PATH:-}"
exec /workflow/venv/bin/python optimization/run_integrated.py --root $ROOT --variant gemm --tag "tokens$T-r11" --iterations $ITER --profile 1 \
  --plugins /workflow/$PLUGIN/plugins.xml --gateup-folder /workflow/optimization/gateup-probes/prefix-geometry-w2x4 \
  --tuning-file /workflow/optimization/tokens$T-r11-tuning.json
