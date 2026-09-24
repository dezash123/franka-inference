#!/usr/bin/env bash
set -euo pipefail
cd /workflow; source /workflow/framework-env.sh; export OV_DEVICE=GPU.0
export LD_LIBRARY_PATH="/workflow/custom-plugin-cams-r11:/workflow/level-zero/usr/lib/x86_64-linux-gnu:/workflow/venv/lib/python3.12/site-packages/openvino/libs:${LD_LIBRARY_PATH:-}"
exec /workflow/venv/bin/python optimization/run_integrated.py --root /workflow/c2t64 --variant gemm --tag c2t64-r11-prof --iterations 10 --profile 1 \
  --plugins /workflow/custom-plugin-cams-r11/plugins.xml --gateup-folder /workflow/optimization/gateup-probes/prefix-geometry-w2x4 --tuning-file /workflow/optimization/c2t64-r11-tuning.json
