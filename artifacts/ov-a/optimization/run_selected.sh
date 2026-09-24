#!/usr/bin/env bash
# Run only on the persistent dstack B580 runner.
set -euo pipefail
source /workflow/framework-env.sh
export LD_LIBRARY_PATH="/workflow/custom-plugin-event-queue-r6:/workflow/venv/lib/python3.12/site-packages/openvino/libs:${LD_LIBRARY_PATH:-}"
exec /workflow/venv/bin/python /workflow/optimization/run_integrated.py \
  --variant gemm --tag "${1:-selected-inorder-21289041}" \
  --plugins /workflow/custom-plugin-event-queue-r6/plugins.xml \
  --gateup-folder /workflow/prefix-geometry-resume/prefix-geometry-w2x4 \
  --tuning-file /workflow/optimization/selected-tuning.json
