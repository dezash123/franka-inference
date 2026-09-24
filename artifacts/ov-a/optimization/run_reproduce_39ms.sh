#!/usr/bin/env bash
# Reproduce the archived spring-edge event-queue configuration on dstack.
set -euo pipefail
source /workflow/framework-env.sh
export OV_DEVICE="${OV_DEVICE:-GPU}"
export LD_LIBRARY_PATH="/workflow/custom-plugin-event-queue-r6:/workflow/venv/lib/python3.12/site-packages/openvino/libs:${LD_LIBRARY_PATH:-}"
cd /workflow
sha256sum -c reproduce-39ms-SHA256SUMS
(
  cd model
  sha256sum -c SHA256SUMS
)
/workflow/venv/bin/python /workflow/optimization/generate_validation_cases.py
trial_tag="${1:-reproduce-39ms-r9}"
for trial in 1 2 3; do
  /workflow/venv/bin/python /workflow/optimization/run_integrated.py \
    --variant gemm --tag "$trial_tag" --iterations 100 \
    --result-suffix "repeat-${trial}" \
    --plugins /workflow/custom-plugin-event-queue-r6/plugins.xml \
    --gateup-folder /workflow/prefix-geometry-resume/prefix-geometry-w2x4 \
    --tuning-file /workflow/optimization/reproduce-39ms-tuning.json
done
