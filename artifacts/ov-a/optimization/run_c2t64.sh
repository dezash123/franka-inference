#!/usr/bin/env bash
# 2-camera, 64-token workload. usage: run_c2t64.sh <stage> [iterations] [tag-suffix]
#   stage=reference : per-token quantization + fusions, stock kernels (bitwise reference)
#   stage=optimized : full stack with c2t64-r11-tuning.json
set -euo pipefail
STAGE=$1; ITER=${2:-100}; SUFFIX=${3:-}
R=/workflow/c2t64
PLUGIN=${PLUGIN:-custom-plugin-cams-r11}
TUNING=${TUNING:-c2t64-r11-tuning.json}
cd /workflow
source /workflow/framework-env.sh
export OV_DEVICE=GPU.0
export LD_LIBRARY_PATH="/workflow/$PLUGIN:/workflow/level-zero/usr/lib/x86_64-linux-gnu:/workflow/venv/lib/python3.12/site-packages/openvino/libs:${LD_LIBRARY_PATH:-}"
PY=/workflow/venv/bin/python
SAMPLES=zero,random,validation-0,validation-1,validation-2,validation-3,validation-4,validation-5
case "$STAGE" in
  reference)
    export PI05_PREFIX_TOKENS=576 PI05_CAMERAS=2
    exec $PY optimization/profile_openvino.py --root $R --model $R/model/pi05_droid_dynamic_w8a8.xml \
      --output $R/results/custom-per-token-fusedpad --profile 0 --iterations $ITER \
      --plugins /workflow/$PLUGIN/plugins.xml --cache-tag reference-c2t64 \
      --fuse-qkv vision,language,expert --batch-vision --align-vision-mlp --fuse-vision-padding \
      --config '{"DYNAMIC_QUANTIZATION_GROUP_SIZE": "18446744073709551615"}' \
      --validation-samples $SAMPLES
    ;;
  optimized)
    exec $PY optimization/run_integrated.py --root $R --variant gemm --tag "c2t64-r11$SUFFIX" --iterations $ITER \
      --plugins /workflow/$PLUGIN/plugins.xml \
      --gateup-folder /workflow/optimization/gateup-probes/prefix-geometry-w2x4 \
      --tuning-file /workflow/optimization/$TUNING
    ;;
esac
