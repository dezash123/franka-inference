#!/usr/bin/env bash
# Token-variant runs. usage: run_tokens.sh <T> <stage> [iterations]
#   stage=stock     : stock OpenVINO GPU plugin, group-128 dynamic W8A8, no fusions
#   stage=reference : custom plugin, per-token quantization + graph fusions, stock kernels
#                     (establishes custom-per-token-fusedpad reference outputs)
#   stage=optimized : full selected kernel stack with the variant tuning file
set -euo pipefail
T=$1; STAGE=$2; ITER=${3:-100}
ROOT=/workflow/tokens-$T
PLUGIN=custom-plugin-tokens-r11
cd /workflow
source /workflow/framework-env.sh
export OV_DEVICE=GPU.0
export LD_LIBRARY_PATH="/workflow/$PLUGIN:/workflow/level-zero/usr/lib/x86_64-linux-gnu:/workflow/venv/lib/python3.12/site-packages/openvino/libs:${LD_LIBRARY_PATH:-}"
PY=/workflow/venv/bin/python
SAMPLES=zero,random,validation-0,validation-1,validation-2,validation-3,validation-4,validation-5
case "$STAGE" in
  stock)
    exec $PY optimization/profile_openvino.py --root $ROOT --model $ROOT/model/pi05_droid_dynamic_w8a8.xml \
      --output $ROOT/results/stock-dynamic-w8a8 --profile 0 --iterations $ITER --cache-tag stock-t$T \
      --validation-samples $SAMPLES
    ;;
  reference)
    export PI05_PREFIX_TOKENS=$((768 + T))
    exec $PY optimization/profile_openvino.py --root $ROOT --model $ROOT/model/pi05_droid_dynamic_w8a8.xml \
      --output $ROOT/results/custom-per-token-fusedpad --profile 0 --iterations $ITER \
      --plugins /workflow/$PLUGIN/plugins.xml --cache-tag reference-t$T \
      --fuse-qkv vision,language,expert --batch-vision --align-vision-mlp --fuse-vision-padding \
      --config '{"DYNAMIC_QUANTIZATION_GROUP_SIZE": "18446744073709551615"}' \
      --validation-samples $SAMPLES
    ;;
  optimized)
    exec $PY optimization/run_integrated.py --root $ROOT --variant gemm --tag "tokens$T-r11" --iterations $ITER \
      --plugins /workflow/$PLUGIN/plugins.xml \
      --gateup-folder /workflow/optimization/gateup-probes/prefix-geometry-w2x4 \
      --tuning-file /workflow/optimization/tokens$T-r11-tuning.json
    ;;
esac
