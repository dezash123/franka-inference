#!/usr/bin/env bash
# One optimized c2t64 run on a chosen lane. usage: run_c2t64_lane.sh <device> <root> <tag> <tuning> <plugin> [gateup-folder] [iters]
set -euo pipefail
DEV=$1; R=$2; TAG=$3; TUNING=$4; PLUGIN=$5; GF=${6:-/workflow/optimization/gateup-probes/prefix-geometry-w2x4}; ITER=${7:-100}
cd /workflow
source /workflow/framework-env.sh
export OV_DEVICE=$DEV
export LD_LIBRARY_PATH="/workflow/$PLUGIN:/workflow/level-zero/usr/lib/x86_64-linux-gnu:/workflow/venv/lib/python3.12/site-packages/openvino/libs:${LD_LIBRARY_PATH:-}"
exec /workflow/venv/bin/python optimization/run_integrated.py --root $R --variant gemm --tag "$TAG" --iterations $ITER \
  --plugins /workflow/$PLUGIN/plugins.xml --gateup-folder "$GF" --tuning-file /workflow/optimization/$TUNING ${LANE_FLOAT_GATE:+--float-gate $LANE_FLOAT_GATE} ${LANE_ALLOW_ROUNDOFF:+--allow-roundoff}
