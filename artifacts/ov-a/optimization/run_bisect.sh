#!/usr/bin/env bash
# Bisect which lowering group breaks a token variant: run_bisect.sh <T> <group>
set -euo pipefail
T=$1; GROUP=$2
ROOT=/workflow/tokens-$T
PLUGIN=custom-plugin-tokens-r11
cd /workflow
source /workflow/framework-env.sh
export OV_DEVICE=GPU.0
export LD_LIBRARY_PATH="/workflow/$PLUGIN:/workflow/level-zero/usr/lib/x86_64-linux-gnu:/workflow/venv/lib/python3.12/site-packages/openvino/libs:${LD_LIBRARY_PATH:-}"
P=$((768 + T))
export PI05_PREFIX_TOKENS=$P
export PI05_DQ_REUSE=1
python3 - "$GROUP" <<'EOF' > /tmp/bisect-env.sh
import json, sys, os
t = json.load(open(f"/workflow/optimization/tokens{os.environ['PI05_PREFIX_TOKENS'] and (int(os.environ['PI05_PREFIX_TOKENS'])-768)}-r11-tuning.json"))
group = sys.argv[1]
keep = set()
if "q" in group: keep |= {k for k in t if any(s in k for s in ("RMS_QUANTIZE","MVN_","ADALN","LARGE_QUANTIZE","TRANSPOSE_QUANTIZE","VISION_QUANTIZE"))}
if "r" in group: keep |= {k for k in t if any(s in k for s in ("QKV_SPLIT","PACKED_ROPE","QKV_ROPE","VISION_QKV_VIEWS"))}
if "g" in group: keep |= {k for k in t if "GATEUP" in k}
if "s" in group: keep |= {k for k in t if "SDPA" in k}
if "p" in group: keep |= {k for k in t if "PREFETCH" in k}
if "x" in group: keep |= {k for k in t if "QKV_SPLIT" in k}
if "y" in group: keep |= {k for k in t if "PACKED_ROPE" in k}
if "z" in group: keep |= {k for k in t if "QKV_ROPE" in k and "KV" not in k}
if "k" in group: keep |= {k for k in t if "QKV_ROPE" in k}
if "v" in group: keep |= {k for k in t if "VISION_QKV_VIEWS" in k}
for k in sorted(keep):
    print(f"export {k}={json.dumps(t[k])}")
EOF
source /tmp/bisect-env.sh
env | grep ^PI05_ | cut -c1-80
exec /workflow/venv/bin/python optimization/profile_openvino.py --root $ROOT --model $ROOT/model/pi05_droid_dynamic_w8a8.xml \
  --output $ROOT/results/bisect-$GROUP --profile 0 --iterations 1 --warmups 1 \
  --plugins /workflow/$PLUGIN/plugins.xml --cache-tag bisect-t$T-$GROUP \
  --fuse-qkv vision,language,expert --batch-vision --align-vision-mlp --fuse-vision-padding \
  --config '{"DYNAMIC_QUANTIZATION_GROUP_SIZE": "18446744073709551615"}' \
  --validation-samples zero,random
