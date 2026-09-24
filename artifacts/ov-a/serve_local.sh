#!/usr/bin/env bash
# The shipped multi-window OpenVINO W8A8 server, relocatable.
#
#   ./serve_local.sh                 # windows 64,256 on $OV_DEVICE (default GPU)
#   OVMUX_WINDOWS=64 ./serve_local.sh
#   OV_DEVICE=GPU.1 OVMUX_CLOCK_POLICY=min_freq=2850 ./serve_local.sh
#
# It speaks JSON-lines on stdin/stdout (one chunk per line; `{"ready":true,...}`
# first).  One worker process per window: the plugin latches
# PI05_PREFIX_TOKENS in a process-local static.
#
# Each resident window costs 4.28 GB of VRAM.  A 12 GB B580 has 11.33 GB
# usable, so all three (12.85 GB) do NOT fit -- they compile and then the first
# inference dies with CL_OUT_OF_RESOURCES.  The shipped resident set is
# 64 + 256 = 8.57 GB, which serves every prompt up to 256 tokens (a 65..128
# token prompt runs on the 256 geometry).
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$ROOT/env-local.sh"
WINDOWS="${OVMUX_WINDOWS:-64,256}"
CACHE="${OVMUX_CACHE:-$ROOT/runs/mux-cache}"
mkdir -p "$(dirname "$CACHE")"

# The three tuning JSONs pin absolute kernel-source paths; resolve them here.
RT="$ROOT/runs/mux-tuning"; mkdir -p "$RT"
for w in 64 128 256; do
  src="$ROOT/ovmux/c2t$w-r11-tuning.json"
  [ -f "$src" ] && sed "s#/workflow#$ROOT#g" "$src" > "$RT/c2t$w-r11-tuning.json"
done

args=()
case ",$WINDOWS," in *,64,*)  args+=(--ir64  "$ROOT/model/t64/pi05_droid_dynamic_w8a8.xml"  --tuning64  "$RT/c2t64-r11-tuning.json");;  esac
case ",$WINDOWS," in *,128,*) args+=(--ir128 "$ROOT/model/t128/pi05_droid_dynamic_w8a8.xml" --tuning128 "$RT/c2t128-r11-tuning.json");; esac
case ",$WINDOWS," in *,256,*) args+=(--ir256 "$ROOT/model/t256/pi05_droid_dynamic_w8a8.xml" --tuning256 "$RT/c2t256-r11-tuning.json");; esac

exec "$PI05_PYTHON" "$ROOT/ovmux/ov_mux_server.py" \
  "${args[@]}" \
  --plugin "$PI05_PLUGINS" \
  --opt-dir "$ROOT/optimization" \
  --cache-dir "$CACHE" \
  --device "$OV_DEVICE" \
  --clock-policy "${OVMUX_CLOCK_POLICY:-unset}"
