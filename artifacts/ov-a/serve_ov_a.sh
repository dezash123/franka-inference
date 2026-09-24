#!/usr/bin/env bash
# The resident OpenVINO W8A8 server for CHECKPOINT A (openpi pi05_droid_jointpos,
# joint POSITION) on rome, in the shape sprilicim/unit_ov.py expects from a
# --serve-launch:
#
#     bash serve_ov_a.sh <gpu index>
#
# One positional argument, JSON-lines protocol on stdin/stdout, the stack reported
# on the first line.  It is a thin wrapper over the relocatable serve_local.sh in
# this directory, which owns the tuning retarget (the shipped JSONs hardcode
# /workflow kernel-source paths), plugins.local.xml, --opt-dir and the compile
# cache.  This file only adds the single-card guard and a truthful clock label.
#
#   windows   64 + 256 resident (4.58 + 4.61 GB on the 11.33 GiB B580); all three do
#             not fit.  A 65..256-token prompt runs on the 256 geometry; above 256 the
#             server refuses the chunk, it never truncates.
#   model/    model/t64 and model/t256 hold the CHECKPOINT-A IRs (see ../README.md for
#             the pinned sha256s); the XML at T=64 is byte-identical to the checkpoint-B
#             export, the .bin is not.
#   card      rome has one Intel GPU, which OpenVINO enumerates as "GPU"; the index is
#             accepted for protocol compatibility and must be 0.
#   clock     nothing here pins min_freq.  The label on the ready line is READ from
#             sysfs (the B580 at PCI 0000:03:00.0, whatever card number it enumerates
#             as) so the receipt records the floor that was actually in force.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${1:-0}"
case "$GPU" in
  0) ;;
  ''|*[!0-9]*) echo "usage: serve_ov_a.sh <gpu index>" >&2; exit 2 ;;
  *) echo "serve_ov_a.sh: rome has a single B580 (OpenVINO device GPU); index $GPU does not exist" >&2; exit 2 ;;
esac
for w in 64 256; do
  ir="$ROOT/model/t$w/pi05_droid_dynamic_w8a8.xml"
  [ -f "$ir" ] && [ -f "${ir%.xml}.bin" ] || { echo "serve_ov_a.sh: missing checkpoint-A IR $ir" >&2; exit 3; }
done
PCI="${PCI:-0000:03:00.0}"
CARD="$(basename "$(readlink -f "/dev/dri/by-path/pci-${PCI}-card" 2>/dev/null || echo card0)")"
FREQ="/sys/class/drm/$CARD/device/tile0/gt0/freq0/min_freq"
CLOCK="min_freq=$(cat "$FREQ" 2>/dev/null || echo unknown)"
exec env \
  OV_DEVICE="${OV_DEVICE:-GPU}" \
  OVMUX_WINDOWS="${OVMUX_WINDOWS:-64,256}" \
  OVMUX_CACHE="${OVMUX_CACHE:-$ROOT/runs/serve/cache}" \
  OVMUX_CLOCK_POLICY="${OVMUX_CLOCK_POLICY:-$CLOCK}" \
  bash "$ROOT/serve_local.sh"
