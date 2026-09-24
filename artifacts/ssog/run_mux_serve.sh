#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
IMAGE=${IMAGE:-ssog-b580:neo2513}
PCI=${PCI:-0000:03:00.0}
CARD="$(readlink -f "/dev/dri/by-path/pci-${PCI}-card")"
RENDER="$(readlink -f "/dev/dri/by-path/pci-${PCI}-render")"
RGID="$(getent group render | cut -d: -f3)"
VGID="$(getent group video | cut -d: -f3)"
[[ -c "$CARD" && -c "$RENDER" ]] || { echo "No B580 device nodes for $PCI" >&2; exit 3; }
exec docker run --rm -i --entrypoint bash --name "pi05-mux-$$" \
  --device "$CARD" --device "$RENDER" --group-add "$VGID" --group-add "$RGID" \
  --ulimit memlock=8388608:8388608 \
  -v "$ROOT":/work -v /dev/shm:/dev/shm -w /work \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 -e SYCL_CACHE_PERSISTENT=0 \
  "$IMAGE" -lc '
    source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
    n=$(sycl-ls 2>/dev/null | grep -c level_zero)
    [ "$n" -ge 1 ] || { echo "No Level-Zero GPU visible" >&2; exit 3; }
    exec flock /work/gpu0.lock flock /work/.gpu0.lock \
      /work/muxfinal/bin/fused_mux_droid \
      --weights /work/campaign13/share/weights-A \
      --ref /work/muxfinal/shim/dk \
      --ref-128 /work/muxfinal/shim/dk128 \
      --ref-256 /work/muxfinal/shim/dk256 \
      --case 0 --text-valid 58 --graph 1 --serve
  '
