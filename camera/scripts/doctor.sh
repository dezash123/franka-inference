#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
OUT="$ROOT/evidence/doctor-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"
uname -a > "$OUT/kernel.txt"
cat /etc/os-release > "$OUT/os-release.txt"
lsusb > "$OUT/lsusb.txt"
lsusb -t > "$OUT/usb-topology.txt"
dpkg-query -W -f='${Package}\t${Version}\t${Architecture}\n' python3 python3-opencv python3-numpy libopencv-videoio406 libopencv-imgcodecs406 > "$OUT/packages.tsv"
systemctl --user --no-pager status zed-camera-viewer.service > "$OUT/service.txt" || true
journalctl --user -u zed-camera-viewer.service --no-pager -n 80 > "$OUT/service-log.txt"
curl --fail --silent http://127.0.0.1:8765/api/status > "$OUT/status.json"
printf 'Saved diagnostics: %s\n' "$OUT"
cat "$OUT/status.json"
