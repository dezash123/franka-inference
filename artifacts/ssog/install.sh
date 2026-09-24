#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
ROOT=${1:-"$HOME/pi05-ssog-a-mc2-mux"}
BASE="$HERE/pi05-b580-ssog-a-mc2.tar.zst"
OVERLAY="$HERE/pi05-ssog-a-mc2-mux-overlay.tar.zst"
[[ -f "$BASE" && -f "$OVERLAY" ]] || { echo "Missing base or overlay archive" >&2; exit 2; }
sha256sum -c "$HERE/SHA256SUMS"
rm -rf "$ROOT.installing"
mkdir -p "$ROOT.installing/base"
tar --zstd -xf "$BASE" -C "$ROOT.installing/base"
PKG="$ROOT.installing/base/pi05-b580-ssog-a-mc2-package"
[[ -d "$PKG/weights-A" && -d "$PKG/share/ref" && -d "$PKG/runtime/env" ]] || { echo "Unexpected base archive layout" >&2; exit 3; }
mkdir -p "$ROOT.installing/campaign13/share"
mv "$PKG/weights-A" "$ROOT.installing/campaign13/share/weights-A"
mv "$PKG/share/ref" "$ROOT.installing/campaign13/share/ref"
mv "$PKG/runtime" "$ROOT.installing/runtime"
[[ -d "$PKG/unit" ]] && mv "$PKG/unit" "$ROOT.installing/unit"
tar --zstd -xf "$OVERLAY" -C "$ROOT.installing"
cp "$HERE/CANONICAL.sha256" "$ROOT.installing/campaign13/RuntimeCore/CANONICAL.sha256"
rm -rf "$ROOT.installing/base"
chmod +x "$ROOT.installing/run_mux_serve.sh" "$ROOT.installing/smoke_mux.sh"
printf '%s  %s\n' \
  db89756ca6b0451040cbbf6e3490d28606d5ed510de3d4cc371bfce3259d6edb "$ROOT.installing/muxfinal/bin/fused_mux_droid" \
  0376a60ab7c1833c6dcdfa7bd04281a8c1f0a2804e8c45743738a2c5001ff103 "$ROOT.installing/campaign13/RuntimeCore/CANONICAL.sha256" \
  | sha256sum -c -
rm -rf "$ROOT.previous"
[[ -e "$ROOT" ]] && mv "$ROOT" "$ROOT.previous"
mv "$ROOT.installing" "$ROOT"
docker image inspect ssog-b580:neo2513 >/dev/null 2>&1 \
  || docker build -t ssog-b580:neo2513 "$ROOT/runtime/env"
echo "Installed at $ROOT"
echo "Run: $ROOT/smoke_mux.sh"
