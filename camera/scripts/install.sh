#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if ! /usr/bin/python3 -c 'import cv2, numpy' >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends python3-opencv python3-numpy
fi
if ! command -v cc >/dev/null 2>&1 || [ ! -f /usr/include/linux/videodev2.h ]; then
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends build-essential linux-libc-dev
fi
bash "$ROOT/scripts/build.sh"
if ! id -nG | tr ' ' '\n' | grep -qx video; then
  echo 'Camera access requires the video group. Run: sudo usermod -aG video "$USER", then log out and back in.' >&2
  exit 1
fi
mkdir -p "$HOME/.config/systemd/user" "$ROOT/evidence"
cat > "$HOME/.config/systemd/user/zed-camera-viewer.service" <<EOF
[Unit]
Description=Camera browser viewer (ZED and RealSense, loopback)
After=default.target

[Service]
Type=simple
WorkingDirectory=$ROOT
ExecStart=/usr/bin/python3 $ROOT/viewer.py --bind 127.0.0.1 --port 8765 --resolution auto --fps 15
Restart=on-failure
RestartSec=3
TimeoutStopSec=5
NoNewPrivileges=true
UMask=0077

[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload
systemctl --user enable --now zed-camera-viewer.service
systemctl --user --no-pager status zed-camera-viewer.service
