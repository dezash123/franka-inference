#!/bin/bash
set -e
cd /home/spring/robo_run/franka_droid
source ./env.sh
export XDG_RUNTIME_DIR=/run/user/1000
# This capture needs exclusive device access. Restore the existing preview.
viewer_was_running=false
if systemctl --user is-active --quiet zed-camera-viewer.service; then
  viewer_was_running=true
  systemctl --user stop zed-camera-viewer.service
fi
cleanup() { if "$viewer_was_running"; then systemctl --user start zed-camera-viewer.service; fi; }
trap cleanup EXIT
timeout 35 python check_cameras.py "$@"
