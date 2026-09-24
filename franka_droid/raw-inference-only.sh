#!/bin/bash
set -e
cd /home/spring/robo_run/franka_droid
source ./env.sh
if [ "$(id -u)" != 0 ]; then echo 'Root required for read-only FCI realtime access.' >&2; exit 1; fi
viewer_was_running=false
if runuser -u spring -- env XDG_RUNTIME_DIR=/run/user/1000 systemctl --user is-active --quiet zed-camera-viewer.service; then
 viewer_was_running=true
 runuser -u spring -- env XDG_RUNTIME_DIR=/run/user/1000 systemctl --user stop zed-camera-viewer.service
fi
cleanup() { if "$viewer_was_running"; then runuser -u spring -- env XDG_RUNTIME_DIR=/run/user/1000 systemctl --user start zed-camera-viewer.service; fi; }
trap cleanup EXIT
python -u raw_inference_only.py
