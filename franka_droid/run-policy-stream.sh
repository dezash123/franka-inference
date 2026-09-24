#!/bin/bash
set -e
cd /home/spring/robo_run/franka_droid
if [ "$(id -u)" != 0 ]; then echo 'Run as root for realtime scheduling.' >&2; exit 1; fi
# Enter a top-level slice so normal system/user cpusets cannot restrict the RT worker.
if ! grep -q '^0::/franka.slice/' /proc/self/cgroup; then
  unit="franka-droid-run-$(date +%s%N)"
  stop_child() { systemctl stop "$unit.service" >/dev/null 2>&1 || true; }
  trap stop_child EXIT INT TERM
  systemd-run --quiet --wait --pipe --collect --unit="$unit" --slice=franka.slice \
    --property=Type=exec --property=CPUAffinity='0-3 6-9' \
    --property=LimitRTPRIO=99 --property=LimitMEMLOCK=infinity \
    --property=KillSignal=SIGINT --property=TimeoutStopSec=5 --property=RuntimeMaxSec=infinity \
    /bin/bash "$PWD/run-policy-stream.sh" "$@" &
  runner=$!
  result=0
  wait "$runner" || result=$?
  stop_child
  trap - EXIT INT TERM
  exit "$result"
fi
/usr/local/sbin/franka-realtime-setup
set -a
source /etc/franka/realtime.env
set +a
source ./env.sh
if systemctl is-active --quiet franka-ros-state.service; then echo 'ROS hardware service owns FCI.' >&2; exit 1; fi
viewer_was_running=false
if runuser -u spring -- env XDG_RUNTIME_DIR=/run/user/1000 systemctl --user is-active --quiet zed-camera-viewer.service; then
  viewer_was_running=true
  runuser -u spring -- env XDG_RUNTIME_DIR=/run/user/1000 systemctl --user stop zed-camera-viewer.service
fi
cleanup() { if "$viewer_was_running"; then runuser -u spring -- env XDG_RUNTIME_DIR=/run/user/1000 systemctl --user start zed-camera-viewer.service; fi; }
trap cleanup EXIT
ulimit -r 99
ulimit -l unlimited
setpriv --reuid=spring --regid=spring --init-groups taskset -c "$FRANKA_IO_CPUS" \
  python -u run_policy_stream.py "$@"
