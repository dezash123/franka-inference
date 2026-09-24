#!/usr/bin/env bash
# Build only: never connects to a robot or starts services.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
output_dir=${1:-"$repo_dir/.build/native"}
franka_prefix=${FRANKA_PREFIX:-/var/lib/spring-data/workloads/robo_run/vendor/franka_install}
[ -f "$franka_prefix/include/franka/robot.h" ] || { echo "Set FRANKA_PREFIX to the built libfranka 0.21.3 install" >&2; exit 1; }
mkdir -p "$output_dir/camera"
for target in stream_fci read_fci_errors read_fci_snapshot read_fci_realtime recover_fci; do
  "${CXX:-g++}" -std=c++17 -O2 -pthread \
    -I"$repo_dir/franka_droid" -I"$franka_prefix/include" -I/usr/include/eigen3 \
    "$repo_dir/franka_droid/$target.cpp" \
    -L"$franka_prefix/lib" -Wl,-rpath,"$franka_prefix/lib" -lfranka -ltinyxml2 \
    -o "$output_dir/$target"
done
"${CC:-cc}" -O2 -std=gnu11 -Wall -Wextra -Werror -fPIC -shared \
  "$repo_dir/camera/capture_v4l2.c" -o "$output_dir/camera/libcapture_v4l2.so"
printf 'Built native programs and camera helper in %s. Nothing executed.\n' "$output_dir"
