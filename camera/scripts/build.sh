#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$ROOT/build"
cc -O2 -std=gnu11 -Wall -Wextra -Werror -fPIC -shared \
  "$ROOT/capture_v4l2.c" -o "$ROOT/build/libcapture_v4l2.so.tmp"
mv "$ROOT/build/libcapture_v4l2.so.tmp" "$ROOT/build/libcapture_v4l2.so"
