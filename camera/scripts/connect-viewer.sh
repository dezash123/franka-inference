#!/usr/bin/env bash
# Run this on the computer containing your browser, not on Rome.
set -euo pipefail
TARGET=${1:-spring@100.95.186.107}
LOCAL_PORT=${2:-18765}
printf 'Keep this terminal open, then visit http://127.0.0.1:%s\n' "$LOCAL_PORT"
exec ssh -NT -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L "127.0.0.1:$LOCAL_PORT:127.0.0.1:8765" "$TARGET"
