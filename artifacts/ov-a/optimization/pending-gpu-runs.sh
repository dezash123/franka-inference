#!/usr/bin/env bash
# Queued GPU runs for when billy access returns. Run from /workflow on billy
# after syncing optimization/ and the plugin directories. Posts nothing; the
# operator posts results to the dashboard from the result.json files.
set -euo pipefail
cd /workflow
./run_r11.sh r11-schedule-critical schedule-critical-r11-tuning.json custom-plugin-schedule-r11 100 > r11-schedule-critical.out 2>&1 || true
for c in mb8-g8 mb16-g8 mb16-g16 mb32-g16; do
  ./run_r11.sh sweep-gateup-pf-$c sweep-gateup-pf-$c.json custom-plugin-gateup-prefetch-r11 100 > sweep-gateup-pf-$c.out 2>&1 || true
done
./run_r11.sh sweep-gateup-pf-mb16-sched sweep-gateup-pf-mb16-sched.json custom-plugin-gateup-prefetch-r11 100 > sweep-gateup-pf-mb16-sched.out 2>&1 || true
for f in r11-schedule-critical sweep-gateup-pf-mb8-g8 sweep-gateup-pf-mb16-g8 sweep-gateup-pf-mb16-g16 sweep-gateup-pf-mb32-g16 sweep-gateup-pf-mb16-sched; do
  echo "$f $(grep -oE 'median_ms.: [0-9.]+' $f.out | head -1)"
done
