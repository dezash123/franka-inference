#!/bin/bash
set -e
cd /var/lib/spring-data/workloads/robo_run/franka_droid
source ./env.sh
backend=$(python -c 'import json; print(json.load(open("config/setup.json"))["policy"]["backend"])')
if [ "$backend" = openvino ]; then
  exec ov-venv/bin/python -u serve_policy.py --start
else
  exec venv/bin/python -u serve_policy.py --start
fi
