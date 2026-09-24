#!/bin/bash
# Source this file. No services, cameras, model, or controller are started.
source /var/lib/spring-data/franka-network-20260923/ros/env.sh
export FRANKA_DROID_ROOT=/var/lib/spring-data/workloads/robo_run/franka_droid
export PATH="$FRANKA_DROID_ROOT/venv/bin:$PATH"
export PYTHONPATH="/home/spring/robo_run/openpi/src:/home/spring/robo_run/openpi/packages/openpi-client/src:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu
export TOKENIZERS_PARALLELISM=false
