#!/bin/bash
set -e
source /var/lib/spring-data/franka-network-20260923/ros/env.sh
exec ros2 launch /var/lib/spring-data/franka-network-20260923/ros/rome_hardware.launch.py
