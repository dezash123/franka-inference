from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    bringup = Path(get_package_share_directory('franka_bringup'))
    return LaunchDescription([
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(bringup / 'launch/franka.launch.py')),
                                 launch_arguments={
                                     'robot_type': 'fr3v2_1',
                                     'robot_ip': '172.16.0.2',
                                     'namespace': 'franka',
                                     'load_gripper': 'false',
                                     'use_fake_hardware': 'false',
                                     'joint_state_rate': '100',
                                     'controllers_yaml': '/var/lib/spring-data/franka-network-20260923/ros/controllers.yaml',
                                 }.items()),
        Node(package='controller_manager', executable='spawner',
             arguments=['arm_controller', '--inactive', '-c', '/franka/controller_manager',
                        '--controller-manager-timeout', '60'], output='screen'),
    ])
