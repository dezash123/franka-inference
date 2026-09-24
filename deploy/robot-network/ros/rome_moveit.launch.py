from pathlib import Path
import yaml
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    bringup = Path(get_package_share_directory('franka_bringup'))
    description = Path(get_package_share_directory('franka_description'))
    config = Path(get_package_share_directory('franka_fr3_moveit_config'))
    model = 'fr3v2_1'
    urdf = xacro.process_file(str(bringup / 'urdf/franka_arm.urdf.xacro'), mappings={
        'robot_type': model, 'robot_ip': '172.16.0.2', 'hand': 'false',
        'ee_id': 'none', 'use_fake_hardware': 'false',
    }).toxml()
    srdf = xacro.process_file(str(description / 'robots' / model / (model + '.srdf.xacro')),
                              mappings={'hand': 'false', 'ee_id': 'none'}).toxml()
    ompl = yaml.safe_load((config / 'config/ompl_planning.yaml').read_text().replace('fr3_', 'fr3v2_1_'))
    ompl.update({
        'planning_plugins': ['ompl_interface/OMPLPlanner'],
        'request_adapters': [
            'default_planning_request_adapters/ResolveConstraintFrames',
            'default_planning_request_adapters/ValidateWorkspaceBounds',
            'default_planning_request_adapters/CheckStartStateBounds',
            'default_planning_request_adapters/CheckStartStateCollision',
        ],
        'response_adapters': ['default_planning_response_adapters/AddTimeOptimalParameterization',
                              'default_planning_response_adapters/ValidateSolution'],
    })
    joints = [f'{model}_joint{i}' for i in range(1, 8)]
    parameters = {
        'robot_description': urdf,
        'robot_description_semantic': srdf,
        'robot_description_kinematics': {f'{model}_arm': {
            'kinematics_solver': 'kdl_kinematics_plugin/KDLKinematicsPlugin',
            'kinematics_solver_search_resolution': 0.005,
            'kinematics_solver_timeout': 0.1,
        }},
        'planning_pipelines': ['ompl'],
        'default_planning_pipeline': 'ompl',
        'ompl': ompl,
        'moveit_controller_manager': 'moveit_simple_controller_manager/MoveItSimpleControllerManager',
        'moveit_simple_controller_manager': {
            'controller_names': ['arm_controller'],
            'arm_controller': {'type': 'FollowJointTrajectory', 'action_ns': 'follow_joint_trajectory',
                               'default': True, 'joints': joints},
        },
        'moveit_manage_controllers': False,
        'trajectory_execution.allowed_start_tolerance': 0.003,
        'publish_planning_scene': True,
        'publish_geometry_updates': True,
        'publish_state_updates': True,
        'publish_transforms_updates': True,
    }
    return LaunchDescription([Node(package='moveit_ros_move_group', executable='move_group',
                                   namespace='franka', parameters=[parameters], output='screen')])
