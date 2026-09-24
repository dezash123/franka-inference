"""DROID data contract. Pure transformations; no robot command transport."""
import json
from pathlib import Path
import numpy as np
from openpi_client.image_tools import resize_with_pad

ROOT = Path(__file__).resolve().parent
IMAGE_KEYS = ('observation/exterior_image_1_left', 'observation/wrist_image_left')

def settings():
    return json.loads((ROOT / 'config/setup.json').read_text())

def validate_observation(obs):
    for key in IMAGE_KEYS:
        im = np.asarray(obs[key])
        if im.shape != (224, 224, 3) or im.dtype != np.uint8:
            raise ValueError(f'{key}: require uint8 RGB 224x224x3, got {im.shape}/{im.dtype}')
    for key, shape in [('observation/joint_position', (7,)), ('observation/gripper_position', (1,))]:
        value = np.asarray(obs[key])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f'Invalid {key}')
    if not 0 <= float(obs['observation/gripper_position'][0]) <= 1:
        raise ValueError('Gripper closure must be in [0,1]')
    if not isinstance(obs['prompt'], str) or not obs['prompt'].strip():
        raise ValueError('A task prompt is required')

def make_observation(external_rgb, wrist_rgb, joints, gripper_closure, prompt):
    result = {
        IMAGE_KEYS[0]: resize_with_pad(external_rgb, 224, 224),
        IMAGE_KEYS[1]: resize_with_pad(wrist_rgb, 224, 224),
        'observation/joint_position': np.asarray(joints, dtype=np.float32),
        'observation/gripper_position': np.asarray([gripper_closure], dtype=np.float32),
        'prompt': prompt,
    }
    validate_observation(result)
    return result

def gripper_closure_from_width(width, maximum):
    if maximum is None or maximum <= 0 or not np.isfinite(width) or not 0 <= width <= maximum:
        raise ValueError('Need a valid measured gripper width and calibrated maximum width')
    return 1.0 - width / maximum

def decode_droid_action(action, current_q, max_width_m):
    """Inspection only. DROID uses normalized joint increments, not SI rad/s.

    Upstream RobotIKSolver.joint_velocity_to_delta multiplies by 0.2 rad.
    RobotEnv applies a new desired joint position at 15 Hz. This is NOT sent
    to ROS here and is not equivalent to streaming these values as rad/s.
    """
    a = np.asarray(action, dtype=np.float64)
    q = np.asarray(current_q, dtype=np.float64)
    if a.shape != (8,) or q.shape != (7,) or not np.isfinite(a).all() or not np.isfinite(q).all():
        raise ValueError('Invalid DROID action or joint state')
    if np.max(np.abs(a)) > 1:
        raise ValueError('DROID action outside [-1,1]; do not silently clip')
    if max_width_m is None or max_width_m <= 0:
        raise ValueError('Unknown gripper calibration')
    closure = float(a[7] > 0.5)
    return {'desired_joints_rad': (q + 0.2*a[:7]).tolist(),
            'gripper_width_m': max_width_m*(1-closure), 'commands_sent': False}
