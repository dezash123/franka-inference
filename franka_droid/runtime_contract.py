"""Explicit inference and action contracts for each selectable runtime."""
NORMALIZED = 'normalized_joint_increment_7_and_absolute_gripper_0_open_1_closed'
ABSOLUTE = 'absolute_joint_position_radians_7_and_absolute_gripper_0_open_1_closed'
RUNTIMES = {
    'openvino': {'label': 'OpenVINO FP16', 'precision': 'FP16', 'denoise_steps': 5, 'action_space': NORMALIZED},
    'torch_eager': {'label': 'Torch eager FP16', 'precision': 'FP16', 'denoise_steps': 5, 'action_space': NORMALIZED},
    'torch_compile': {'label': 'torch.compile FP16', 'precision': 'FP16', 'denoise_steps': 5, 'action_space': NORMALIZED},
    'ssog_mc2': {'label': 'SSOG A MC2 · INT8/INT4 · 2 passes', 'precision': 'W8A8/W4A8', 'denoise_steps': 2, 'action_space': ABSOLUTE},
    'ov_a_w8a8': {'label': 'OpenVINO W8A8 ckpt-A · 64+256 · 5 steps', 'precision': 'W8A8', 'denoise_steps': 5, 'action_space': ABSOLUTE},
}


def validate_runtime(metadata, backend):
    expected = RUNTIMES[backend]
    if metadata.get('backend') != backend:
        raise RuntimeError('Selected runtime does not match the policy server')
    for key in ('precision', 'denoise_steps', 'action_space'):
        if metadata.get(key) != expected[key]:
            raise RuntimeError(f'Policy {key} does not match {expected["label"]}')
    if expected['action_space'] == ABSOLUTE and metadata.get('checkpoint') != 'pi05_droid_jointpos':
        raise RuntimeError('Absolute joint-position runtime must serve pi05_droid_jointpos')
    return expected
