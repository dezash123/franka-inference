"""OpenVINO W8A8 checkpoint-A (pi05_droid_jointpos) mux adapter, with its checkpoint-specific CPU transforms.

The served stack is the OvCkptA line: the custom GPU plugin, the T=64 + T=256 checkpoint-A
IRs resident in one ov_mux_server, five Euler steps inside the graph, and the CPU transform
that produced its RoboLab row (openpi's DROID pi05 inputs; quantile Unnormalize on actions
AND state, then AbsoluteActions on the seven joints).  Same case-directory / JSON-lines
protocol as the native SSOG adapter; the graph is stateless, so `reset` is a protocol no-op.
"""
import atexit
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time

import numpy as np

from contract import ROOT, IMAGE_KEYS, validate_observation

INSTALL = Path('/var/lib/spring-data/workloads/pi05-final/ov-a')
sys.path.insert(0, str(INSTALL / 'unit/src'))
sys.path.insert(0, str(INSTALL / 'unit'))
from unit_ov import CHECKPOINTS, Observation, OvServe, Transform, route_window, write_case

ACTION_SPACE = 'absolute_joint_position_radians_7_and_absolute_gripper_0_open_1_closed'
PRECISION = 'W8A8'
CHECKPOINT = 'pi05_droid_jointpos'
PROFILE = CHECKPOINTS['A']
ASSETS = INSTALL / 'unit/assets_ckptA'
LAUNCH = INSTALL / 'serve_ov_a.sh'
# The OvCkptA receipt's identities (campaign13/RuntimeCore/evidence/ov-ckptA-20260924 §3.1, §4).
IR_SHA256 = {
    64: {'xml': '12ce852195a82564c3ef8f4804a1f069a5f2df9e245efe4547e9675b3a8748f8',
         'bin': 'd84fcd2cf4f0f307a2218a0f94b8e546239ae6683894a13ec68af0cd6d0ddb8b'},
    256: {'xml': '3895ab861e9c70d544e0d09652fe627e2ed71c28d69ed5a0ee235385722c3f95',
          'bin': '785a45d0da322245942ab4f4ad9fe7bb04bf91e42207e66ff65d7dc124bd36da'},
}
PLUGIN_SHA256 = '0dac1fd9d724ad025899dbdc6abbcdad50b1144ff7c1656a90b8bea36e493b96'
SERVER_SHA256 = '78f971a40313a9e34a3cea84d86e79e42437c20e7f111529a93fe2c77a0e3ea4'


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


class OwnedOvServe(OvServe):
    """Reuse the Unit's IPC; make construction failures leave no server behind."""
    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
        except BaseException:
            if hasattr(self, '_process'):
                self.kill()
            if hasattr(self, '_log'):
                self._log.close()
            raise


class OvAPolicy:
    steps = 5
    precision = PRECISION
    checkpoint = CHECKPOINT
    action_space = ACTION_SPACE
    device_name = 'Intel Arc B580'

    def __init__(self):
        self.lock = threading.Lock()
        self._closed = False
        self._engine = None
        self._case_dir = None
        self._episode_key = None
        self._chunk = 0
        self._salt = os.urandom(16).hex()
        digests = {'normalization_sha256': _sha256(ASSETS / 'norm_stats.json'),
                   'tokenizer_sha256': _sha256(ASSETS / 'paligemma_tokenizer.model'),
                   'plugin_so_sha256': _sha256(INSTALL / 'custom-plugin-cams-r11/libopenvino_intel_gpu_plugin.so'),
                   'server_sha256': _sha256(INSTALL / 'ovmux/ov_mux_server.py'),
                   'serve_launch_sha256': _sha256(LAUNCH)}
        if digests['normalization_sha256'] != PROFILE['norm_stats_sha256']:
            raise RuntimeError('norm_stats.json is not checkpoint A\'s (OvCkptA profile sha mismatch)')
        if digests['plugin_so_sha256'] != PLUGIN_SHA256:
            raise RuntimeError('custom GPU plugin checksum mismatch')
        if digests['server_sha256'] != SERVER_SHA256:
            raise RuntimeError('ov_mux_server.py checksum mismatch')
        self.runtime = {'precision': PRECISION, 'checkpoint': CHECKPOINT, 'checkpoint_kind': PROFILE['checkpoint_kind'],
                        'line': 'ov-w8a8-c2-mux/ckptA', 'sampler': 'Euler, 5 steps inside the graph', 'model_passes': 1,
                        'action_space': ACTION_SPACE, 'outputs': PROFILE['outputs'],
                        'noise_policy': 'PCG64 N(0,1), SHA256(salt|episode|chunk) seed', **digests}
        try:
            self._case_dir = Path(tempfile.mkdtemp(prefix='franka-ova-', dir='/dev/shm'))
            self._engine = OwnedOvServe(LAUNCH, 0, ROOT / 'evidence/ov-a-policy-engine.log')
            ready = self._engine.ready
            windows = tuple(sorted(int(w) for w in ready.get('windows', [])))
            if not windows or windows[0] != 64 or not set(windows) <= set(IR_SHA256):
                raise RuntimeError(f'Unexpected resident windows: {ready.get("windows")}')
            if ready.get('batch_vision') is not False:
                raise RuntimeError('the server must run with batch_vision disabled')
            if ready.get('plugin_so_sha256') != PLUGIN_SHA256:
                raise RuntimeError(f'server loaded plugin {ready.get("plugin_so_sha256")}, expected {PLUGIN_SHA256}')
            for window in windows:
                stack = ready[f't{window}']
                for kind in ('xml', 'bin'):
                    if stack[f'ir_{kind}_sha256'] != IR_SHA256[window][kind]:
                        raise RuntimeError(f'T={window} IR {kind} is {stack[f"ir_{kind}_sha256"]}, '
                                           f'expected checkpoint A {IR_SHA256[window][kind]}')
            self._transform = Transform(ASSETS, windows, PROFILE['absolute_actions'])
            self.runtime['text_windows'] = list(windows)
            self.runtime['engine_ready'] = {k: v for k, v in ready.items() if k != 'ready'}
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._engine is not None:
            self._engine.quit()
        if self._case_dir is not None:
            shutil.rmtree(self._case_dir, ignore_errors=True)

    def infer(self, obs):
        with self.lock:
            return self._infer(obs)

    def _infer(self, obs):
        begin = time.perf_counter()
        validate_observation(obs)
        session = str(obs.get('_session_id', 'standalone'))
        view = str(obs.get('_external_view', 'unspecified'))
        key = (session, view, obs['prompt'])
        if key != self._episode_key:
            self._engine.reset('arm')
            self._episode_key = key
        seed = int.from_bytes(hashlib.sha256(f'{self._salt}|{key}|{self._chunk}'.encode()).digest()[:8], 'little')
        noise = np.random.default_rng(seed).standard_normal((15, 32)).astype(np.float32)
        observation = Observation(episode_id='arm', env_id=0, step=0, chunk_index=self._chunk,
                                  instruction=obs['prompt'], seed=seed,
                                  images={k.removeprefix('observation/'): obs[k] for k in IMAGE_KEYS},
                                  joint_position=np.asarray(obs['observation/joint_position']),
                                  gripper_position=np.asarray(obs['observation/gripper_position']))
        case = self._transform.case(observation, noise)
        write_case(self._case_dir, case)
        prepared = time.perf_counter()
        before_ms = self._engine.device_ms
        normalized = self._engine.chunk('arm', self._case_dir)
        device_done = time.perf_counter()
        actions = self._transform.unnormalize(normalized, case['state'])
        if actions.shape != (15, 8) or not np.isfinite(actions).all():
            raise RuntimeError('OpenVINO checkpoint-A line returned invalid actions')
        self._chunk += 1
        end = time.perf_counter()
        window = route_window(case['text_valid'], self._transform.windows)
        self.runtime['latest_text_valid'] = case['text_valid']
        self.runtime['latest_window'] = window
        return {'actions': actions,
                'policy_timing': {'preprocess_ms': (prepared - begin) * 1000,
                                  'infer_ms': (device_done - prepared) * 1000,
                                  'archive_reported_ms': self._engine.device_ms - before_ms,
                                  'total_ms': (end - begin) * 1000,
                                  'text_window': window, 'text_valid': case['text_valid']}}
