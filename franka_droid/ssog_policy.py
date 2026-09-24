"""Archived SSOG A MC2 MUX adapter, with its checkpoint-specific CPU transforms."""
import atexit
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

from contract import ROOT, IMAGE_KEYS, validate_observation

INSTALL = Path('/var/lib/spring-data/workloads/pi05-final/ssog-a-mc2')
sys.path.insert(0, str(INSTALL / 'unit/src'))
sys.path.insert(0, str(INSTALL / 'unit'))
from unit_b580 import Assets, EmbedTable, FusedServe, Observation, Transform, write_case

ACTION_SPACE = 'absolute_joint_position_radians_7_and_absolute_gripper_0_open_1_closed'
PRECISION = 'W8A8/W4A8'
CHECKPOINT = 'pi05_droid_jointpos'
BINARY_SHA256 = 'db89756ca6b0451040cbbf6e3490d28606d5ed510de3d4cc371bfce3259d6edb'


class MuxTransform(Transform):
    """Preserve the archived recipe, padding to the selected mux text window."""
    def tokenize(self, prompt, state):
        cleaned = prompt.strip().replace('_', ' ').replace('\n', ' ')
        discretized = np.digitize(state, bins=self._bins) - 1
        full = f"Task: {cleaned}, State: {' '.join(map(str, discretized))};\nAction: "
        tokens = self._tokenizer.encode(full, add_bos=True)
        window = next((n for n in (64, 128, 256) if len(tokens) <= n), None)
        if window is None:
            raise ValueError(f'Prompt and state need {len(tokens)} tokens; SSOG supports 256')
        padding = window - len(tokens)
        return (np.asarray(tokens + [0] * padding, dtype=np.int64),
                np.asarray([True] * len(tokens) + [False] * padding, dtype=bool))


class OwnedMuxServe(FusedServe):
    """Reuse archive IPC; cleanup only this adapter's exact Docker container."""
    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
        except BaseException:
            if hasattr(self, '_process'):
                self.kill()
            if hasattr(self, '_log'):
                self._log.close()
            raise

    def _reap_stale_containers(self):
        pass

    def _containers(self):
        process = getattr(self, '_process', None)
        return [f'pi05-mux-{process.pid}'] if process else []

    def kill(self):
        if self._process.poll() is None:
            self._process.kill()
            self._process.wait(timeout=3)
        for name in self._containers():
            subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=5)

    def quit(self):
        try:
            if self._process.poll() is None:
                self._request({'cmd': 'quit'}, timeout_s=3)
                self._process.wait(timeout=3)
        finally:
            self.kill()
            self._log.close()


class SSOGPolicy:
    steps = 2
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
        binary = INSTALL / 'muxfinal/bin/fused_mux_droid'
        if hashlib.sha256(binary.read_bytes()).hexdigest() != BINARY_SHA256:
            raise RuntimeError('SSOG mux binary checksum mismatch')
        assets = Assets(INSTALL / 'unit/assets', INSTALL / 'campaign13/share/weights-A', INSTALL / 'run_mux_serve.sh')
        digests = assets.digests(verify_weights=True)
        self._transform = MuxTransform(assets.root, EmbedTable(assets.weights / 'gemma_embed.safetensors'))
        self.runtime = {'precision': PRECISION, 'checkpoint': CHECKPOINT, 'binary_sha256': BINARY_SHA256,
                        'sampler': 'MeanCache MC2', 'model_passes': 2, 'integration_grid_nodes': 10,
                        'text_windows': [64, 128, 256], 'action_space': ACTION_SPACE,
                        'visual_pruning_keep_ratio': 0.5, 'visual_pruning_history': 'reset on session, prompt, or external view change',
                        'noise_policy': 'PCG64 N(0,1), SHA256(salt|episode|chunk) seed', **digests}
        try:
            self._case_dir = Path(tempfile.mkdtemp(prefix='franka-ssog-', dir='/dev/shm'))
            self._engine = OwnedMuxServe(assets.launch, 0, ROOT / 'evidence/ssog-policy-engine.log')
            ready = self._engine.ready
            if ready.get('line') != 'ssog-A-mc2' or not ready.get('mux') or ready.get('steps') != 2:
                raise RuntimeError(f'Unexpected archived runtime: {ready}')
            self.runtime['engine_ready'] = ready
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
        actions = self._transform.served(normalized, case['state'])
        if actions.shape != (15, 8) or not np.isfinite(actions).all():
            raise RuntimeError('SSOG returned invalid actions')
        self._chunk += 1
        end = time.perf_counter()
        window = len(case['tokens'])
        self.runtime['latest_text_valid'] = case['text_valid']
        self.runtime['latest_window'] = window
        return {'actions': actions,
                'policy_timing': {'preprocess_ms': (prepared - begin) * 1000,
                                  'infer_ms': (device_done - prepared) * 1000,
                                  'archive_reported_ms': self._engine.device_ms - before_ms,
                                  'total_ms': (end - begin) * 1000,
                                  'text_window': window, 'text_valid': case['text_valid']}}
