#!/usr/bin/env python3
"""The SpriliCim policy worker ("the Unit") serving the OpenVINO W8A8 line from a B580.

The policy forward is one resident ``ov_mux_server.py`` (handoff/israel-c2t64/) holding
the three corrected W8A8 IRs -- T = 64 / 128 / 256 text tokens -- and routing every chunk
on its ``text_valid``.  The device stack is the one
``evidence/w8a8-c2t64-20260922/RESULTS.md`` §11.2 ships: the corrected IR (the int32
CumSum export fix), the custom plugin ``custom-plugin-cams-r11`` with every custom kernel
family and ``fuse_qkv`` / ``align_vision_mlp`` / ``fuse_vision_padding`` enabled, and
``batch_vision`` DISABLED -- the plugin's batch-2 camera tower is defective (§9) and
disabling it is what buys 0.0278 instead of 0.2374 rel-L2 against eager PyTorch.

This adapter does on the CPU what the served torch Units do around the model.  It is
``unit_b580.py``'s structure (case directory in, ``actions.f32`` out, JSON lines, the
same noise policy, the same identity discipline) with ``unit_thor_trt.py``'s transform,
because this line and Desmond's TensorRT line are **the same checkpoint**:

* openpi's DROID pi05 input transform -- ``DroidInputs`` (state = joint[7] ++ gripper[1]),
  quantile ``Normalize`` on the 8 real state dims, the pi05 prompt with the discretized
  state, the strict PaliGemma tokenizer, uint8 frames to [-1, 1] CHW, pad camera dropped.
  The IR consumes token **ids**, not embedding rows: the Gemma lookup is inside the graph.
  The prompt is zero-padded to the window it ROUTES to (64 / 128 / 256, the smallest that
  holds it); above 256 it raises.  Nothing is ever truncated.
* openpi's output transform -- quantile ``Unnormalize`` (32-wide statistics), first 8 dims.

TWO THINGS ARE NOT LIKE ``unit_b580.py``, and both are load-bearing:

1. **Checkpoint B is joint-VELOCITY.**  The IRs were traced from HF ``lerobot/pi05_droid``
   revision 72824c0a93f00ce5bb8bedb7feb58953ba1da364 (``run_framework_baselines.py:67``),
   i.e. openpi's ``pi05_droid`` config: ``DroidInputs``/``DroidOutputs`` with **no**
   ``AbsoluteActions``, so its 8 served dims are ``joint_velocity[7] ++ gripper[1]``.
   RoboLab's DROID robot takes absolute joint-position targets, so the chunk MUST be
   integrated at the environment's control rate (``dt = 1/(60*2)``, ``decimation = 8`` =>
   ``env_dt = 8/120 = 1/15 s``):

       q_target[k] = q_observed + dt * sum_{j<=k} v[j],    dt = 1/15 s

   open loop across the 15 rungs, seeded from the observation's measured joint position;
   the gripper dim is absolute in both spaces and passes through.  This is DesmondTRT's
   adapter (``unit_thor_trt.py::decode_actions``), reused unchanged, named in the identity
   hash and switchable to ``raw`` so the decision stays visible.  The checkpoint's own
   normalization statistics (``assets_ckptb/norm_stats.json``, checkpoint B) are used --
   NOT ``assets_b580``'s, which belong to the joint-position checkpoint A.

2. **The graph is stateless.**  No pruner ring, no sampler warm start, no KV between
   chunks: every chunk is a full forward over both camera towers, the prefix and all five
   flow steps.  Per-episode reset is a no-op and ``chunk_index`` changes nothing.  That is
   this line's temporal contract, not an omission.

``--checkpoint`` selects which of the two DROID checkpoints the resident IRs hold.  ``B``
(the default) is everything above, unchanged; ``A`` is openpi's ``pi05_droid_jointpos`` --
the joint-POSITION checkpoint -- whose output transform carries ``AbsoluteActions``, so
its chunk is decoded with ``native_joint_position`` (the state is added back inside the
transform, nothing is integrated) against ``assets_ckptA/norm_stats.json``.  The two
profiles differ only in the checkpoint identity, the output recipe and the decode; the
Unit refuses a profile whose ``norm_stats.json`` digest is the other checkpoint's.

Noise: N(0,1) float32 [15,32] from a PCG64 stream seeded by
sha256(salt | episode_id | chunk_index), as ``unit_b580.py`` does, so a redelivered
observation is served bit-identically.

The identity binds the three IR digests (xml and bin), the plugin .so, the three tuning
files, the server, the checkpoint, the normalization statistics, the tokenizer, the
transform recipe, the action decode and the noise policy.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import pathlib
import selectors
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sprilicim.policy.base import Observation, Policy
from sprilicim.policy.transport import HttpHqPolicyClient
from sprilicim.policy.worker import PolicyWorker
from sprilicim.protocol.schemas import IMAGE_KEYS, PolicySpec

log = logging.getLogger("ssog.unit_ov")

HORIZON = 15
ACTION_DIM = 32
SERVED_DIM = 8
# The window ladder this line can serve.  Which of them are RESIDENT is the
# server's business and is read off its ready line: each window costs 4.28 GB of
# VRAM and a 12 GB B580 holds two, so the shipped set is (64, 256) and a
# 65..128-token prompt runs on the 256 geometry.  A prompt is always padded to
# the smallest RESIDENT window that holds it, and never truncated.
WINDOWS = (64, 128, 256)
MAX_TOKEN_LEN = WINDOWS[-1]
LINE = "ov-w8a8-c2-mux"
CHECKPOINT = "lerobot/pi05_droid"
REVISION = "72824c0a93f00ce5bb8bedb7feb58953ba1da364"
# RoboLab droid jointpos registration: sim dt = 1/(60*2), decimation = 8.
ENV_DT_S = 8.0 / 120.0


def route_window(n_tokens: int, windows: tuple[int, ...] = WINDOWS) -> int:
    for window in sorted(windows):
        if n_tokens <= window:
            return window
    raise ValueError(f"Token length ({n_tokens}) exceeds max length ({max(windows)}).")


METHOD: dict[str, Any] = {
    "config": "pi05_droid",
    "checkpoint_kind": "checkpoint_B_joint_velocity",
    "line": LINE,
    "execution": "openvino_w8a8_custom_plugin_mux_serve",
    "precision": "INT8_SYM weights (NNCF 3.3.0, data-free RTN, per-output-channel) with "
                 "per-token INT8 activations; f16 elsewhere",
    "graph": "corrected IR (int32 CumSum export fix); batch_vision DISABLED (plugin defect)",
    "rewrites": ["fuse_qkv(vision,language,expert)", "align_vision_mlp", "fuse_vision_padding"],
    "num_steps": 5,
    "sampler": "euler",
    "cameras": 2,
    "drop_pad_camera": True,
    "max_token_len": MAX_TOKEN_LEN,
    "text_windows": list(WINDOWS),
    "stateful": False,
    "reset": "no-op: the graph carries no state between chunks",
}

TRANSFORM: dict[str, Any] = {
    "inputs": [
        "DroidInputs: state = joint_position[7] ++ gripper_position[1]; cameras base_0_rgb, left_wrist_0_rgb",
        "Normalize(use_quantiles): (x - q01) / (q99 - q01 + 1e-6) * 2 - 1 on the first 8 state dims",
        "images 224x224 uint8 RGB -> float32 / 255 * 2 - 1, HWC -> CHW, pad camera dropped",
        "PaligemmaTokenizer(strict, max 256): 'Task: <prompt>, State: <digitize(state, linspace(-1,1,257)[:-1]) - 1>;\\nAction: ', BOS, zero-padded to the ROUTED window (64 / 128 / 256 = the smallest that holds the prompt); over 256 raises, never truncates",
        "token ids and the boolean prompt mask go to the IR; the embedding lookup is inside the graph",
    ],
    "outputs": [
        "Unnormalize(use_quantiles, 32-wide, checkpoint B statistics): (x + 1) / 2 * (q99 - q01 + 1e-6) + q01",
        "DroidOutputs: actions[:, :8] = joint_velocity[7] ++ gripper_position[1] (NO AbsoluteActions; pi05_droid is the JOINT_VELOCITY config)",
    ],
    "noise": "N(0,1) float32 [15,32] from numpy PCG64 seeded by sha256(salt|episode_id|chunk_index)[:8]",
}

DECODES: dict[str, str] = {
    "integrate_15hz": (
        "q_target[k] = joint_position_observed + (1/15 s) * cumsum(joint_velocity)[k] for k in 0..14; "
        "gripper passed through. RoboLab's DROID action term is JointPositionActionCfg "
        "(use_default_offset=False) at env_dt = 1/15 s, so the policy's velocity chunk is "
        "integrated at the control rate the robot's velocity controller would use. "
        "Adapter taken unchanged from sprilicim/unit_thor_trt.py (DesmondTRT)."
    ),
    "raw": "the joint-velocity chunk is served unchanged (NOT consumable by RoboLab's joint-position action term)",
    "native_joint_position": (
        "no adapter: checkpoint A's own output transform has already produced absolute joint "
        "positions. openpi's pi05_droid_jointpos baseline carries AbsoluteActions(mask 7 x True, "
        "1 x False), so the model's DELTA chunk gets the quantile round-tripped observed state "
        "added back inside the transform and RoboLab's JointPositionActionCfg takes the result "
        "unchanged. There is nothing left to integrate. NOT valid for checkpoint B."
    ),
}

ABSOLUTE_SPACE = "absolute_joint_position_radians_7_and_absolute_gripper_0_open_1_closed"
VELOCITY_SPACE = "joint_velocity_radians_per_second_7_and_absolute_gripper_0_open_1_closed"

# The two checkpoints this OpenVINO stack can hold.  Everything above is checkpoint B's and
# stays exactly what it was: `--checkpoint B` (the default) reproduces the shipped
# `ov-w8a8-c2-mux` Unit byte for byte, `identity_sha256` included.
#
# Checkpoint A is openpi's `pi05_droid_jointpos` -- the same architecture, different weights,
# and a DIFFERENT OUTPUT CONTRACT: `_PRETRAINED_BASELINES["pi05_droid_jointpos"]` sets
# `absolute_actions=True`, so its served chunk is
#     Unnormalize(32-wide quantiles) on the actions AND on the normalized state
#     -> AbsoluteActions(mask 7 x True, 1 x False): actions[:, :7] += state[:7]
#     -> DroidOutputs: actions[:, :8]
# The model emits joint DELTAS; the state added back is the quantile ROUND TRIP
# unnormalize(normalize(joint ++ gripper)), not the raw observed joint position.  That is
# `unit_b580.py::Transform.served()`, and serving checkpoint A without it is silently wrong.
CHECKPOINTS: dict[str, dict[str, Any]] = {
    "B": {
        "id": "B",
        "line": LINE,
        "config": METHOD["config"],
        "checkpoint": CHECKPOINT,
        "checkpoint_revision": REVISION,
        "checkpoint_kind": METHOD["checkpoint_kind"],
        "norm_stats_sha256": "403b3a22f897e9ae5dd617966a3c8f7d1835ac79dfd5a8993179514be26a3b8b",
        "absolute_actions": False,
        "outputs": TRANSFORM["outputs"],
        "policy_action_space": VELOCITY_SPACE,
        "decodes": ("integrate_15hz", "raw"),
        "absolute_decodes": ("integrate_15hz",),
        "model_space_metric": "velocity_abs_max_rad_s",
        "torch_ref_suffix": "b5",
        "tokens_must_match_accA": False,
        "reference_note": "the gate reference's torch actions are checkpoint A (joint position); "
                          "they are NOT a corridor for this checkpoint-B line and are not used as one",
    },
    "A": {
        "id": "A",
        "line": "ov-w8a8-c2-ckptA",
        "config": "pi05_droid_jointpos",
        "checkpoint": "gs://openpi-assets-simeval/pi05_droid_jointpos",
        "checkpoint_revision": "d040eb20b320096dc3de24359e06828bf01c6150",
        "checkpoint_kind": "checkpoint_A_joint_position",
        "norm_stats_sha256": "57ce9956f9e07d65f8a8205aabec72d436a2c8927f53edb40c7a77b14a5a90c7",
        "absolute_actions": True,
        "outputs": [
            "Unnormalize(use_quantiles, 32-wide, checkpoint A statistics): (x + 1) / 2 * (q99 - q01 + 1e-6) + q01 "
            "on the actions AND on the normalized state",
            "AbsoluteActions(mask 7 x True, 1 x False): actions[:, :7] += state[:7]",
            "DroidOutputs: actions[:, :8] = absolute joint_position[7] ++ gripper_position[1] "
            "(pi05_droid_jointpos IS the JOINT_POSITION config; openpi sets absolute_actions=True for it)",
        ],
        "policy_action_space":
            "delta_joint_position_radians_7_relative_to_the_observed_state_and_absolute_gripper_0_open_1_closed",
        "decodes": ("native_joint_position",),
        "absolute_decodes": ("native_joint_position",),
        "model_space_metric": "delta_abs_max_rad",
        "torch_ref_suffix": "a5",
        "tokens_must_match_accA": True,
        "reference_note": "the gate reference's torch actions are checkpoint A but the ssog-A-mc2 METHOD "
                          "(10-step MeanCache, VLA-Pruner, W4A8); they are a sanity corridor for this line, "
                          "not its corridor, and are not used as one",
    },
}


def sha256_file(path: pathlib.Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk):
            digest.update(block)
    return digest.hexdigest()


class Transform:
    """openpi's DROID pi05 input/output transforms, on the CPU, without openpi."""

    def __init__(self, assets: pathlib.Path, windows: tuple[int, ...] = WINDOWS,
                 absolute_actions: bool = False) -> None:
        import sentencepiece

        stats = json.loads((assets / "norm_stats.json").read_text())["norm_stats"]
        self.state_q01 = np.asarray(stats["state"]["q01"], dtype=np.float64)
        self.state_q99 = np.asarray(stats["state"]["q99"], dtype=np.float64)
        self.action_q01 = np.asarray(stats["actions"]["q01"], dtype=np.float64)
        self.action_q99 = np.asarray(stats["actions"]["q99"], dtype=np.float64)
        for name, value in (("state", self.state_q01), ("actions", self.action_q01)):
            if value.shape != (ACTION_DIM,):
                raise RuntimeError(f"{name} statistics are {value.shape}, expected ({ACTION_DIM},)")
        self._tokenizer = sentencepiece.SentencePieceProcessor(
            model_proto=(assets / "paligemma_tokenizer.model").read_bytes()
        )
        self._bins = np.linspace(-1, 1, 256 + 1)[:-1]
        self.windows = tuple(sorted(windows))
        # Checkpoint A's output transform has an AbsoluteActions step; checkpoint B's does not.
        self.absolute_actions = absolute_actions

    def normalize_state(self, joint: NDArray[np.float64], gripper: NDArray[np.float64]) -> NDArray[np.float64]:
        state = np.concatenate([np.asarray(joint, np.float32), np.asarray(gripper, np.float32)])
        q01, q99 = self.state_q01[: state.shape[-1]], self.state_q99[: state.shape[-1]]
        return (state - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0

    def tokenize(self, prompt: str, state: NDArray[np.float64]) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
        discretized = np.digitize(state, bins=self._bins) - 1
        full = f"Task: {cleaned}, State: {' '.join(map(str, discretized))};\nAction: "
        tokens = self._tokenizer.encode(full, add_bos=True)
        # route_window raises above 256; nothing here truncates a prompt.
        padding = route_window(len(tokens), self.windows) - len(tokens)
        return (
            np.asarray(tokens + [0] * padding, dtype=np.int64),
            np.asarray([True] * len(tokens) + [False] * padding, dtype=bool),
        )

    @staticmethod
    def images(obs_images: dict[str, NDArray[np.uint8]]) -> NDArray[np.float32]:
        planes = []
        for key in IMAGE_KEYS:
            image = np.asarray(obs_images[key])
            if image.dtype != np.uint8 or image.shape != (224, 224, 3):
                raise ValueError(f"{key}: expected uint8 (224, 224, 3), got {image.dtype} {image.shape}")
            planes.append(np.transpose(image.astype(np.float32) / 255.0 * 2.0 - 1.0, (2, 0, 1)))
        return np.ascontiguousarray(np.stack(planes).astype(np.float32))

    def case(self, obs: Observation, noise: NDArray[np.float32]) -> dict[str, Any]:
        state8 = self.normalize_state(obs.joint_position, obs.gripper_position)
        tokens, mask = self.tokenize(obs.instruction, state8)
        state32 = np.zeros(ACTION_DIM, dtype=state8.dtype)
        state32[: state8.shape[0]] = state8
        return {
            "siglip.in": self.images(obs.images),
            "tokens": tokens,
            "tokens_mask": mask,
            "text_valid": int(mask.sum()),
            "state": state32,
            "noise": np.ascontiguousarray(noise, dtype=np.float32),
        }

    def unnormalize_actions(self, actions: NDArray[np.float32]) -> NDArray[np.float64]:
        """The quantile step alone: the model's own action space, before this checkpoint's
        AbsoluteActions.  For checkpoint B that IS the served space (joint velocity); for
        checkpoint A it is the joint DELTA the model predicts."""
        if actions.shape != (HORIZON, ACTION_DIM):
            raise RuntimeError(f"device chunk is {actions.shape}, expected ({HORIZON}, {ACTION_DIM})")
        wide = np.asarray(actions, np.float64)
        return (wide + 1.0) / 2.0 * (self.action_q99 - self.action_q01 + 1e-6) + self.action_q01

    def unnormalize(self, actions: NDArray[np.float32],
                    state32: NDArray[np.float64] | None = None) -> NDArray[np.float64]:
        """openpi's model-space -> action-space transform: quantiles, then this checkpoint's
        AbsoluteActions when it has one (checkpoint A does, checkpoint B does not)."""
        wide = self.unnormalize_actions(actions)
        if self.absolute_actions:
            if state32 is None:
                raise RuntimeError("this checkpoint's AbsoluteActions needs the case's normalized state")
            state = ((np.asarray(state32, np.float64) + 1.0) / 2.0
                     * (self.state_q99 - self.state_q01 + 1e-6) + self.state_q01)
            mask = np.asarray([True] * 7 + [False])
            wide[..., : mask.shape[-1]] += np.expand_dims(np.where(mask, state[: mask.shape[-1]], 0), axis=-2)
        return wide[..., :SERVED_DIM]


def decode_actions(chunk: NDArray[np.float64], joint_position: NDArray[np.float64], mode: str) -> NDArray[np.float64]:
    """[15,8] joint_velocity ++ gripper -> what RoboLab's joint-position action term takes.

    Verbatim from sprilicim/unit_thor_trt.py (DesmondTRT); the two Units serve the same
    checkpoint and must decode it identically.
    """
    if mode in ("raw", "native_joint_position"):
        return chunk
    if mode != "integrate_15hz":
        raise ValueError(f"unknown action decode {mode!r}")
    out = np.array(chunk, dtype=np.float64, copy=True)
    out[:, :7] = np.asarray(joint_position, np.float64)[:7] + ENV_DT_S * np.cumsum(chunk[:, :7], axis=0)
    return out


def write_case(directory: pathlib.Path, case: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "actions.f32").unlink(missing_ok=True)
    for name in ("siglip.in", "tokens", "tokens_mask", "state", "noise"):
        np.save(directory / f"{name}.npy", np.ascontiguousarray(case[name]))
    (directory / "text_valid.txt").write_text(f"{case['text_valid']}\n")


class OvServe:
    """One resident ``ov_mux_server.py`` on one card, spoken to over JSON lines."""

    def __init__(self, launch: pathlib.Path, gpu: int, stderr_log: pathlib.Path,
                 ready_timeout_s: float = 900.0) -> None:
        self._gpu = gpu
        self._log = stderr_log.open("ab")
        self._process = subprocess.Popen(
            ["bash", str(launch), str(gpu)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            cwd=str(launch.parent),
        )
        self._selector = selectors.DefaultSelector()
        self._selector.register(self._process.stdout, selectors.EVENT_READ)
        self._lock = threading.Lock()
        self.ready = self._read_line(ready_timeout_s)
        if not self.ready.get("ready"):
            raise RuntimeError(f"ov_mux_server did not report ready: {self.ready}")
        log.info("ov_mux_server ready: %s", json.dumps(self.ready, sort_keys=True))
        self.chunks = 0
        self.device_ms = 0.0
        self.router_ms = 0.0
        self.windows = tuple(sorted(self.ready.get("windows", WINDOWS)))
        self.window_counts: dict[int, int] = {}

    def _read_line(self, timeout_s: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while True:
            if self._process.poll() is not None:
                raise RuntimeError(f"ov_mux_server exited with {self._process.returncode}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.kill()
                raise RuntimeError(f"ov_mux_server answered nothing in {timeout_s:.0f}s; killed")
            if not self._selector.select(timeout=min(remaining, 1.0)):
                continue
            line = self._process.stdout.readline()
            if not line:
                raise RuntimeError("ov_mux_server closed its stdout")
            text = line.decode(errors="replace").strip()
            if not text:
                continue
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                log.warning("non-JSON line from the server's stdout: %r", text)

    def _request(self, message: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        with self._lock:
            self._process.stdin.write((json.dumps(message) + "\n").encode())
            self._process.stdin.flush()
            reply = self._read_line(timeout_s)
        if not reply.get("ok"):
            raise RuntimeError(f"ov_mux_server refused {message}: {reply}")
        return reply

    def reset(self, episode: str) -> None:
        """A no-op on this line; sent anyway so the protocol stays the served one."""
        self._request({"cmd": "reset", "episode": episode}, timeout_s=60.0)

    def chunk(self, episode: str, directory: pathlib.Path, timeout_s: float = 300.0) -> NDArray[np.float32]:
        reply = self._request({"cmd": "chunk", "episode": episode, "dir": str(directory)}, timeout_s)
        self.chunks += 1
        self.device_ms += float(reply.get("ms", 0.0))
        self.router_ms += float(reply.get("router_ms", 0.0)) - float(reply.get("ms", 0.0))
        window = int(reply["window"])
        self.window_counts[window] = self.window_counts.get(window, 0) + 1
        expected = route_window(int(reply["text_valid"]), self.windows)
        if window != expected:
            raise RuntimeError(f"server routed text_valid={reply['text_valid']} to T={window}, expected {expected}")
        actions = np.fromfile(directory / "actions.f32", dtype="<f4")
        if actions.shape != (HORIZON * ACTION_DIM,):
            raise RuntimeError(f"actions.f32 holds {actions.shape[0]} floats, expected {HORIZON * ACTION_DIM}")
        return actions.reshape(HORIZON, ACTION_DIM)

    def quit(self) -> None:
        if self._process.poll() is None:
            try:
                self._request({"cmd": "quit"}, timeout_s=60.0)
                self._process.wait(timeout=60.0)
            except Exception:
                log.warning("ov_mux_server did not quit cleanly", exc_info=True)
                self.kill()
        self._log.close()

    def kill(self) -> None:
        if self._process.poll() is None:
            self._process.kill()
            self._process.wait()

    def stats(self) -> dict[str, Any]:
        return {
            "device_chunks": self.chunks,
            "device_ms_per_chunk": self.device_ms / self.chunks if self.chunks else None,
            "router_ms_per_chunk": self.router_ms / self.chunks if self.chunks else None,
            "window_histogram": {str(k): v for k, v in sorted(self.window_counts.items())},
        }


@dataclasses.dataclass
class Assets:
    root: pathlib.Path
    launch: pathlib.Path
    checkpoint: str = "B"

    def digests(self) -> dict[str, Any]:
        profile = CHECKPOINTS[self.checkpoint]
        return {
            "checkpoint": profile["checkpoint"],
            "checkpoint_revision": profile["checkpoint_revision"],
            "checkpoint_kind": profile["checkpoint_kind"],
            "normalization_sha256": sha256_file(self.root / "norm_stats.json"),
            "tokenizer_sha256": sha256_file(self.root / "paligemma_tokenizer.model"),
            "serve_launch_sha256": sha256_file(self.launch),
        }


class OvPolicy(Policy):
    def __init__(
        self,
        name: str,
        assets: Assets,
        gpu: int,
        max_batch: int,
        noise_salt: str,
        work_dir: pathlib.Path,
        stderr_log: pathlib.Path,
        action_decode: str = "integrate_15hz",
    ) -> None:
        profile = CHECKPOINTS[assets.checkpoint]
        if action_decode not in profile["decodes"]:
            raise ValueError(f"checkpoint {assets.checkpoint} does not take action decode "
                             f"{action_decode!r}; its decodes are {list(profile['decodes'])}")
        self._profile = profile
        self._digests = assets.digests()
        # The one mis-wiring that would be silent: checkpoint A's IRs served with checkpoint
        # B's quantiles, or the reverse.  Both are 32-wide and both load.
        if self._digests["normalization_sha256"] != profile["norm_stats_sha256"]:
            raise RuntimeError(
                f"--assets {assets.root} holds norm_stats.json "
                f"{self._digests['normalization_sha256']}; checkpoint {assets.checkpoint} "
                f"needs {profile['norm_stats_sha256']}")
        self._serve = OvServe(assets.launch, gpu, stderr_log)
        served_windows = tuple(sorted(self._serve.ready.get("windows", [])))
        if not served_windows or not set(served_windows) <= set(WINDOWS) or served_windows[0] != WINDOWS[0]:
            raise RuntimeError(f"server holds windows {served_windows}; this line's ladder is {WINDOWS} "
                               "and the narrowest one must be resident")
        assert self._serve.windows == served_windows
        self._transform = Transform(assets.root, served_windows, profile["absolute_actions"])
        if self._serve.ready.get("batch_vision") is not False:
            raise RuntimeError("the server must run with batch_vision disabled (evidence §9)")
        self._case_dir = work_dir
        self._case_dir.mkdir(parents=True, exist_ok=True)
        self._noise_salt = noise_salt
        self._decode = action_decode
        self.chunks = 0
        self.resets = 0
        self.transform_s = 0.0
        self.device_s = 0.0
        method = {**METHOD, "config": profile["config"], "checkpoint_kind": profile["checkpoint_kind"],
                  "line": profile["line"]}
        recipe = {**TRANSFORM, "outputs": profile["outputs"]}
        self._metadata = {
            "config": profile["config"],
            "line": profile["line"],
            "backend": "openvino_ov_mux_serve",
            "server": self._serve.ready,
            "method": method,
            "transform": recipe,
            "action_decode": action_decode,
            "action_decode_note": DECODES[action_decode],
            "env_dt_s": ENV_DT_S,
            "noise_salt": noise_salt,
            "action_horizon": HORIZON,
            "action_dim": SERVED_DIM,
            "policy_action_space": profile["policy_action_space"],
            "served_action_space": (
                ABSOLUTE_SPACE
                if action_decode in profile["absolute_decodes"]
                else profile["policy_action_space"]
            ),
            "max_token_len": MAX_TOKEN_LEN,
            **self._digests,
        }
        self._spec = PolicySpec(
            name=name,
            identity_sha256=self.identity(self._metadata),
            image_size=(224, 224),
            horizon=HORIZON,
            action_dim=SERVED_DIM,
            max_batch=max_batch,
        )

    @staticmethod
    def identity(metadata: dict[str, Any]) -> str:
        """Bind the eval record to the IRs, the plugin, the tuning AND the serving recipe."""
        server = metadata["server"]
        body = {
            "line": metadata["line"],
            "backend": metadata["backend"],
            "windows": server.get("windows"),
            "irs": {key: {"xml": server[key]["ir_xml_sha256"], "bin": server[key]["ir_bin_sha256"],
                          "tuning": server[key]["tuning_sha256"], "prefix": server[key]["prefix_tokens"]}
                    for key in ("t64", "t128", "t256") if key in server},
            "plugin_so_sha256": server.get("plugin_so_sha256"),
            "server_sha256": server.get("server_sha256"),
            "rewrites": server.get("rewrites"),
            "batch_vision": server.get("batch_vision"),
            "openvino": server.get("openvino"),
            "checkpoint": metadata["checkpoint"],
            "checkpoint_revision": metadata["checkpoint_revision"],
            "normalization_sha256": metadata["normalization_sha256"],
            "tokenizer_sha256": metadata["tokenizer_sha256"],
            "method": metadata["method"],
            "transform": metadata["transform"],
            "action_decode": metadata["action_decode"],
            "env_dt_s": metadata["env_dt_s"],
            "noise_salt": metadata["noise_salt"],
        }
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()

    @property
    def spec(self) -> PolicySpec:
        return self._spec

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def noise(self, episode_id: str, chunk_index: int) -> NDArray[np.float32]:
        seed = hashlib.sha256(f"{self._noise_salt}|{episode_id}|{chunk_index}".encode()).digest()[:8]
        generator = np.random.default_rng(int.from_bytes(seed, "little"))
        return generator.standard_normal((HORIZON, ACTION_DIM), dtype=np.float32)

    def stats(self) -> dict[str, Any]:
        return {
            "chunks": self.chunks,
            "episodes_started": self.resets,
            "transform_ms_per_chunk": self.transform_s / self.chunks * 1000 if self.chunks else None,
            "device_roundtrip_ms_per_chunk": self.device_s / self.chunks * 1000 if self.chunks else None,
            **self._serve.stats(),
        }

    def normalized_chunk(self, obs: Observation, noise: NDArray[np.float32] | None = None) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """The device's [15,32] chunk in model space, plus the case that produced it."""
        if noise is None:
            noise = self.noise(obs.episode_id, obs.chunk_index)
        started = time.monotonic()
        case = self._transform.case(obs, noise)
        write_case(self._case_dir, case)
        self.transform_s += time.monotonic() - started
        if obs.chunk_index == 0:
            self._serve.reset(obs.episode_id)
            self.resets += 1
        started = time.monotonic()
        actions = self._serve.chunk(obs.episode_id, self._case_dir)
        self.device_s += time.monotonic() - started
        self.chunks += 1
        if self.chunks % 500 == 0:
            log.info("unit stats %s", json.dumps(self.stats()))
        return actions, case

    def infer_one(self, obs: Observation, noise: NDArray[np.float32] | None = None) -> NDArray[np.float64]:
        actions, case = self.normalized_chunk(obs, noise)
        served = decode_actions(self._transform.unnormalize(actions, case["state"]),
                                np.asarray(obs.joint_position, np.float64), self._decode)
        if not np.isfinite(served).all():
            raise RuntimeError(f"{obs.episode_id} chunk {obs.chunk_index}: device chunk is not finite")
        return served

    def infer(self, batch: list[Observation]) -> list[NDArray[np.float64]]:
        out: list[NDArray[np.float64] | None] = [None] * len(batch)
        order = sorted(range(len(batch)), key=lambda i: (batch[i].episode_id, batch[i].chunk_index))
        for index in order:
            out[index] = self.infer_one(batch[index])
        return [chunk for chunk in out if chunk is not None]

    def close(self) -> None:
        self._serve.quit()


def fake_batch(count: int, chunk_index: int = 0, instruction: str = "Put the lizards in the bin") -> list[Observation]:
    generator = np.random.default_rng(7)
    return [
        Observation(
            episode_id=f"selftest-{index}",
            env_id=index,
            step=chunk_index * 15,
            chunk_index=chunk_index,
            instruction=instruction,
            seed=0,
            images={key: generator.integers(256, size=(224, 224, 3), dtype=np.uint8) for key in IMAGE_KEYS},
            joint_position=generator.normal(size=7),
            gripper_position=generator.uniform(size=1),
        )
        for index in range(count)
    ]


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def gate(policy: OvPolicy, reference: pathlib.Path, acc_root: pathlib.Path,
         torch_ref: pathlib.Path, direct: pathlib.Path | None, floor: float) -> dict[str, Any]:
    """Eight real observations, three independent checks.

    1. TRANSFORM -- the Unit's CPU transform against openpi's own transformed inputs in
       the gate reference.  Images and noise are checkpoint-independent and must be exact.
       The reference and the acc_A cases were built with checkpoint A's normalization: on
       ``--checkpoint A`` the state digits and therefore the token ids MUST match exactly
       and that is a pass condition; on ``--checkpoint B`` the count of differing token ids
       is only reported, because the quantiles are a different checkpoint's.
    2. DEVICE, BITWISE -- the same eight acc_A case directories fed straight to the
       server, compared bit for bit with the outputs the Unit's own client got.
    3. DEVICE, CORRIDOR -- those outputs against the bf16 eager PyTorch reference for THIS
       checkpoint (``<torch_ref>/accA-s??.{b5,a5}.npy``), all-32 and task dims.
    """
    z = np.load(reference)
    meta = json.loads(bytes(z["__meta__"]).decode())
    profile = policy._profile  # noqa: SLF001
    transform = policy._transform  # noqa: SLF001
    suffix = profile["torch_ref_suffix"]
    corridor_key = f"accA_rel_l2_vs_bf16_eager_{profile['id']}"
    corridor_task_key = f"accA_rel_l2_task_vs_bf16_eager_{profile['id']}"
    rows = []
    for sample in meta["samples"]:
        tag, index = sample["tag"], sample["control_index"]
        obs = Observation(
            episode_id=f"gate-{meta['trajectory']}",
            env_id=0,
            step=index * 15,
            chunk_index=index,
            instruction=sample["instruction"],
            seed=0,
            images={key: z[f"{tag}.image.{key}"] for key in IMAGE_KEYS},
            joint_position=z[f"{tag}.joint_position"],
            gripper_position=z[f"{tag}.gripper_position"],
        )
        noise = np.load(acc_root / tag / "noise.npy")
        actions, case = policy.normalized_chunk(obs, noise)
        served = decode_actions(transform.unnormalize(actions, case["state"]),
                                np.asarray(obs.joint_position, np.float64), policy._decode)  # noqa: SLF001
        acc_tokens = np.load(acc_root / tag / "tokens.npy")
        acc_valid = int((acc_root / tag / "text_valid.txt").read_text().split()[0])
        checks = {
            "images_max_abs_diff_vs_accA": float(np.max(np.abs(
                case["siglip.in"] - np.load(acc_root / tag / "siglip.in.npy")))),
            "noise_equal_accA": bool(np.array_equal(case["noise"], noise)),
            "text_valid": case["text_valid"],
            "text_valid_accA_checkpointA": acc_valid,
            "routed_window": route_window(case["text_valid"], transform.windows),
            "token_ids_differing_vs_accA_checkpointA": int(np.sum(
                case["tokens"][:acc_valid] != acc_tokens[:acc_valid])),
        }
        # 2 + 3: the acc_A case itself, straight through the server.
        device = np.fromfile(acc_root / tag / "actions.f32", dtype="<f4").reshape(HORIZON, ACTION_DIM) \
            if (acc_root / tag / "actions.f32").exists() else None
        row = {
            "tag": tag,
            "control_index": index,
            "transform": checks,
            "served_abs_max_rad": float(np.max(np.abs(served[:, :7]))),
            # The model's own space, BEFORE the checkpoint's AbsoluteActions: joint
            # velocity for B, the predicted joint delta for A.  Never a copy of the
            # served number.
            profile["model_space_metric"]: float(np.max(np.abs(
                transform.unnormalize_actions(actions)[:, :7]))),
            "finite": bool(np.isfinite(served).all()),
        }
        if direct is not None:
            expected = np.load(direct / f"{tag}.npy")
            accA_case = policy._serve.chunk(f"accA-{tag}", acc_root / tag)  # noqa: SLF001
            row["accA_bitwise_vs_direct_server"] = bool(np.array_equal(accA_case, expected))
            row["accA_rel_l2_vs_direct_server"] = rel_l2(accA_case, expected)
            reference_torch = np.load(torch_ref / f"accA-{tag}.{suffix}.npy").reshape(HORIZON, ACTION_DIM)
            row[corridor_key] = rel_l2(accA_case, reference_torch)
            row[corridor_task_key] = rel_l2(accA_case[:, :8], reference_torch[:, :8])
        elif device is not None:
            row["accA_actions_present"] = True
        rows.append(row)
        log.info("gate %s", json.dumps(row))

    def mean(key: str) -> float | None:
        values = [r[key] for r in rows if key in r]
        return float(np.mean(values)) if values else None

    summary = {
        "n": len(rows),
        "floor": floor,
        # acc_A's frames and the gate reference's raw uint8 frames are two decodes of the
        # same DROID frames and differ by one uint8 step (2/255 = 0.0078431964); the
        # ssog-A-mc2 Unit's own accepted gate records exactly this value
        # (sprilicim/gate-b580-report.json), so the bar is one step, not zero.
        "images_within_one_uint8_step": all(
            r["transform"]["images_max_abs_diff_vs_accA"] <= 2.0 / 255 + 1e-6 for r in rows),
        "images_max_abs_diff_vs_accA": max(
            r["transform"]["images_max_abs_diff_vs_accA"] for r in rows),
        "noise_exact": all(r["transform"]["noise_equal_accA"] for r in rows),
        "routing": sorted({r["transform"]["routed_window"] for r in rows}),
        "accA_bitwise_vs_direct_server": all(r.get("accA_bitwise_vs_direct_server", False) for r in rows),
        "tokens_exact_vs_accA": all(
            r["transform"]["token_ids_differing_vs_accA_checkpointA"] == 0 for r in rows),
        corridor_key: {
            "mean": mean(corridor_key),
            "worst": max((r[corridor_key] for r in rows if corridor_key in r), default=None),
            "task_mean": mean(corridor_task_key),
        },
        "all_finite": all(r["finite"] for r in rows),
        "rows": rows,
        "torch_unit_in_reference": meta.get("torch_unit", {}).get("identity_sha256"),
        "reference_note": profile["reference_note"],
    }
    corridor = summary[corridor_key]["mean"]
    summary["pass"] = bool(
        summary["images_within_one_uint8_step"] and summary["noise_exact"] and summary["all_finite"]
        and summary["accA_bitwise_vs_direct_server"]
        and (summary["tokens_exact_vs_accA"] or not profile["tokens_must_match_accA"])
        and corridor is not None and corridor <= floor
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ssog-unit-ov", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="policy name; must equal the request's policy.name")
    parser.add_argument("--gpu", type=int, required=True, help="OpenVINO GPU index handed to serve_mux.sh")
    parser.add_argument("--checkpoint", choices=sorted(CHECKPOINTS), default="B",
                        help="B = lerobot/pi05_droid (joint VELOCITY, the shipped line); "
                             "A = openpi pi05_droid_jointpos (joint POSITION, AbsoluteActions)")
    parser.add_argument("--assets", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parent / "assets_ckptb",
                        help="norm_stats.json + paligemma_tokenizer.model for --checkpoint "
                             "(assets_ckptb for B, assets_ckptA for A); the norm_stats sha256 is verified")
    parser.add_argument("--serve-launch", type=pathlib.Path, default=pathlib.Path("/workflow/ovmux/serve_mux.sh"))
    parser.add_argument("--action-decode", choices=sorted(DECODES), default=None,
                        help="default: integrate_15hz for checkpoint B, native_joint_position for checkpoint A")
    parser.add_argument("--work-dir", type=pathlib.Path, default=None)
    parser.add_argument("--server-log", type=pathlib.Path, default=None)
    parser.add_argument("--max-batch", type=int, default=32)
    parser.add_argument("--noise-salt", default="ov-w8a8-c2-mux/seed0")
    parser.add_argument("--metadata-out", type=pathlib.Path, default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="register with HQ and serve until the eval ends")
    serve.add_argument("--hq", required=True)
    serve.add_argument("--token-file", required=True)
    check = commands.add_parser("selftest", help="shapes, routing, timing and determinism")
    check.add_argument("--batch", type=int, default=4)
    check.add_argument("--steps", type=int, default=4)
    gate_cmd = commands.add_parser("gate", help="fidelity gate on real observations")
    gate_cmd.add_argument("--reference", type=pathlib.Path, required=True, help="gate_ref.npz")
    gate_cmd.add_argument("--acc", type=pathlib.Path, required=True, help="acc_A case root (s00..s39)")
    gate_cmd.add_argument("--torch-ref", type=pathlib.Path, required=True,
                          help="directory of the bf16 eager references for this checkpoint: "
                               "accA-s??.b5.npy for B, accA-s??.a5.npy for A")
    gate_cmd.add_argument("--direct", type=pathlib.Path, default=None,
                          help="directory of the server's own outputs (ov_mux_client fidelity --out)")
    gate_cmd.add_argument("--floor", type=float, default=0.05)
    gate_cmd.add_argument("--report", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)
    if args.action_decode is None:
        args.action_decode = CHECKPOINTS[args.checkpoint]["decodes"][0]

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    work = args.work_dir or pathlib.Path(f"/dev/shm/ssog-unit-ov-{args.gpu}")
    work.mkdir(parents=True, exist_ok=True)
    server_log = args.server_log or pathlib.Path(__file__).resolve().parent / f"ov-server-{args.gpu}.log"
    policy = OvPolicy(
        args.name,
        Assets(args.assets, args.serve_launch, args.checkpoint),
        args.gpu,
        args.max_batch,
        args.noise_salt,
        work / "case",
        server_log,
        action_decode=args.action_decode,
    )
    try:
        log.info("SERVED METADATA %s", json.dumps(policy.metadata, default=str, sort_keys=True))
        log.info("POLICY SPEC %s", policy.spec.model_dump_json())
        if args.metadata_out is not None:
            args.metadata_out.write_text(json.dumps({
                "served_metadata": json.loads(json.dumps(policy.metadata, default=str)),
                "policy_spec": json.loads(policy.spec.model_dump_json()),
                "method": METHOD,
            }, indent=1, sort_keys=True))

        if args.command == "selftest":
            started = time.monotonic()
            served = 0
            for chunk_index in range(args.steps):
                chunks = policy.infer(fake_batch(args.batch, chunk_index))
                for chunk in chunks:
                    if chunk.shape != (HORIZON, SERVED_DIM):
                        raise SystemExit(f"chunk shape {chunk.shape} is not the declared spec")
                served += len(chunks)
                log.info("chunk_index=%d served=%d stats=%s", chunk_index, len(chunks), json.dumps(policy.stats()))
            elapsed = time.monotonic() - started
            # The graph is stateless: the same observation must come back bit-identical,
            # and a long prompt must route to a wider window without changing that.
            observation = fake_batch(1, 0)[0]
            first, _ = policy.normalized_chunk(observation, policy.noise("determinism", 0))
            second, _ = policy.normalized_chunk(observation, policy.noise("determinism", 0))
            long_prompt = fake_batch(1, 0, instruction=" ".join(["pick up the small red block and place it"] * 6))[0]
            long_chunk, long_case = policy.normalized_chunk(long_prompt, policy.noise("determinism", 0))
            report = {
                "ok": True,
                "chunks": served,
                "seconds": elapsed,
                "ms_per_chunk": elapsed / served * 1000,
                "stats": policy.stats(),
                "deterministic": bool(np.array_equal(first, second)),
                "long_prompt_text_valid": long_case["text_valid"],
                "long_prompt_window": route_window(long_case["text_valid"], policy._transform.windows),  # noqa: SLF001
                "long_prompt_finite": bool(np.isfinite(long_chunk).all()),
                "identity_sha256": policy.spec.identity_sha256,
                "server": policy.metadata["server"],
            }
            if not report["deterministic"]:
                raise SystemExit(f"the stateless graph returned two different chunks: {report}")
            print(json.dumps(report, indent=1))
            return 0

        if args.command == "gate":
            report = gate(policy, args.reference, args.acc, args.torch_ref, args.direct, args.floor)
            report["identity_sha256"] = policy.spec.identity_sha256
            report["server"] = policy.metadata["server"]
            text = json.dumps(report, indent=1)
            if args.report is not None:
                args.report.write_text(text)
            print(text)
            return 0 if report["pass"] else 3

        token = pathlib.Path(args.token_file).read_text().strip()
        worker = PolicyWorker(policy, HttpHqPolicyClient(args.hq, token))
        try:
            worker.run()
        finally:
            log.info("unit stats at exit: %s", json.dumps(policy.stats()))
        return 0
    finally:
        policy.close()


if __name__ == "__main__":
    sys.exit(main())
