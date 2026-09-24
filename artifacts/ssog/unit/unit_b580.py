#!/usr/bin/env python3
"""The SpriliCim policy worker ("the Unit") serving ``ssog-A-mc2`` from the B580 kernels.

The policy forward is the device chunk: a resident ``fused --serve`` process per B580 card
(MegaTower5's RuntimeCore line: config A quantization, on-device VLA-Pruner, MC2 sampler,
15 rungs) driven over JSON lines on its stdin/stdout. This adapter does on the CPU exactly
what the served torch Units do around the model:

* openpi's DROID input transform -- ``DroidInputs`` (state = joint[7] ++ gripper[1]),
  quantile ``Normalize`` on the 8 real state dims, the pi05 prompt with the discretized state
  (``Task: <prompt>, State: <256-bin digits>;\\nAction: ``), the strict PaliGemma tokenizer at
  ``max_token_len 64``, uint8 frames to [-1, 1] CHW, the pad camera dropped
  (``drop_pad_camera``), the Gemma embedding rows scaled by bf16(sqrt(2048)) = 45.25
  (DataPrep's ``export_acc.py`` recipe, which is what the device consumes as ``text_embed``);
* openpi's output transform -- quantile ``Unnormalize`` (32-wide statistics) of the chunk
  and of the state, ``AbsoluteActions`` on the 7 joints, the first 8 dims served;
* one temporal state per ``episode_id``: the pruner ring lives in the fused process, keyed
  by episode id; ``chunk_index == 0`` resets it (a re-attempt replays from chunk 0 under the
  same id, so the reset is explicit, not "first time seen").

The flow noise: the torch Unit draws ``torch.normal`` on the device per chunk, unseeded.
This Unit draws the same N(0, 1) float32 [15, 32] per chunk from a PCG64 stream seeded by
sha256(salt | episode_id | chunk_index), so a redelivered or replayed observation is served
bit-identically and the draw is recorded in the identity.

The identity binds the served binary (its md5 from the ready line), the weight files the
binary loads (``INDEX.json``'s per-file sha256, verified at start), the checkpoint provenance,
the normalization statistics, the tokenizer, the transform recipe and the noise policy.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import pathlib
import selectors
import struct
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

log = logging.getLogger("ssog.unit_b580")

HORIZON = 15
ACTION_DIM = 32
SERVED_DIM = 8
MAX_TOKEN_LEN = 64
EMBED_SCALE = np.float32(45.25)  # bf16(sqrt(2048)), the scale the device's text rows carry
LINE = "ssog-A-mc2"

# The served configuration, as data, so it lands in the identity hash. It is config A's
# METHOD from unit.py plus the MC2 sampler unit_supervise_mc2.sh serves, executed by the
# device line instead of the fake-quant torch stack.
METHOD: dict[str, Any] = {
    "config": "pi05_droid_jointpos",
    "line": LINE,
    "execution": "b580_fused_serve",
    "vla_pruner_keep_ratio": 0.5,
    "vla_pruner_prune_layer": 3,
    "vla_pruner_history_window": 3,
    "vla_pruner_history_decay": 0.8,
    # The device prunes from an episode's first chunk with a zero ring; the torch stack
    # runs the first three observations dense while the ring fills (contract 4.3).
    "vla_pruner_warmup": "cold_ring",
    "w4a8": "rtn",
    "w4a8_scope": ["siglip", "prefix", "expert", "flow_head"],
    "w4a8_bits": {"siglip": 8, "prefix": 8, "expert": 4, "flow_head": 8},
    "num_steps": 10,
    "sampler": {
        "name": "meancache",
        "nodes": [0, 5, 10],
        "schedule": "path",
        "gamma": 4.0,
        "jvp": "cached",
        "jvp_scale": 0.65,
    },
    "drop_pad_camera": True,
    "max_token_len": MAX_TOKEN_LEN,
}

TRANSFORM: dict[str, Any] = {
    "inputs": [
        "DroidInputs: state = joint_position[7] ++ gripper_position[1]; cameras base_0_rgb, left_wrist_0_rgb",
        "Normalize(use_quantiles): (x - q01) / (q99 - q01 + 1e-6) * 2 - 1 on the first 8 state dims",
        "images 224x224 uint8 RGB -> float32 / 255 * 2 - 1, HWC -> CHW, pad camera dropped",
        "PaligemmaTokenizer(64, strict): 'Task: <prompt>, State: <digitize(state, linspace(-1,1,257)[:-1]) - 1>;\\nAction: ', BOS, zero-padded",
        "text_embed = bf16_to_f32(gemma.embed.w[tokens]) * 45.25",
    ],
    "outputs": [
        "Unnormalize(use_quantiles, 32-wide): (x + 1) / 2 * (q99 - q01 + 1e-6) + q01 on actions and state",
        "AbsoluteActions(mask 7 x True, 1 x False): actions[:, :7] += state[:7]",
        "DroidOutputs: actions[:, :8]",
    ],
    "noise": "N(0,1) float32 [15,32] from numpy PCG64 seeded by sha256(salt|episode_id|chunk_index)[:8]",
}


def sha256_file(path: pathlib.Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def bf16_to_f32(u: np.ndarray) -> np.ndarray:
    return (u.astype(np.uint32) << 16).view(np.float32)


class EmbedTable:
    """The Gemma embedding table as the device's weight export stores it: bf16 bits, memory-mapped."""

    def __init__(self, path: pathlib.Path) -> None:
        with path.open("rb") as stream:
            (header_len,) = struct.unpack("<Q", stream.read(8))
            header = json.loads(stream.read(header_len))
        entry = header["gemma.embed.w"]
        if entry["dtype"] != "U16" or len(entry["shape"]) != 2:
            raise RuntimeError(f"unexpected embed table layout {entry}")
        start, end = entry["data_offsets"]
        self.rows, self.dim = entry["shape"]
        if end - start != self.rows * self.dim * 2:
            raise RuntimeError("embed table byte count does not match its shape")
        self._table = np.memmap(path, dtype="<u2", mode="r", offset=8 + header_len + start, shape=(self.rows, self.dim))

    def __call__(self, tokens: NDArray[np.int64]) -> NDArray[np.float32]:
        return (bf16_to_f32(np.asarray(self._table[tokens])) * EMBED_SCALE).astype(np.float32)


class Transform:
    """openpi's DROID pi05 input/output transforms, on the CPU, without openpi."""

    def __init__(self, assets: pathlib.Path, embed: EmbedTable) -> None:
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
        self._embed = embed
        self._bins = np.linspace(-1, 1, 256 + 1)[:-1]

    def normalize_state(self, joint: NDArray[np.float64], gripper: NDArray[np.float64]) -> NDArray[np.float64]:
        state = np.concatenate([joint.astype(np.float32), gripper.astype(np.float32)])
        q01, q99 = self.state_q01[: state.shape[-1]], self.state_q99[: state.shape[-1]]
        return (state - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0

    def tokenize(self, prompt: str, state: NDArray[np.float64]) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
        discretized = np.digitize(state, bins=self._bins) - 1
        full = f"Task: {cleaned}, State: {' '.join(map(str, discretized))};\nAction: "
        tokens = self._tokenizer.encode(full, add_bos=True)
        if len(tokens) > MAX_TOKEN_LEN:
            raise ValueError(f"Token length ({len(tokens)}) exceeds max length ({MAX_TOKEN_LEN}).")
        padding = MAX_TOKEN_LEN - len(tokens)
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
            "text_embed": self._embed(tokens),
            "state": state32,
            "noise": np.ascontiguousarray(noise, dtype=np.float32),
        }

    def served(self, actions: NDArray[np.float32], state32: NDArray[np.float64]) -> NDArray[np.float64]:
        """openpi's output transforms on the model's normalized chunk."""
        if actions.shape != (HORIZON, ACTION_DIM):
            raise RuntimeError(f"device chunk is {actions.shape}, expected ({HORIZON}, {ACTION_DIM})")
        actions = (actions + 1.0) / 2.0 * (self.action_q99 - self.action_q01 + 1e-6) + self.action_q01
        state = (state32 + 1.0) / 2.0 * (self.state_q99 - self.state_q01 + 1e-6) + self.state_q01
        mask = np.asarray([True] * 7 + [False])
        actions[..., : mask.shape[-1]] += np.expand_dims(np.where(mask, state[: mask.shape[-1]], 0), axis=-2)
        return np.asarray(actions[..., :SERVED_DIM], dtype=np.float64)


def write_case(directory: pathlib.Path, case: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "actions.f32").unlink(missing_ok=True)
    for name in ("siglip.in", "tokens", "tokens_mask", "text_embed", "state", "noise"):
        np.save(directory / f"{name}.npy", np.ascontiguousarray(case[name]))
    (directory / "text_valid.txt").write_text(f"{case['text_valid']}\n")


class FusedServe:
    """One resident ``fused --serve`` on one card, spoken to over JSON lines."""

    def __init__(self, launch: pathlib.Path, gpu: int, stderr_log: pathlib.Path, ready_timeout_s: float = 600.0) -> None:
        self._gpu = gpu
        self._reap_stale_containers()
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
            raise RuntimeError(f"fused --serve did not report ready: {self.ready}")
        log.info("fused ready: %s", json.dumps(self.ready, sort_keys=True))
        self.chunks = 0
        self.device_ms = 0.0
        self.implicit_resets = 0

    def _read_line(self, timeout_s: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while True:
            if self._process.poll() is not None:
                raise RuntimeError(f"fused --serve exited with {self._process.returncode}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.kill()
                raise RuntimeError(f"fused --serve answered nothing in {timeout_s:.0f}s; killed")
            if not self._selector.select(timeout=min(remaining, 1.0)):
                continue
            line = self._process.stdout.readline()
            if not line:
                raise RuntimeError("fused --serve closed its stdout")
            text = line.decode(errors="replace").strip()
            if not text:
                continue
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                log.warning("non-JSON line from fused stdout: %r", text)

    def _request(self, message: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        with self._lock:
            self._process.stdin.write((json.dumps(message) + "\n").encode())
            self._process.stdin.flush()
            reply = self._read_line(timeout_s)
        if not reply.get("ok"):
            raise RuntimeError(f"fused --serve refused {message}: {reply}")
        return reply

    def reset(self, episode: str) -> None:
        self._request({"cmd": "reset", "episode": episode}, timeout_s=60.0)

    def chunk(self, episode: str, directory: pathlib.Path, timeout_s: float = 120.0) -> NDArray[np.float32]:
        reply = self._request({"cmd": "chunk", "episode": episode, "dir": str(directory)}, timeout_s)
        self.chunks += 1
        self.device_ms += float(reply.get("ms", 0.0))
        if reply.get("reset"):
            self.implicit_resets += 1
        actions = np.fromfile(directory / "actions.f32", dtype="<f4")
        if actions.shape != (HORIZON * ACTION_DIM,):
            raise RuntimeError(f"actions.f32 holds {actions.shape[0]} floats, expected {HORIZON * ACTION_DIM}")
        return actions.reshape(HORIZON, ACTION_DIM)

    def quit(self) -> None:
        if self._process.poll() is None:
            try:
                self._request({"cmd": "quit"}, timeout_s=30.0)
                self._process.wait(timeout=30.0)
            except Exception:
                log.warning("fused --serve did not quit cleanly", exc_info=True)
                self.kill()
        self._log.close()

    def _containers(self) -> list[str]:
        """serve_launch.sh names its container ssog-serve-gpu<N>-<pid of its docker client>."""
        listed = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}", "--filter", f"name=^ssog-serve-gpu{self._gpu}-"],
            capture_output=True, text=True, check=False,
        )
        return [name for name in listed.stdout.split() if name]

    def _reap_stale_containers(self) -> None:
        """A container whose docker client is gone (the Unit was SIGKILLed) still holds the
        card and 11 GB of host RAM; the flock it was launched under is already free, so a
        fresh launch would land beside it. A live client (another Unit on this card) is
        left alone -- the launch then waits on the flock, which is the intended outcome."""
        for name in self._containers():
            alive = subprocess.run(["pgrep", "-f", f"name {name}"], capture_output=True, check=False).returncode == 0
            if alive:
                log.warning("container %s belongs to a live launch on this card; not touching it", name)
                continue
            log.warning("killing stale fused --serve container %s", name)
            subprocess.run(["docker", "kill", name], capture_output=True, check=False)

    def kill(self) -> None:
        if self._process.poll() is None:
            self._process.kill()
            self._process.wait()
        # Killing the docker client leaves the container (and the card) behind.
        for name in self._containers():
            if subprocess.run(["pgrep", "-f", f"name {name}"], capture_output=True, check=False).returncode != 0:
                subprocess.run(["docker", "kill", name], capture_output=True, check=False)

    def stats(self) -> dict[str, Any]:
        return {
            "device_chunks": self.chunks,
            "device_ms_per_chunk": self.device_ms / self.chunks if self.chunks else None,
            "implicit_cold_resets": self.implicit_resets,
        }


@dataclasses.dataclass
class Assets:
    root: pathlib.Path
    weights: pathlib.Path
    launch: pathlib.Path

    def digests(self, verify_weights: bool) -> dict[str, Any]:
        provenance = json.loads((self.root / "provenance.json").read_text())
        index = json.loads((self.weights / "INDEX.json").read_text())
        weights: dict[str, str] = {}
        for name, entry in sorted(index.items()):
            path = self.weights / name
            if not path.is_file():
                if name.startswith("unserved"):
                    continue
                raise RuntimeError(f"weight file {path} is missing")
            if verify_weights:
                actual = sha256_file(path)
                if actual != entry["sha256_file"]:
                    raise RuntimeError(f"{path} is {actual}, INDEX.json says {entry['sha256_file']}")
            weights[name] = entry["sha256_file"]
        normalization = sha256_file(self.root / "norm_stats.json")
        if normalization != provenance["normalization_sha256"]:
            raise RuntimeError("norm_stats.json is not the checkpoint's (provenance sha mismatch)")
        return {
            "checkpoint_model_sha256": provenance["model_sha256"],
            "checkpoint_provenance_sha256": sha256_file(self.root / "provenance.json"),
            "checkpoint_normalization_sha256": normalization,
            "tokenizer_sha256": sha256_file(self.root / "paligemma_tokenizer.model"),
            "weights_index_sha256": sha256_file(self.weights / "INDEX.json"),
            "weights": weights,
            "serve_launch_sha256": sha256_file(self.launch),
        }


class B580Policy(Policy):
    def __init__(
        self,
        name: str,
        assets: Assets,
        gpu: int,
        max_batch: int,
        noise_salt: str,
        work_dir: pathlib.Path,
        stderr_log: pathlib.Path,
        verify_weights: bool = True,
    ) -> None:
        self._digests = assets.digests(verify_weights)
        self._embed = EmbedTable(assets.weights / "gemma_embed.safetensors")
        self._transform = Transform(assets.root, self._embed)
        self._fused = FusedServe(assets.launch, gpu, stderr_log)
        if self._fused.ready.get("line") != LINE:
            raise RuntimeError(f"fused --serve serves line {self._fused.ready.get('line')!r}, expected {LINE!r}")
        self._case_dir = work_dir
        self._case_dir.mkdir(parents=True, exist_ok=True)
        self._noise_salt = noise_salt
        self._episodes: dict[str, int] = {}
        self.chunks = 0
        self.resets = 0
        self.orphaned = 0
        self.replayed = 0
        self.transform_s = 0.0
        self.device_s = 0.0
        self._metadata = {
            "config": METHOD["config"],
            "line": LINE,
            "backend": "b580_fused_serve",
            "fused": self._fused.ready,
            "method": METHOD,
            "transform": TRANSFORM,
            "noise_salt": noise_salt,
            "action_horizon": HORIZON,
            "action_dim": SERVED_DIM,
            "action_space": "absolute_joint_position_radians_7_and_absolute_gripper_0_open_1_closed",
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
        """Bind the eval record to the binary, the weights it loads AND the exact serving recipe."""
        body = {
            "line": metadata["line"],
            "backend": metadata["backend"],
            "fused_md5": metadata["fused"].get("md5"),
            "fused_sampler": metadata["fused"].get("sampler"),
            "checkpoint_model_sha256": metadata["checkpoint_model_sha256"],
            "checkpoint_normalization_sha256": metadata["checkpoint_normalization_sha256"],
            "checkpoint_provenance_sha256": metadata["checkpoint_provenance_sha256"],
            "tokenizer_sha256": metadata["tokenizer_sha256"],
            "weights": metadata["weights"],
            "method": metadata["method"],
            "transform": metadata["transform"],
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
            "episodes_tracked": len(self._episodes),
            "chunks_replayed": self.replayed,
            "histories_orphaned": self.orphaned,
            "transform_ms_per_chunk": self.transform_s / self.chunks * 1000 if self.chunks else None,
            "device_roundtrip_ms_per_chunk": self.device_s / self.chunks * 1000 if self.chunks else None,
            **self._fused.stats(),
        }

    def infer_one(self, obs: Observation, noise: NDArray[np.float32] | None = None) -> NDArray[np.float64]:
        """One observation, its own episode's ring installed in the fused process."""
        if noise is None:
            noise = self.noise(obs.episode_id, obs.chunk_index)
        started = time.monotonic()
        case = self._transform.case(obs, noise)
        write_case(self._case_dir, case)
        self.transform_s += time.monotonic() - started
        pending = self._episodes.get(obs.episode_id)
        if obs.chunk_index == 0:
            # A fresh episode, or a re-attempt replaying its episode from chunk 0 under the
            # same id: either way the ring starts from zero, exactly like the first attempt.
            self._fused.reset(obs.episode_id)
            self.resets += 1
            if pending is not None:
                self.replayed += 1
        elif pending is None:
            # Worker restart (or the fused process's) lost the ring; the device starts it cold.
            self.orphaned += 1
            log.warning("no ring for %s chunk %d; the device restarts it cold", obs.episode_id, obs.chunk_index)
        elif pending != obs.chunk_index:
            # HQ redelivered a leased observation or replayed a chunk: the ring already
            # advanced past it. Served with the ring as it is; counted so it is visible.
            self.replayed += 1
            log.warning("%s: expected chunk %d, serving chunk %d", obs.episode_id, pending, obs.chunk_index)
        started = time.monotonic()
        actions = self._fused.chunk(obs.episode_id, self._case_dir)
        self.device_s += time.monotonic() - started
        self._episodes[obs.episode_id] = obs.chunk_index + 1
        if len(self._episodes) > 4096:
            del self._episodes[next(iter(self._episodes))]
        self.chunks += 1
        if self.chunks % 500 == 0:
            log.info("unit stats %s", json.dumps(self.stats()))
        served = self._transform.served(actions, case["state"])
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
        self._fused.quit()


def fake_batch(count: int, chunk_index: int = 0) -> list[Observation]:
    generator = np.random.default_rng(7)
    return [
        Observation(
            episode_id=f"selftest-{index}",
            env_id=index,
            step=chunk_index * 15,
            chunk_index=chunk_index,
            instruction="Put the lizards in the bin",
            seed=0,
            images={key: generator.integers(256, size=(224, 224, 3), dtype=np.uint8) for key in IMAGE_KEYS},
            joint_position=generator.normal(size=7),
            gripper_position=generator.uniform(size=1),
        )
        for index in range(count)
    ]


def isolation_check(policy: B580Policy, episodes: int, steps: int) -> dict[str, Any]:
    """The ring travels with its episode: interleaved and sequential passes must agree exactly."""
    generator = np.random.default_rng(1234)
    noise = {(e, c): generator.standard_normal((HORIZON, ACTION_DIM), dtype=np.float32) for e in range(episodes) for c in range(steps)}
    grid = {c: fake_batch(episodes, c) for c in range(steps)}
    interleaved = {(e, c): policy.infer_one(grid[c][e], noise[(e, c)]) for c in range(steps) for e in range(episodes)}
    sequential = {(e, c): policy.infer_one(grid[c][e], noise[(e, c)]) for e in range(episodes) for c in range(steps)}
    worst = max(float(np.max(np.abs(interleaved[k] - sequential[k]))) for k in interleaved)
    digest = hashlib.sha256()
    for key in sorted(interleaved):
        digest.update(np.asarray(interleaved[key], dtype=np.float64).tobytes())
    return {
        "episodes": episodes,
        "steps": steps,
        "max_abs_diff_interleaved_vs_sequential": worst,
        "pinned_output_sha256": digest.hexdigest(),
    }


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def gate(policy: B580Policy, reference: pathlib.Path, acc_root: pathlib.Path | None, floor: float) -> dict[str, Any]:
    """The fidelity gate: 8 real observations through this Unit vs the torch Unit and acc_A."""
    z = np.load(reference)
    meta = json.loads(bytes(z["__meta__"]).decode())
    transform = policy._transform  # noqa: SLF001
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
        noise = z[f"{tag}.noise"]
        case = transform.case(obs, noise)
        # 1. the CPU transform against openpi's own transformed inputs
        checks = {
            "tokens_equal": bool(np.array_equal(case["tokens"], z[f"{tag}.in.tokenized_prompt"])),
            "mask_equal": bool(np.array_equal(case["tokens_mask"], z[f"{tag}.in.tokenized_prompt_mask"])),
            "state_max_abs_diff": float(np.max(np.abs(case["state"] - z[f"{tag}.in.state"]))),
        }
        if acc_root is not None and (acc_root / tag / "siglip.in.npy").is_file():
            checks["acc_siglip_in_max_abs_diff"] = float(np.max(np.abs(case["siglip.in"] - np.load(acc_root / tag / "siglip.in.npy"))))
            checks["acc_tokens_equal"] = bool(np.array_equal(case["tokens"], np.load(acc_root / tag / "tokens.npy")))
            checks["acc_text_embed_max_abs_diff"] = float(np.max(np.abs(case["text_embed"] - np.load(acc_root / tag / "text_embed.npy"))))
            checks["acc_noise_equal"] = bool(np.array_equal(case["noise"], np.load(acc_root / tag / "noise.npy")))
            checks["acc_text_valid"] = int((acc_root / tag / "text_valid.txt").read_text().strip()) == case["text_valid"]
        # 2. the device chunk, as served
        served = policy.infer_one(obs, noise)
        device_normalized = np.fromfile(policy._case_dir / "actions.f32", dtype="<f4").reshape(HORIZON, ACTION_DIM)  # noqa: SLF001
        torch_served = z[f"{tag}.actions_served"]
        torch_normalized = z[f"{tag}.actions_normalized"]
        row = {
            "tag": tag,
            "control_index": index,
            "torch_warmup": sample["pruner"].get("warmup"),
            "transform": checks,
            "served_rel_l2_vs_torch_unit": rel_l2(served, torch_served),
            "served_rel_l2_vs_torch_unit_no_gripper": rel_l2(served[:, :7], torch_served[:, :7]),
            "normalized_rel_l2_vs_torch_unit": rel_l2(device_normalized, torch_normalized),
            "normalized_rel_l2_vs_torch_unit_no_col7": rel_l2(np.delete(device_normalized, 7, axis=1), np.delete(torch_normalized, 7, axis=1)),
            # the post-processing alone: the torch Unit's normalized chunk through this Unit's output transform
            "postprocess_max_abs_diff": float(np.max(np.abs(transform.served(torch_normalized.copy(), case["state"]) - torch_served))),
        }
        if acc_root is not None and (acc_root / tag / "actions.configAmc2_torch.npy").is_file():
            acc = np.load(acc_root / tag / "actions.configAmc2_torch.npy")
            row["normalized_rel_l2_vs_acc_configAmc2_torch"] = rel_l2(device_normalized, acc)
            row["normalized_rel_l2_vs_acc_configAmc2_torch_no_col7"] = rel_l2(np.delete(device_normalized, 7, axis=1), np.delete(acc, 7, axis=1))
            row["torch_unit_vs_acc_configAmc2_torch"] = rel_l2(torch_normalized, acc)
        rows.append(row)
        log.info("gate %s", json.dumps(row))

    def mean(key: str, subset=None) -> float | None:
        values = [r[key] for r in rows if key in r and (subset is None or subset(r))]
        return float(np.mean(values)) if values else None

    summary = {
        "n": len(rows),
        "floor": floor,
        "served_vs_torch_unit": {
            "mean": mean("served_rel_l2_vs_torch_unit"),
            "worst": max(r["served_rel_l2_vs_torch_unit"] for r in rows),
            "mean_no_gripper": mean("served_rel_l2_vs_torch_unit_no_gripper"),
        },
        "normalized_vs_torch_unit": {
            "mean": mean("normalized_rel_l2_vs_torch_unit"),
            "worst": max(r["normalized_rel_l2_vs_torch_unit"] for r in rows),
            "mean_no_col7": mean("normalized_rel_l2_vs_torch_unit_no_col7"),
            "mean_warmup_steps_0_2": mean("normalized_rel_l2_vs_torch_unit", lambda r: r["control_index"] < 3),
            "mean_steady_steps_3plus": mean("normalized_rel_l2_vs_torch_unit", lambda r: r["control_index"] >= 3),
        },
        "normalized_vs_acc_configAmc2_torch": {
            "mean": mean("normalized_rel_l2_vs_acc_configAmc2_torch"),
            "mean_no_col7": mean("normalized_rel_l2_vs_acc_configAmc2_torch_no_col7"),
            "mean_warmup_steps_0_2": mean("normalized_rel_l2_vs_acc_configAmc2_torch", lambda r: r["control_index"] < 3),
            "mean_steady_steps_3plus": mean("normalized_rel_l2_vs_acc_configAmc2_torch", lambda r: r["control_index"] >= 3),
        },
        "transform_exact": all(r["transform"]["tokens_equal"] and r["transform"]["mask_equal"] for r in rows),
        "postprocess_worst_abs_diff": max(r["postprocess_max_abs_diff"] for r in rows),
        "torch_unit": meta["torch_unit"],
        "rows": rows,
    }
    summary["pass"] = bool(
        summary["transform_exact"]
        and summary["normalized_vs_torch_unit"]["mean"] <= floor
        and (summary["normalized_vs_acc_configAmc2_torch"]["mean"] is None or summary["normalized_vs_acc_configAmc2_torch"]["mean"] <= floor)
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ssog-unit-b580", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="policy name; must equal the request's policy.name")
    parser.add_argument("--gpu", type=int, required=True, help="level_zero card index handed to serve_launch.sh")
    parser.add_argument("--assets", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parent / "assets")
    parser.add_argument("--weights", type=pathlib.Path, default=pathlib.Path("/home/spring/ssog/campaign13/share/weights-A"))
    parser.add_argument("--serve-launch", type=pathlib.Path, default=pathlib.Path("/home/spring/ssog/campaign13/RuntimeCore/serve_launch.sh"))
    parser.add_argument("--work-dir", type=pathlib.Path, default=None, help="tmpfs case directory (default /dev/shm/ssog-unit-<gpu>)")
    parser.add_argument("--fused-log", type=pathlib.Path, default=None, help="fused --serve stderr (default <work>/fused-<gpu>.log)")
    parser.add_argument("--max-batch", type=int, default=32)
    parser.add_argument("--noise-salt", default="ssog-b580-A-mc2/seed0")
    parser.add_argument("--metadata-out", type=pathlib.Path, default=None)
    parser.add_argument("--skip-weight-check", action="store_true", help="trust INDEX.json without re-hashing 4.5 GB of weights")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="register with HQ and serve until the eval ends")
    serve.add_argument("--hq", required=True)
    serve.add_argument("--token-file", required=True)
    check = commands.add_parser("selftest", help="shapes, timing and per-episode ring isolation")
    check.add_argument("--batch", type=int, default=4)
    check.add_argument("--steps", type=int, default=6)
    gate_cmd = commands.add_parser("gate", help="fidelity gate against the torch Unit's reference on real observations")
    gate_cmd.add_argument("--reference", type=pathlib.Path, required=True, help="gate_ref.npz from gate_b580_pod.py")
    gate_cmd.add_argument("--acc", type=pathlib.Path, default=None, help="share/ref/acc_A root")
    gate_cmd.add_argument("--floor", type=float, default=0.05)
    gate_cmd.add_argument("--report", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    work = args.work_dir or pathlib.Path(f"/dev/shm/ssog-unit-{args.gpu}")
    work.mkdir(parents=True, exist_ok=True)
    fused_log = args.fused_log or pathlib.Path(__file__).resolve().parent / f"fused-{args.gpu}.log"
    policy = B580Policy(
        args.name,
        Assets(args.assets, args.weights, args.serve_launch),
        args.gpu,
        args.max_batch,
        args.noise_salt,
        work / "case",
        fused_log,
        verify_weights=not args.skip_weight_check,
    )
    try:
        log.info("SERVED METADATA %s", json.dumps(policy.metadata, default=str, sort_keys=True))
        log.info("POLICY SPEC %s", policy.spec.model_dump_json())
        if args.metadata_out is not None:
            args.metadata_out.write_text(
                json.dumps(
                    {
                        "served_metadata": json.loads(json.dumps(policy.metadata, default=str)),
                        "policy_spec": json.loads(policy.spec.model_dump_json()),
                        "method": METHOD,
                    },
                    indent=1,
                    sort_keys=True,
                )
            )

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
            stats = policy.stats()
            if stats["episodes_started"] != args.batch or stats["histories_orphaned"] or stats["implicit_cold_resets"]:
                raise SystemExit(f"per-episode bookkeeping is off: {stats}")
            isolation = isolation_check(policy, min(args.batch, 4), min(args.steps, 4))
            if isolation["max_abs_diff_interleaved_vs_sequential"] != 0.0:
                raise SystemExit(f"per-episode ring is not isolated: {isolation}")
            report = {
                "ok": True,
                "chunks": served,
                "seconds": elapsed,
                "chunks_per_second": served / elapsed,
                "ms_per_chunk": elapsed / served * 1000,
                "stats": stats,
                "isolation": isolation,
                "identity_sha256": policy.spec.identity_sha256,
                "fused": policy.metadata["fused"],
            }
            print(json.dumps(report, indent=1))
            return 0

        if args.command == "gate":
            report = gate(policy, args.reference, args.acc, args.floor)
            report["identity_sha256"] = policy.spec.identity_sha256
            report["fused"] = policy.metadata["fused"]
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
