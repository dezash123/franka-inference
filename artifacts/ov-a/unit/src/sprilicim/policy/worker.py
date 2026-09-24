"""Pull, decode, infer, push, repeat.

The worker never waits on a particular environment: whatever is pending when it
returns is what it serves next. Each pull also refreshes the worker's heartbeat
at HQ, so as long as the pull wait is shorter than HQ's staleness window no
separate heartbeat is needed. A summary line every thirty seconds says how much
time was spent waiting for work, which is what tells you the fleet is too small.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx
import numpy as np

from sprilicim.policy.base import Observation, Policy
from sprilicim.policy.transport import HqPolicyClient
from sprilicim.protocol.codec import png_decode
from sprilicim.protocol.schemas import ActionChunk, ObservationItem, PushRequest

log = logging.getLogger("sprilicim.policy")

# HQ revokes the eval's tokens when it finishes or is cancelled (401), and refuses a
# worker whose eval no longer accepts work (403, 409). None of these recover by retrying.
FINISHED = {401, 403, 409}


@dataclass
class Stats:
    items: int = 0
    batches: int = 0
    infer_s: float = 0.0
    wait_s: float = 0.0
    rejected: int = 0

    def line(self, elapsed_s: float) -> str:
        rate = self.items / elapsed_s if elapsed_s else 0.0
        mean = self.infer_s / self.batches * 1000 if self.batches else 0.0
        waiting = self.wait_s / elapsed_s * 100 if elapsed_s else 0.0
        return (
            f"served {self.items} items in {self.batches} batches, {rate:.2f} chunks/s, "
            f"{mean:.0f} ms per batch, {waiting:.0f}% of time waiting for work, "
            f"{self.rejected} rejected"
        )


def decode_item(item: ObservationItem) -> Observation:
    return Observation(
        episode_id=item.episode_id,
        env_id=item.env_id,
        step=item.step,
        chunk_index=item.chunk_index,
        instruction=item.instruction,
        seed=item.seed,
        images={key: png_decode(data) for key, data in item.images.items()},
        joint_position=np.asarray(item.joint_position, dtype=np.float64),
        gripper_position=np.asarray(item.gripper_position, dtype=np.float64),
    )


class PolicyWorker:
    def __init__(
        self,
        policy: Policy,
        client: HqPolicyClient,
        *,
        wait_s: float = 20.0,
        report_every_s: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._policy = policy
        self._client = client
        self._wait_s = wait_s
        self._report_every_s = report_every_s
        self._sleep = sleep
        self._clock = clock
        self.stats = Stats()

    def run(self, stop: threading.Event | None = None) -> None:
        stop = stop or threading.Event()
        spec = self._policy.spec
        worker_id = self._register()
        started = last_report = self._clock()
        backoff = 1.0
        while not stop.is_set():
            try:
                self._serve_once(worker_id, spec.max_batch)
                backoff = 1.0
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in FINISHED:
                    log.info("HQ answered %s: the eval is over; exiting", exc.response.status_code)
                    return
                log.warning("HQ answered %s; retrying in %.0fs", exc.response.status_code, backoff)
                self._sleep(backoff)
                backoff = min(backoff * 2, 30.0)
            except Exception:
                log.warning("transport error; retrying in %.0fs", backoff, exc_info=True)
                self._sleep(backoff)
                backoff = min(backoff * 2, 30.0)
            now = self._clock()
            if now - last_report >= self._report_every_s:
                log.info(self.stats.line(now - started))
                last_report = now

    def serve_once(self, worker_id: str) -> int:
        """One pull-infer-push cycle. Returns the number of items served."""
        return self._serve_once(worker_id, self._policy.spec.max_batch)

    def _register(self) -> str:
        response = self._client.register(self._policy.spec)
        log.info("registered as %s for eval %s", response.worker_id, response.eval_id)
        return response.worker_id

    def _serve_once(self, worker_id: str, max_items: int) -> int:
        waited_from = self._clock()
        pulled = self._client.pull(worker_id, max_items, self._wait_s)
        self.stats.wait_s += self._clock() - waited_from
        if not pulled.items:
            return 0
        batch = [decode_item(item) for item in pulled.items]
        infer_from = self._clock()
        chunks = self._policy.infer(batch)
        self.stats.infer_s += self._clock() - infer_from
        self.stats.batches += 1
        if len(chunks) != len(batch):
            raise RuntimeError(
                f"policy returned {len(chunks)} chunks for {len(batch)} observations"
            )
        spec = self._policy.spec
        request = PushRequest(
            lease_id=pulled.lease_id,
            chunks=[
                ActionChunk(
                    episode_id=obs.episode_id,
                    chunk_index=obs.chunk_index,
                    actions=_as_rows(chunk, spec.horizon, spec.action_dim),
                )
                for obs, chunk in zip(batch, chunks, strict=True)
            ],
        )
        pushed = self._client.push(request)
        self.stats.items += pushed.accepted
        self.stats.rejected += len(pushed.rejected)
        for rejected in pushed.rejected:
            log.warning("chunk rejected for %s: %s", rejected.episode_id, rejected.reason)
        return pushed.accepted


def _as_rows(chunk: np.ndarray, horizon: int, action_dim: int) -> list[list[float]]:
    array = np.asarray(chunk, dtype=np.float64)
    if array.shape != (horizon, action_dim):
        raise RuntimeError(f"chunk shape {array.shape} is not ({horizon}, {action_dim})")
    return [[float(value) for value in row] for row in array]
