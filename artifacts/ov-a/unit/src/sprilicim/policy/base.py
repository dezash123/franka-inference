"""The one interface a runtime implements to be evaluated."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sprilicim.protocol.schemas import PolicySpec


@dataclass(frozen=True)
class Observation:
    """One environment at one moment, images already decoded to uint8 arrays."""

    episode_id: str
    env_id: int
    step: int
    chunk_index: int
    instruction: str
    seed: int
    images: dict[str, NDArray[np.uint8]]
    joint_position: NDArray[np.float64]
    gripper_position: NDArray[np.float64]


class Policy(ABC):
    """Implement ``spec`` and ``infer``; the worker handles everything else.

    ``infer`` receives a batch and returns one chunk per observation, in order,
    each shaped (spec.horizon, spec.action_dim). Nothing carries over between
    calls: every chunk depends only on its observation.
    """

    @property
    @abstractmethod
    def spec(self) -> PolicySpec: ...

    @abstractmethod
    def infer(self, batch: list[Observation]) -> list[NDArray[np.float64]]: ...
