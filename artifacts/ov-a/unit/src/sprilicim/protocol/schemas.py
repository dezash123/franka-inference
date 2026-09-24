"""Message schemas for the data path.

Every message carries ``protocol_version`` so that HQ can refuse a client from
another era with a clear error instead of a confusing shape mismatch.

Observation payloads use openpi's key names on purpose: a stock openpi policy
consumes them unmodified, and a custom runtime has a documented contract.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

PROTOCOL_VERSION: Final = 1

# Observation keys for the DROID embodiment RoboLab evaluates, in openpi's spelling.
IMAGE_KEYS: tuple[str, ...] = ("exterior_image_1_left", "wrist_image_left")
JOINT_DIM = 7
GRIPPER_DIM = 1

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class _Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_version: Literal[1] = 1


class ActionSpace(str, Enum):
    JOINT_POSITION = "joint_position"


class PolicySpec(_Message):
    """What a policy worker declares at registration; the sim side adapts to it."""

    name: str
    identity_sha256: str
    image_size: tuple[int, int] = Field(description="(height, width) the sim resizes frames to")
    horizon: int = Field(ge=1, le=64, description="actions per chunk")
    action_space: ActionSpace = ActionSpace.JOINT_POSITION
    action_dim: int = Field(default=JOINT_DIM + GRIPPER_DIM, ge=1, le=64)
    gripper_binarize: bool = True
    max_batch: int = Field(ge=1, le=256)
    max_infer_s: float = Field(default=60.0, ge=1.0, le=600.0)

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        if not _IDENTIFIER.match(value):
            raise ValueError(
                "name must be 1-80 characters of [A-Za-z0-9._-], starting alphanumeric"
            )
        return value

    @field_validator("identity_sha256")
    @classmethod
    def _identity(cls, value: str) -> str:
        if not _SHA256.match(value):
            raise ValueError("identity_sha256 must be 64 lowercase hex characters")
        return value

    @field_validator("image_size")
    @classmethod
    def _image_size(cls, value: tuple[int, int]) -> tuple[int, int]:
        height, width = value
        if not (16 <= height <= 4096 and 16 <= width <= 4096):
            raise ValueError("image_size dimensions must be within 16..4096")
        return value


class ObservationItem(_Message):
    """One environment at one moment. Self-contained: a policy needs nothing else to act."""

    episode_id: str
    env_id: int = Field(ge=0)
    step: int = Field(ge=0)
    chunk_index: int = Field(ge=0)
    instruction: str
    seed: int = Field(ge=0)
    images: dict[str, bytes] = Field(description="image key -> PNG bytes")
    joint_position: list[float] = Field(min_length=JOINT_DIM, max_length=JOINT_DIM)
    gripper_position: list[float] = Field(min_length=GRIPPER_DIM, max_length=GRIPPER_DIM)


class ActionChunk(_Message):
    episode_id: str
    chunk_index: int = Field(ge=0)
    actions: list[list[float]] = Field(description="horizon rows of action_dim floats")


class InferRequest(_Message):
    """A sim worker's batch: every environment that needs a new chunk on this tick."""

    batch_id: str
    eval_id: str
    shard_id: str
    items: list[ObservationItem] = Field(min_length=1)


class InferResponse(_Message):
    batch_id: str
    chunks: list[ActionChunk]


class RegisterRequest(_Message):
    spec: PolicySpec


class RegisterResponse(_Message):
    worker_id: str
    eval_id: str


class PullRequest(_Message):
    worker_id: str
    max_items: int = Field(ge=1, le=256)
    wait_s: float = Field(default=30.0, ge=0.0, le=60.0)


class PullResponse(_Message):
    """Zero items means the wait elapsed with nothing pending; ``lease_id`` is then empty."""

    lease_id: str
    lease_s: float
    items: list[ObservationItem]


class PushRequest(_Message):
    lease_id: str
    chunks: list[ActionChunk] = Field(min_length=1)


class RejectedChunk(_Message):
    episode_id: str
    chunk_index: int
    reason: str


class PushResponse(_Message):
    accepted: int
    rejected: list[RejectedChunk]


class HeartbeatRequest(_Message):
    worker_id: str


class HeartbeatResponse(_Message):
    ok: bool


class EpisodeEventKind(str, Enum):
    BEGIN = "begin"
    END = "end"


class EpisodeEvent(_Message):
    kind: EpisodeEventKind
    episode_id: str
    task_id: str
    instruction: str = ""
    seed: int = Field(default=0, ge=0)


class EpisodeEventBatch(_Message):
    events: list[EpisodeEvent] = Field(min_length=1, max_length=1000)
