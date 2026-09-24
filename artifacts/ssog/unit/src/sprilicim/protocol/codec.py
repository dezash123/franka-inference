"""Lossless encoding for the data path.

Messages travel as MessagePack of their plain-data form; images inside them are
PNG, which decodes to the identical uint8 array that was encoded. Nothing on the
wire depends on NumPy, so any language can implement a policy worker.
"""

from __future__ import annotations

import io
from typing import TypeVar

import msgpack
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel, ValidationError

CONTENT_TYPE = "application/msgpack"

_M = TypeVar("_M", bound=BaseModel)


class CodecError(ValueError):
    """The bytes were not a valid message of the requested type."""


def encode(message: BaseModel) -> bytes:
    payload = message.model_dump(mode="python")
    packed: bytes = msgpack.packb(payload, use_bin_type=True)
    return packed


def decode(message_type: type[_M], data: bytes) -> _M:
    try:
        payload = msgpack.unpackb(data, raw=False)
    except Exception as exc:  # msgpack raises several unrelated types
        raise CodecError(f"not valid MessagePack: {exc}") from exc
    if not isinstance(payload, dict):
        raise CodecError("message must be a map")
    try:
        return message_type.model_validate(payload)
    except ValidationError as exc:
        raise CodecError(str(exc)) from exc


def png_encode(image: NDArray[np.uint8]) -> bytes:
    """Encode an HWC uint8 RGB array losslessly."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise CodecError("image must be an HWC uint8 RGB array")
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG", compress_level=6)
    return buffer.getvalue()


def png_decode(data: bytes) -> NDArray[np.uint8]:
    try:
        with Image.open(io.BytesIO(data)) as frame:
            if frame.mode != "RGB":
                raise CodecError(f"image must be RGB, got {frame.mode}")
            return np.asarray(frame, dtype=np.uint8).copy()
    except CodecError:
        raise
    except Exception as exc:
        raise CodecError(f"not a decodable PNG: {exc}") from exc
