"""The policy worker's view of HQ, with an HTTP implementation and a protocol for fakes."""

from __future__ import annotations

from typing import Protocol

import httpx

from sprilicim.protocol.codec import CONTENT_TYPE, decode, encode
from sprilicim.protocol.schemas import (
    PolicySpec,
    PullRequest,
    PullResponse,
    PushRequest,
    PushResponse,
    RegisterRequest,
    RegisterResponse,
)


class HqPolicyClient(Protocol):
    def register(self, spec: PolicySpec) -> RegisterResponse: ...

    def pull(self, worker_id: str, max_items: int, wait_s: float) -> PullResponse: ...

    def push(self, request: PushRequest) -> PushResponse: ...


class HttpHqPolicyClient:
    def __init__(self, hq_url: str, token: str, verify: bool | str = True) -> None:
        self._api = httpx.Client(
            base_url=hq_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=httpx.Timeout(30.0, read=90.0),
            verify=verify,
        )

    def register(self, spec: PolicySpec) -> RegisterResponse:
        response = self._api.post(
            "/v1/policy/register", json=RegisterRequest(spec=spec).model_dump(mode="json")
        )
        response.raise_for_status()
        return RegisterResponse.model_validate(response.json())

    def pull(self, worker_id: str, max_items: int, wait_s: float) -> PullResponse:
        body = PullRequest(worker_id=worker_id, max_items=max_items, wait_s=wait_s)
        response = self._api.post(
            "/v1/policy/pull", content=encode(body), headers={"content-type": CONTENT_TYPE}
        )
        response.raise_for_status()
        return decode(PullResponse, response.content)

    def push(self, request: PushRequest) -> PushResponse:
        response = self._api.post(
            "/v1/policy/push", content=encode(request), headers={"content-type": CONTENT_TYPE}
        )
        response.raise_for_status()
        return decode(PushResponse, response.content)
