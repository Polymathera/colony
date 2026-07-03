"""VllmRemoteDeployment forwards the colony paging surface to the fleet
(§5.8 Option A: thin local forwarder; the fleet owns paging + routing)."""

from __future__ import annotations

import asyncio

from polymathera.colony.cluster.models import InferenceRequest, InferenceResponse
from polymathera.colony.cluster.remote_config import RemoteLLMDeploymentConfig
from polymathera.colony.cluster.vllm_remote_deployment import VllmRemoteDeployment
from polymathera.colony.distributed.ray_utils.serving.context import (
    ExecutionContext,
    Ring,
    execution_context,
)
from polymathera.colony.vcm.models import VirtualContextPage


class _FakeResp:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


class _FakeHttp:
    def __init__(self, data):
        self._data = data
        self.calls: list = []

    async def post(self, path, json):
        self.calls.append((path, json))
        return _FakeResp(self._data)


def _dep(data) -> VllmRemoteDeployment:
    dep = VllmRemoteDeployment(RemoteLLMDeploymentConfig(
        model_name="agent-x", provider="vllm", base_url="http://fleet:8000/v1",
    ))
    dep._http = _FakeHttp(data)
    return dep


def _ctx() -> ExecutionContext:
    # A request/page context is tenant-scoped (USER ring).
    return ExecutionContext(ring=Ring.USER, colony_id="c1", tenant_id="t1", origin="test")


def _run(coro_factory):
    """Run a deployment coroutine under a tenant execution context."""
    async def _main():
        return await coro_factory()

    with execution_context(ring=Ring.USER, colony_id="c1", tenant_id="t1", origin="test"):
        return asyncio.run(_main())


def test_evict_page_forwards() -> None:
    dep = _dep(True)
    assert _run(lambda: dep.evict_page("page-1")) is True
    assert dep._http.calls == [("/evict_page", {"page_id": "page-1"})]


def test_load_page_forwards_the_page() -> None:
    dep = _dep(True)
    page = VirtualContextPage(page_id="p1", tokens=[1, 2], size=2, syscontext=_ctx())
    assert _run(lambda: dep.load_page(page)) is True
    path, body = dep._http.calls[0]
    assert path == "/load_page" and body["page"]["page_id"] == "p1"


def test_infer_with_suffix_forwards_and_parses() -> None:
    resp = InferenceResponse(
        request_id="r1", generated_text="ok", tokens_generated=2, latency_ms=1.0,
    )
    dep = _dep(resp.model_dump(mode="json"))
    req = InferenceRequest(request_id="r1", prompt="hi", syscontext=_ctx())
    out = _run(lambda: dep.infer_with_suffix("base-1", req, suffix_tokens=[7]))
    assert isinstance(out, InferenceResponse) and out.generated_text == "ok"
    path, body = dep._http.calls[0]
    assert path == "/infer_with_suffix"
    assert body["base_page_id"] == "base-1" and body["suffix_tokens"] == [7]
    assert body["request"]["request_id"] == "r1"
