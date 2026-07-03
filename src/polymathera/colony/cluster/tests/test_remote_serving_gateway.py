"""The fleet partial-OpenAI shim dispatches every route to the
VLLMDeployment handle + maps the OpenAI chat shape + gates on auth."""

from __future__ import annotations

from fastapi.testclient import TestClient

from polymathera.colony.cluster.models import InferenceRequest, InferenceResponse
from polymathera.colony.cluster.remote_serving_gateway import create_serving_gateway
from polymathera.colony.distributed.ray_utils.serving.context import ExecutionContext, Ring
from polymathera.colony.vcm.models import VirtualContextPage


class _FakeHandle:
    def __init__(self):
        self.calls: list = []

    async def load_page(self, page):
        self.calls.append(("load_page", page))
        return True

    async def evict_page(self, page_id):
        self.calls.append(("evict_page", page_id))
        return True

    async def infer_with_suffix(self, base_page_id, request, suffix_tokens):
        self.calls.append(("infer_with_suffix", base_page_id, suffix_tokens))
        return InferenceResponse(
            request_id=request.request_id, generated_text="suf",
            tokens_generated=3, latency_ms=1.0,
        )

    async def infer(self, request):
        self.calls.append(("infer", request))
        return InferenceResponse(
            request_id=request.request_id, generated_text="chat",
            tokens_generated=5, latency_ms=1.0,
        )

    async def add_lora_adapter(self, adapter):
        self.calls.append(("add_lora_adapter", adapter))
        return adapter.adapter_id


def _ctx() -> ExecutionContext:
    return ExecutionContext(ring=Ring.USER, colony_id="c1", tenant_id="t1", origin="test")


def _client(api_key: str = ""):
    handle = _FakeHandle()
    return TestClient(create_serving_gateway(lambda: handle, api_key=api_key)), handle


def test_load_page_dispatches() -> None:
    client, handle = _client()
    page = VirtualContextPage(page_id="p1", tokens=[1], size=1, syscontext=_ctx())
    r = client.post("/load_page", json={"page": page.model_dump(mode="json")})
    assert r.status_code == 200 and r.json() == {"result": True}
    assert handle.calls[0][0] == "load_page" and handle.calls[0][1].page_id == "p1"


def test_infer_with_suffix_dispatches() -> None:
    client, handle = _client()
    req = InferenceRequest(request_id="r1", prompt="hi", syscontext=_ctx())
    r = client.post("/infer_with_suffix", json={
        "base_page_id": "b1", "request": req.model_dump(mode="json"), "suffix_tokens": [7],
    })
    assert r.status_code == 200 and r.json()["generated_text"] == "suf"
    assert handle.calls[0] == ("infer_with_suffix", "b1", [7])


def test_add_lora_adapter_dispatches() -> None:
    client, handle = _client()
    r = client.post("/add_lora_adapter", json={
        "adapter_id": "agent-x", "adapter_name": "agent-x",
        "base_model_name": "base", "rank": 16, "s3_bucket": "models",
    })
    assert r.status_code == 200 and r.json() == {"adapter_id": "agent-x"}
    assert handle.calls[0][1].adapter_id == "agent-x"


def test_chat_completions_openai_shape_and_adapter_selection() -> None:
    client, handle = _client()
    r = client.post(
        "/v1/chat/completions",
        json={"model": "agent-x", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
        headers={"X-Colony-Id": "c1", "X-Tenant-Id": "t1"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "chat.completion" and data["model"] == "agent-x"
    assert data["choices"][0]["message"]["content"] == "chat"
    infer_req = next(c for c in handle.calls if c[0] == "infer")[1]
    assert infer_req.requirements.lora_adapter_id == "agent-x"  # LoRA selected by model


def test_chat_requires_tenant_headers() -> None:
    client, _ = _client()
    assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 400


def test_bearer_auth_enforced() -> None:
    client, _ = _client(api_key="secret")
    assert client.post("/evict_page", json={"page_id": "p"}).status_code == 401
    ok = client.post("/evict_page", json={"page_id": "p"}, headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200


def test_agent_lifecycle_rest_locked_by_omission() -> None:
    # The fleet is headless inference + paging: no start_agent/stop_agent
    # route exists, so those endpoints are REST-locked by omission.
    app = create_serving_gateway(lambda: _FakeHandle())
    paths = {getattr(r, "path", "") for r in app.routes}
    assert "/load_page" in paths and "/v1/chat/completions" in paths
    assert not any("agent" in p for p in paths)
