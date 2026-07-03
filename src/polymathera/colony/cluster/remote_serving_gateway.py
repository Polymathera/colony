"""Partial-OpenAI HTTP shim for a self-hosted colony serving fleet.

Runs ON the fleet and exposes its ``VLLMDeployment`` over HTTP so a remote
:class:`VllmRemoteDeployment` (in a separate cluster) can reach it:

- ``POST /v1/chat/completions`` — OpenAI-compatible chat (the "OpenAI"
  part); maps to ``VLLMDeployment.infer``. Tenant identity travels in the
  ``X-Colony-Id`` / ``X-Tenant-Id`` headers.
- ``POST /load_page`` · ``/evict_page`` · ``/infer_with_suffix`` — colony
  paging passthrough (the "partial", non-OpenAI part); colony types carry
  their own ``syscontext``.
- ``POST /add_lora_adapter`` — hot-add a promoted adapter (the flywheel
  publish target).

``start_agent``/``stop_agent`` are deliberately NOT exposed — the fleet is
headless inference + paging, so those endpoints are REST-locked by
omission. Bearer ``api_key`` gates every route.

The ``VLLMDeployment`` handle is injected (``get_handle``) so the routes
are testable without Ray; the deploy entrypoint wires it to
``get_vllm_deployment``.
"""

from __future__ import annotations

import time
from typing import Any, Callable
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException

from ..distributed.ray_utils.serving.context import (
    ExecutionContext,
    Ring,
    execution_context,
    restore_execution_context,
)
from ..vcm.models import VirtualContextPage
from .config import LoRAAdapterConfig
from .models import InferenceRequest, LLMClientRequirements


def create_serving_gateway(
    get_handle: Callable[[], Any], *, api_key: str = "",
) -> FastAPI:
    """Build the fleet's partial-OpenAI HTTP app dispatching to the
    ``VLLMDeployment`` handle returned by ``get_handle``."""

    app = FastAPI(title="Polymathera remote serving (partial-OpenAI)")

    def _auth(authorization: str | None) -> None:
        if api_key and authorization != f"Bearer {api_key}":
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/load_page")
    async def load_page(body: dict[str, Any], authorization: str | None = Header(default=None)):
        _auth(authorization)
        page = VirtualContextPage.model_validate(body["page"])
        with restore_execution_context(page.syscontext):
            result = await get_handle().load_page(page)
        return {"result": bool(result)}

    @app.post("/evict_page")
    async def evict_page(body: dict[str, Any], authorization: str | None = Header(default=None)):
        _auth(authorization)
        with execution_context(ring=Ring.KERNEL, origin="remote_gateway"):
            result = await get_handle().evict_page(body["page_id"])
        return {"result": bool(result)}

    @app.post("/infer_with_suffix")
    async def infer_with_suffix(body: dict[str, Any], authorization: str | None = Header(default=None)):
        _auth(authorization)
        request = InferenceRequest.model_validate(body["request"])
        with restore_execution_context(request.syscontext):
            resp = await get_handle().infer_with_suffix(
                body["base_page_id"], request, body.get("suffix_tokens"),
            )
        return resp.model_dump(mode="json")

    @app.post("/add_lora_adapter")
    async def add_lora_adapter(body: dict[str, Any], authorization: str | None = Header(default=None)):
        _auth(authorization)
        adapter = LoRAAdapterConfig.model_validate(body)
        with execution_context(ring=Ring.KERNEL, origin="remote_gateway"):
            adapter_id = await get_handle().add_lora_adapter(adapter)
        return {"adapter_id": adapter_id}

    @app.post("/v1/chat/completions")
    async def chat_completions(
        body: dict[str, Any],
        authorization: str | None = Header(default=None),
        x_colony_id: str | None = Header(default=None),
        x_tenant_id: str | None = Header(default=None),
    ):
        _auth(authorization)
        if not (x_colony_id and x_tenant_id):
            raise HTTPException(
                status_code=400,
                detail="X-Colony-Id and X-Tenant-Id headers are required",
            )
        ctx = ExecutionContext(
            ring=Ring.USER, colony_id=x_colony_id, tenant_id=x_tenant_id,
            origin="remote_gateway",
        )
        model = body.get("model") or ""
        prompt = "\n".join(m.get("content", "") for m in body.get("messages", []))
        with restore_execution_context(ctx):
            request = InferenceRequest(
                request_id=str(uuid4()),
                prompt=prompt,
                max_tokens=body.get("max_tokens", 1024),
                temperature=body.get("temperature", 0.7),
                requirements=LLMClientRequirements(lora_adapter_id=model) if model else None,
                syscontext=ctx,
            )
            resp = await get_handle().infer(request)
        return {
            "id": f"chatcmpl-{resp.request_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": resp.generated_text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": resp.tokens_generated,
                "total_tokens": resp.tokens_generated,
            },
        }

    return app


__all__ = ("create_serving_gateway",)
