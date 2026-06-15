"""Unit tests for :class:`KnowledgeRetrievalCapability`.

The retrieval *adapters* are tested in detail elsewhere
(``knowledge/tests/test_retrieval.py``); these tests only verify the
capability surface — adapter selection, query translation, default
mode, action result shape — using the in-memory embedder + vector
store fixtures that the rest of the knowledge suite uses.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from polymathera.colony.agents.patterns.capabilities.knowledge_retrieval import (
    KnowledgeRetrievalCapability,
)
from polymathera.colony.agents.scopes import BlackboardScope
from polymathera.colony.distributed.ray_utils.serving.context import (
    Ring,
    execution_context,
)
from polymathera.colony.knowledge.embedder import InMemoryEmbedder
from polymathera.colony.knowledge.models import (
    Chunk,
    CitationSpan,
    CorpusTier,
    EmbeddedChunk,
)
from polymathera.colony.knowledge.retrieval import RetrievalDeps
from polymathera.colony.knowledge.stores.vector import InMemoryVectorStore


def _run(coro):
    return asyncio.run(coro)


def _with_user_ctx():
    return execution_context(
        ring=Ring.USER, tenant_id="t", colony_id="c", session_id="s",
    )


async def _populate_store(store: InMemoryVectorStore, embedder: InMemoryEmbedder) -> None:
    """Seed the store with three chunks under different sources."""
    payloads = [
        ("authentication tokens are signed JWTs", "auth", "auth_intro"),
        ("payment processing uses Stripe webhooks", "payments", "stripe_flow"),
        ("user sessions expire after 24 hours", "auth", "session_ttl"),
    ]
    items = []
    for text, source, chunk_id in payloads:
        vec = (await embedder.embed([text]))[0]
        items.append(
            EmbeddedChunk(
                chunk=Chunk(
                    chunk_id=chunk_id,
                    text=text,
                    token_count=len(text.split()),
                    citation=CitationSpan(
                        source_uri=source, char_start=0, char_end=len(text),
                    ),
                    data_type="paper_section",
                    source=source,
                    tier=CorpusTier.UNTIERED,
                ),
                vector=tuple(vec),
                embedder=embedder.embedder_id,
            )
        )
    await store.upsert(items)


@pytest.fixture
def cap() -> KnowledgeRetrievalCapability:
    embedder = InMemoryEmbedder()
    store = InMemoryVectorStore()
    _run(_populate_store(store, embedder))
    deps = RetrievalDeps(embedder=embedder, vector_store=store)
    agent = MagicMock(); agent.agent_id = "agent-A"
    with _with_user_ctx():
        return KnowledgeRetrievalCapability(
            agent=agent,
            scope=BlackboardScope.SESSION,
            deps=deps,
        )


def test_list_modes_returns_default_and_all_options(cap) -> None:
    out = _run(cap.list_modes())
    assert out["default"] == "scoped"
    assert set(out["modes"]) == {
        "scoped", "grounded", "graph", "budgeted", "standards",
    }


def test_search_default_scoped_requires_source_prefix(cap) -> None:
    # ScopedRetrievalCapability declines a query with no source_prefix —
    # the capability must surface that as a normal RetrievalResult
    # (zero hits, reason in extra), not raise.
    result = _run(cap.search_knowledge(query="auth"))
    assert result["mode"] == "scoped"
    assert result["hits"] == []
    assert "source_prefix" in result["extra"]["reason"]


def test_scoped_search_with_source_prefix_returns_hits(cap) -> None:
    result = _run(
        cap.search_knowledge(query="auth", source_prefix="auth", top_k=2),
    )
    assert result["mode"] == "scoped"
    assert len(result["hits"]) > 0
    sources = {hit["chunk"]["source"] for hit in result["hits"]}
    assert sources == {"auth"}


def test_unknown_mode_raises_value_error(cap) -> None:
    with pytest.raises(ValueError, match="unknown retrieval retriever"):
        _run(cap.search_knowledge(query="x", mode="bogus"))


def test_retriever_cache_reuses_instance(cap) -> None:
    a1 = cap._get_retriever("scoped")
    a2 = cap._get_retriever("scoped")
    assert a1 is a2
