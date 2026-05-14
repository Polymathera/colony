"""Tests for ``KnowledgeCuratorCapability``."""

from __future__ import annotations

import pytest

from polymathera.colony.agents.roles import KnowledgeCuratorCapability
from polymathera.colony.knowledge import (
    DeterministicClaimExtractor,
    InMemoryEmbedder,
    InMemoryGraphStore,
    InMemoryVectorStore,
    Ingestor,
    KnowledgeFormat,
)
from polymathera.colony.vcm.page_events import PageChangeEvent


pytestmark = pytest.mark.asyncio


@pytest.fixture
def vstore() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.fixture
def gstore() -> InMemoryGraphStore:
    return InMemoryGraphStore()


@pytest.fixture
def ingestor(vstore: InMemoryVectorStore, gstore: InMemoryGraphStore) -> Ingestor:
    return Ingestor(
        embedder=InMemoryEmbedder(),
        vector_store=vstore,
        graph_store=gstore,
        extractors=[DeterministicClaimExtractor()],
        review_sample_rate=0.0,
    )


@pytest.fixture
def curator(ingestor: Ingestor) -> KnowledgeCuratorCapability:
    return KnowledgeCuratorCapability(
        agent=None, scope_id="curator", ingestor=ingestor,
    )


async def test_ingest_text_succeeds(curator: KnowledgeCuratorCapability) -> None:
    record = await curator.ingest_text(
        "Wesson is a tokamak textbook author.",
        source_uri="book:wesson",
        fmt=KnowledgeFormat.PLAIN_TEXT,
    )
    assert record.chunks_produced > 0


async def test_ingest_emits_page_event() -> None:
    captured: list[PageChangeEvent] = []

    async def emitter(event):
        captured.append(event)

    ing = Ingestor(
        embedder=InMemoryEmbedder(),
        vector_store=InMemoryVectorStore(),
        review_sample_rate=0.0,
    )
    cap = KnowledgeCuratorCapability(
        agent=None, scope_id="curator",
        ingestor=ing, page_event_emitter=emitter,
    )
    await cap.ingest_text("hello world", source_uri="src:1", fmt=KnowledgeFormat.PLAIN_TEXT)
    assert captured and captured[0].source == "src:1"
    assert captured[0].data_type == "ingested_source"


async def test_review_queue_unresolved_listing(
    curator: KnowledgeCuratorCapability,
) -> None:
    # Force review by raising sample rate to 1.
    curator._ingestor._review_rate = 1.0
    await curator.ingest_text("text", source_uri="src:x", fmt=KnowledgeFormat.PLAIN_TEXT)
    items = await curator.list_review_queue()
    assert len(items) == 1
    assert items[0].resolved is False


async def test_resolve_review_item(curator: KnowledgeCuratorCapability) -> None:
    curator._ingestor._review_rate = 1.0
    record = await curator.ingest_text(
        "text", source_uri="src:y", fmt=KnowledgeFormat.PLAIN_TEXT,
    )
    item = await curator.resolve_review_item(
        record.record_id, resolution="approved", resolved_by="alice",
    )
    assert item.resolved is True
    assert item.resolution == "approved"


async def test_resolve_unknown_item_raises(
    curator: KnowledgeCuratorCapability,
) -> None:
    with pytest.raises(KeyError):
        await curator.resolve_review_item("nope", "approved")
