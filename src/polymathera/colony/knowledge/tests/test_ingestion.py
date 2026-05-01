"""End-to-end tests for the ``Ingestor``."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from polymathera.colony.knowledge import (
    CorpusTier,
    DeterministicClaimExtractor,
    InMemoryEmbedder,
    InMemoryGraphStore,
    InMemoryVectorStore,
    Ingestor,
    IngestionStatus,
    KnowledgeFormat,
)


pytestmark = pytest.mark.asyncio


@pytest.fixture
def ingestor() -> Ingestor:
    return Ingestor(
        embedder=InMemoryEmbedder(),
        vector_store=InMemoryVectorStore(),
        graph_store=InMemoryGraphStore(),
        extractors=[DeterministicClaimExtractor()],
        review_sample_rate=0.0,
    )


async def test_ingest_markdown_text(ingestor: Ingestor) -> None:
    text = (
        "# Title\n\n"
        "Intro paragraph.\n\n"
        "## Confinement\n\n"
        "JET requires deuterium-tritium fuel.\n"
    )
    record = await ingestor.ingest_text(
        text, source_uri="book:wesson:ch1",
        fmt=KnowledgeFormat.MARKDOWN,
        tier=CorpusTier.TIER_1_FOUNDATIONS,
    )
    assert record.status is IngestionStatus.COMPLETED
    assert record.chunks_produced > 0
    assert record.claims_extracted >= 1
    assert record.tier is CorpusTier.TIER_1_FOUNDATIONS
    assert record.error is None


async def test_ingest_jsonl(tmp_path: Path, ingestor: Ingestor) -> None:
    p = tmp_path / "papers.jsonl"
    p.write_text(
        json.dumps({"title": "Paper A", "doi": "10.1/x"}) + "\n"
        + json.dumps({"title": "Paper B", "doi": "10.2/y"}) + "\n",
        encoding="utf-8",
    )
    record = await ingestor.ingest_file(p)
    assert record.status is IngestionStatus.COMPLETED
    assert record.chunks_produced >= 1
    assert record.detected_format is KnowledgeFormat.JSONL


async def test_ingest_unknown_format_fails(ingestor: Ingestor) -> None:
    record = await ingestor.ingest_text(
        "x", source_uri="thing:1", fmt=KnowledgeFormat.UNKNOWN,
    )
    assert record.status is IngestionStatus.FAILED
    assert record.error is not None


async def test_review_queue_called_when_sampled(tmp_path: Path) -> None:
    queue: list = []

    async def reviewer(record, chunks):
        queue.append((record, chunks))

    ing = Ingestor(
        embedder=InMemoryEmbedder(),
        vector_store=InMemoryVectorStore(),
        review_queue=reviewer,
        review_sample_rate=1.0,
    )
    record = await ing.ingest_text(
        "Body text.", source_uri="x:y", fmt=KnowledgeFormat.PLAIN_TEXT,
    )
    assert record.review_required is True
    assert len(queue) == 1


async def test_pipeline_state_visible_via_record_attributes(
    ingestor: Ingestor,
) -> None:
    text = "para one\n\npara two"
    record = await ingestor.ingest_text(
        text, source_uri="x:y", fmt=KnowledgeFormat.PLAIN_TEXT,
    )
    assert record.detected_format is KnowledgeFormat.PLAIN_TEXT
    assert record.started_at is not None
    assert record.finished_at is not None
    assert record.finished_at >= record.started_at


async def test_ingestor_persists_chunks_to_vector_store() -> None:
    vstore = InMemoryVectorStore()
    ing = Ingestor(
        embedder=InMemoryEmbedder(),
        vector_store=vstore,
        review_sample_rate=0.0,
    )
    await ing.ingest_text(
        "First paragraph.\n\nSecond paragraph.",
        source_uri="x:y", fmt=KnowledgeFormat.PLAIN_TEXT,
    )
    assert await vstore.count() >= 1
