"""Integration tests against a real Qdrant.

Skipped unless ``POLYMATHERA_QDRANT_URL`` is set (e.g.,
``http://localhost:6333``). Run a Qdrant locally with::

    docker compose -f src/polymathera/colony/cli/deploy/docker/docker-compose.yml \\
        up -d qdrant
    POLYMATHERA_QDRANT_URL=http://localhost:6333 \\
        pytest src/polymathera/colony/knowledge/tests/integration/

The collection name is namespaced per test run to avoid cross-test
pollution; each test cleans up after itself.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from polymathera.colony.knowledge import (
    Chunk,
    CitationSpan,
    CorpusTier,
    EmbeddedChunk,
    QdrantVectorStore,
    RetrievalQuery,
)


pytestmark = pytest.mark.asyncio


def _embedded(
    chunk_id: str,
    *,
    vector: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0),
    source: str = "src:integration",
    data_type: str = "paper_section",
    tier: CorpusTier = CorpusTier.UNTIERED,
) -> EmbeddedChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        text=f"text-{chunk_id}",
        token_count=2,
        section_path="1",
        citation=CitationSpan(source_uri=source, section_path="1"),
        data_type=data_type,
        source=source,
        tier=tier,
    )
    return EmbeddedChunk(
        chunk=chunk, vector=vector, embedder="integration:test",
    )


@pytest.fixture
async def store(qdrant_url_or_skip):
    collection = f"polymathera_test_{uuid.uuid4().hex[:8]}"
    s = await QdrantVectorStore.connect(
        url=qdrant_url_or_skip,
        collection=collection,
        embedder_id="integration:test",
        dimensions=4,
    )
    try:
        yield s
    finally:
        try:
            await s._client.delete_collection(collection_name=collection)
        except Exception:
            pass


async def test_real_upsert_and_search(store: QdrantVectorStore) -> None:
    await store.upsert(
        [
            _embedded("a", vector=(1.0, 0.0, 0.0, 0.0)),
            _embedded("b", vector=(0.7, 0.7, 0.0, 0.0)),
            _embedded("c", vector=(0.0, 1.0, 0.0, 0.0)),
        ],
    )
    assert await store.count() == 3
    hits = await store.search(
        query_vector=(0.95, 0.05, 0.0, 0.0),
        query=RetrievalQuery(text="x", max_results=2),
    )
    ids = [h.chunk.chunk_id for h in hits]
    assert ids[0] == "a"
    assert "b" in ids


async def test_real_filter_by_data_type(store: QdrantVectorStore) -> None:
    await store.upsert(
        [
            _embedded("p", data_type="paper_section"),
            _embedded("s", data_type="standard_clause"),
        ],
    )
    hits = await store.search(
        query_vector=(1.0, 0.0, 0.0, 0.0),
        query=RetrievalQuery(text="x", data_types=("standard_clause",)),
    )
    assert [h.chunk.chunk_id for h in hits] == ["s"]


async def test_real_get_round_trip(store: QdrantVectorStore) -> None:
    item = _embedded("a", vector=(1.0, 0.0, 0.0, 0.0))
    await store.upsert([item])
    got = await store.get("a")
    assert got is not None
    assert got.chunk.chunk_id == "a"
    assert tuple(round(v, 6) for v in got.vector) == (1.0, 0.0, 0.0, 0.0)


async def test_real_delete_by_source(store: QdrantVectorStore) -> None:
    await store.upsert(
        [
            _embedded("a", source="arxiv:1"),
            _embedded("b", source="arxiv:2"),
            _embedded("c", source="git:repo:main:1"),
        ],
    )
    n = await store.delete_by_source("arxiv:")
    assert n == 2
    assert await store.count() == 1
