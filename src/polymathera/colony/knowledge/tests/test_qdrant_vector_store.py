"""Unit tests for ``QdrantVectorStore`` against a mocked async client.

Each test substitutes a fake ``AsyncQdrantClient`` (a small in-memory
stand-in matching the real client's coroutine signatures) for the
real one. Integration tests against a running Qdrant instance live in
``tests/integration/`` and are skipped when the ``POLYMATHERA_QDRANT_URL``
env var is unset.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pytest

from polymathera.colony.knowledge import (
    Chunk,
    CitationSpan,
    CorpusTier,
    EmbeddedChunk,
    QdrantVectorStore,
    RetrievalQuery,
    VectorStoreError,
)


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fake AsyncQdrantClient
# ---------------------------------------------------------------------------


class _FakePoint:
    def __init__(self, id_: str, vector: list[float], payload: dict[str, Any]) -> None:
        self.id = id_
        self.vector = vector
        self.payload = payload


class _FakeSearchHit:
    def __init__(self, id_: str, score: float, payload: dict[str, Any]) -> None:
        self.id = id_
        self.score = score
        self.payload = payload


class FakeAsyncQdrantClient:
    """Minimal stand-in for ``qdrant_client.AsyncQdrantClient``.

    Implements just enough of the surface ``QdrantVectorStore`` uses:

    - ``get_collection`` (raises ``ValueError`` if missing).
    - ``create_collection``.
    - ``upsert`` / ``delete`` / ``retrieve`` / ``search`` / ``count`` /
      ``scroll``.

    Stores points in-memory; cosine similarity is computed in pure
    Python so unit tests don't require a running Qdrant.
    """

    def __init__(self) -> None:
        self.collections: dict[str, dict[str, _FakePoint]] = {}
        self.collection_specs: dict[str, dict[str, Any]] = {}
        self.upsert_calls = 0
        self.delete_calls = 0

    async def get_collection(self, *, collection_name: str) -> Any:
        if collection_name not in self.collections:
            raise ValueError(f"missing collection {collection_name!r}")

        class _Info:
            pass

        return _Info()

    async def create_collection(
        self, *, collection_name: str, vectors_config: Any,
    ) -> None:
        if collection_name in self.collections:
            return
        self.collections[collection_name] = {}
        self.collection_specs[collection_name] = {"vectors_config": vectors_config}

    async def upsert(
        self, *, collection_name: str, points: Sequence[Any], wait: bool = True,
    ) -> None:
        self.upsert_calls += 1
        bucket = self.collections.setdefault(collection_name, {})
        for p in points:
            bucket[str(p.id)] = _FakePoint(
                id_=str(p.id),
                vector=list(p.vector),
                payload=dict(p.payload or {}),
            )

    async def delete(
        self, *, collection_name: str, points_selector: Any, wait: bool = True,
    ) -> None:
        self.delete_calls += 1
        bucket = self.collections.get(collection_name) or {}
        for pid in list(points_selector):
            bucket.pop(str(pid), None)

    async def retrieve(
        self,
        *,
        collection_name: str,
        ids: Sequence[str],
        with_payload: bool = True,
        with_vectors: bool = True,
    ) -> list[_FakePoint]:
        bucket = self.collections.get(collection_name) or {}
        return [bucket[i] for i in ids if i in bucket]

    async def query_points(
        self,
        *,
        collection_name: str,
        query: Sequence[float],
        query_filter: Any | None = None,
        limit: int = 10,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> Any:
        bucket = self.collections.get(collection_name) or {}
        candidates = [
            p for p in bucket.values()
            if _filter_matches(query_filter, p.payload)
        ]
        scored = sorted(
            ((_cosine(query, p.vector), p) for p in candidates),
            key=lambda t: -t[0],
        )[:limit]

        class _Resp:
            pass

        resp = _Resp()
        resp.points = [_FakeSearchHit(p.id, s, p.payload) for s, p in scored]
        return resp

    async def count(
        self, *, collection_name: str, exact: bool = True,
    ) -> Any:
        class _Result:
            pass

        result = _Result()
        result.count = len(self.collections.get(collection_name) or {})
        return result

    async def scroll(
        self,
        *,
        collection_name: str,
        limit: int = 256,
        offset: Any = None,
        with_payload: bool = True,
        with_vectors: bool = False,
        scroll_filter: Any | None = None,
    ) -> tuple[list[_FakePoint], Any]:
        bucket = self.collections.get(collection_name) or {}
        items = [
            p for p in bucket.values()
            if _filter_matches(scroll_filter, p.payload)
        ]
        start = int(offset or 0)
        page = items[start : start + limit]
        next_offset = start + limit if start + limit < len(items) else None
        return page, next_offset


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    import math

    if not a or not b or len(a) != len(b):
        return 0.0
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return num / (na * nb)


def _filter_matches(query_filter: Any, payload: Mapping[str, Any]) -> bool:
    if query_filter is None:
        return True
    must = list(getattr(query_filter, "must", []) or [])
    for cond in must:
        key = getattr(cond, "key", None)
        match = getattr(cond, "match", None)
        if key is None or match is None:
            continue
        any_values = getattr(match, "any", None)
        if any_values is not None:
            if payload.get(key) not in any_values:
                return False
            continue
        # MatchValue (single-value equality) — used by
        # QdrantVectorStore.list_chunks_for_source.
        single_value = getattr(match, "value", None)
        if single_value is not None:
            if payload.get(key) != single_value:
                return False
            continue
        # Other match shapes aren't used by QdrantVectorStore.
    return True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_chunk(
    chunk_id: str,
    *,
    text: str = "x",
    source: str = "src:a",
    data_type: str = "paper_section",
    tier: CorpusTier = CorpusTier.UNTIERED,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        token_count=max(1, len(text.split())),
        section_path="1",
        citation=CitationSpan(source_uri=source, section_path="1"),
        data_type=data_type,
        source=source,
        tier=tier,
    )


def _embedded(
    chunk_id: str,
    *,
    vector: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0),
    **kwargs: Any,
) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk=_make_chunk(chunk_id, **kwargs),
        vector=vector,
        embedder="test:fixture",
    )


@pytest.fixture
async def store() -> QdrantVectorStore:
    fake = FakeAsyncQdrantClient()
    return QdrantVectorStore(
        client=fake,
        collection="test",
        embedder_id="test:fixture",
        dimensions=4,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_upsert_creates_collection_lazily(
    store: QdrantVectorStore,
) -> None:
    n = await store.upsert([_embedded("a")])
    assert n == 1
    assert await store.count() == 1


async def test_upsert_dimension_mismatch_raises(
    store: QdrantVectorStore,
) -> None:
    bad = _embedded("a", vector=(1.0, 0.0, 0.0))  # 3-d, store is 4-d
    with pytest.raises(VectorStoreError):
        await store.upsert([bad])


async def test_search_orders_by_cosine(
    store: QdrantVectorStore,
) -> None:
    items = [
        _embedded("x", vector=(1.0, 0.0, 0.0, 0.0)),
        _embedded("y", vector=(0.7, 0.7, 0.0, 0.0)),
        _embedded("z", vector=(0.0, 1.0, 0.0, 0.0)),
    ]
    await store.upsert(items)
    hits = await store.search(
        query_vector=(0.95, 0.05, 0.0, 0.0),
        query=RetrievalQuery(text="x", max_results=2),
    )
    ids = [h.chunk.chunk_id for h in hits]
    assert ids == ["x", "y"]


async def test_search_filters_by_data_type(
    store: QdrantVectorStore,
) -> None:
    await store.upsert(
        [
            _embedded("a", data_type="paper_section"),
            _embedded("b", data_type="standard_clause"),
        ],
    )
    hits = await store.search(
        query_vector=(1.0, 0.0, 0.0, 0.0),
        query=RetrievalQuery(text="x", data_types=("standard_clause",)),
    )
    assert [h.chunk.chunk_id for h in hits] == ["b"]


async def test_search_filters_by_tier(
    store: QdrantVectorStore,
) -> None:
    await store.upsert(
        [
            _embedded("a", tier=CorpusTier.TIER_1_FOUNDATIONS),
            _embedded("b", tier=CorpusTier.TIER_3_RESEARCH),
        ],
    )
    hits = await store.search(
        query_vector=(1.0, 0.0, 0.0, 0.0),
        query=RetrievalQuery(text="x", tiers=(CorpusTier.TIER_1_FOUNDATIONS,)),
    )
    assert [h.chunk.chunk_id for h in hits] == ["a"]


async def test_search_filters_by_source_prefix_post_hoc(
    store: QdrantVectorStore,
) -> None:
    await store.upsert(
        [
            _embedded("a", source="git:repo:main:1"),
            _embedded("b", source="arxiv:2410.12345"),
        ],
    )
    hits = await store.search(
        query_vector=(1.0, 0.0, 0.0, 0.0),
        query=RetrievalQuery(text="x", source_prefix="arxiv:"),
    )
    assert [h.chunk.chunk_id for h in hits] == ["b"]


async def test_search_drops_uncited_when_required(
    store: QdrantVectorStore,
) -> None:
    cited = _embedded("a", source="src:a")
    uncited = EmbeddedChunk(
        chunk=Chunk(
            chunk_id="b",
            text="x",
            token_count=1,
            section_path="1",
            citation=CitationSpan(source_uri="", section_path="1"),
            data_type="paper_section",
            source="src:a",
        ),
        vector=(1.0, 0.0, 0.0, 0.0),
        embedder="test:fixture",
    )
    await store.upsert([cited, uncited])
    hits = await store.search(
        query_vector=(1.0, 0.0, 0.0, 0.0),
        query=RetrievalQuery(text="x", require_citations=True),
    )
    assert [h.chunk.chunk_id for h in hits] == ["a"]


async def test_get_returns_embedded_round_trip(
    store: QdrantVectorStore,
) -> None:
    item = _embedded("a", vector=(1.0, 0.0, 0.0, 0.0))
    await store.upsert([item])
    got = await store.get("a")
    assert got is not None
    assert got.chunk.chunk_id == "a"
    assert got.vector == (1.0, 0.0, 0.0, 0.0)


async def test_get_missing_returns_none(
    store: QdrantVectorStore,
) -> None:
    got = await store.get("ghost")
    assert got is None


async def test_delete_by_chunk_ids(
    store: QdrantVectorStore,
) -> None:
    await store.upsert([_embedded("a"), _embedded("b")])
    n = await store.delete_by_chunk_ids(["a"])
    assert n == 1
    assert await store.count() == 1


async def test_delete_by_chunk_ids_unknown_returns_zero(
    store: QdrantVectorStore,
) -> None:
    n = await store.delete_by_chunk_ids(["ghost"])
    assert n == 0


async def test_delete_by_source_walks_collection(
    store: QdrantVectorStore,
) -> None:
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


# ---------------------------------------------------------------------------
# Q-S0a — list_chunks_for_source / update_tier_for_source
# ---------------------------------------------------------------------------


async def test_list_chunks_for_source_exact_match(
    store: QdrantVectorStore,
) -> None:
    await store.upsert(
        [
            _embedded("a", source="arxiv:2410.12345"),
            _embedded("b", source="arxiv:2410.12345"),
            _embedded("c", source="arxiv:2410.12345.bak"),  # prefix sibling
            _embedded("d", source="git:repo:main:1"),
        ],
    )
    chunks = await store.list_chunks_for_source("arxiv:2410.12345")
    ids = sorted(c.chunk.chunk_id for c in chunks)
    assert ids == ["a", "b"]


async def test_list_chunks_for_source_unknown_returns_empty(
    store: QdrantVectorStore,
) -> None:
    chunks = await store.list_chunks_for_source("ghost:1")
    assert chunks == ()


async def test_update_tier_for_source_rewrites_existing(
    store: QdrantVectorStore,
) -> None:
    await store.upsert(
        [
            _embedded("a", source="arxiv:2410.12345", tier=CorpusTier.TIER_3_RESEARCH),
            _embedded("b", source="arxiv:2410.12345", tier=CorpusTier.TIER_3_RESEARCH),
            _embedded("c", source="other:1", tier=CorpusTier.TIER_3_RESEARCH),
        ],
    )
    n = await store.update_tier_for_source(
        "arxiv:2410.12345", CorpusTier.TIER_1_FOUNDATIONS,
    )
    assert n == 2

    upgraded = await store.list_chunks_for_source("arxiv:2410.12345")
    assert all(c.chunk.tier is CorpusTier.TIER_1_FOUNDATIONS for c in upgraded)
    other = await store.list_chunks_for_source("other:1")
    assert all(c.chunk.tier is CorpusTier.TIER_3_RESEARCH for c in other)


async def test_update_tier_for_unknown_source_is_zero(
    store: QdrantVectorStore,
) -> None:
    n = await store.update_tier_for_source(
        "ghost:1", CorpusTier.TIER_1_FOUNDATIONS,
    )
    assert n == 0
