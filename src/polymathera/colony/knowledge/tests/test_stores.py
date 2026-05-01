"""Tests for ``InMemoryVectorStore`` and ``InMemoryGraphStore``."""

from __future__ import annotations

import pytest

from polymathera.colony.knowledge import (
    Chunk,
    CitationSpan,
    Claim,
    CorpusTier,
    EmbeddedChunk,
    GraphEdge,
    GraphNode,
    GraphStoreError,
    InMemoryGraphStore,
    InMemoryVectorStore,
    KuzuGraphStore,
    QdrantVectorStore,
    RetrievalQuery,
)


pytestmark = pytest.mark.asyncio


def _embedded(
    chunk_id: str,
    *,
    text: str,
    vector: tuple[float, ...],
    source: str = "src:a",
    data_type: str = "paper_section",
    tier: CorpusTier = CorpusTier.UNTIERED,
) -> EmbeddedChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        text=text,
        token_count=max(1, len(text.split())),
        section_path="1",
        citation=CitationSpan(source_uri=source, section_path="1", char_start=0, char_end=len(text)),
        data_type=data_type,
        source=source,
        tier=tier,
    )
    return EmbeddedChunk(chunk=chunk, vector=vector, embedder="test:fixture")


# ---- VectorStore ----------------------------------------------------


async def test_vector_store_upsert_and_count() -> None:
    store = InMemoryVectorStore()
    n = await store.upsert(
        [
            _embedded("c1", text="alpha", vector=(1.0, 0.0)),
            _embedded("c2", text="beta", vector=(0.0, 1.0)),
        ]
    )
    assert n == 2
    assert await store.count() == 2


async def test_vector_store_search_cosine() -> None:
    store = InMemoryVectorStore()
    await store.upsert(
        [
            _embedded("c1", text="alpha", vector=(1.0, 0.0)),
            _embedded("c2", text="beta", vector=(0.7, 0.7)),
            _embedded("c3", text="gamma", vector=(0.0, 1.0)),
        ]
    )
    hits = await store.search(
        query_vector=(0.95, 0.05),
        query=RetrievalQuery(text="x", max_results=2),
    )
    assert len(hits) == 2
    assert hits[0].chunk.chunk_id == "c1"
    assert hits[1].chunk.chunk_id == "c2"


async def test_vector_store_filter_by_data_type() -> None:
    store = InMemoryVectorStore()
    await store.upsert(
        [
            _embedded("a", text="x", vector=(1.0, 0.0), data_type="paper_section"),
            _embedded("b", text="y", vector=(1.0, 0.0), data_type="standard_clause"),
        ]
    )
    hits = await store.search(
        query_vector=(1.0, 0.0),
        query=RetrievalQuery(text="x", data_types=("standard_clause",)),
    )
    assert [h.chunk.chunk_id for h in hits] == ["b"]


async def test_vector_store_filter_by_source_prefix() -> None:
    store = InMemoryVectorStore()
    await store.upsert(
        [
            _embedded("a", text="x", vector=(1.0, 0.0), source="git:repo:main:1"),
            _embedded("b", text="y", vector=(1.0, 0.0), source="arxiv:1234"),
        ]
    )
    hits = await store.search(
        query_vector=(1.0, 0.0),
        query=RetrievalQuery(text="x", source_prefix="arxiv:"),
    )
    assert [h.chunk.chunk_id for h in hits] == ["b"]


async def test_vector_store_delete_by_source() -> None:
    store = InMemoryVectorStore()
    await store.upsert(
        [
            _embedded("a", text="x", vector=(1.0, 0.0), source="arxiv:1"),
            _embedded("b", text="y", vector=(1.0, 0.0), source="arxiv:2"),
            _embedded("c", text="z", vector=(1.0, 0.0), source="git:repo:main:1"),
        ]
    )
    n = await store.delete_by_source("arxiv:")
    assert n == 2
    assert await store.count() == 1


async def test_qdrant_constructible_without_client_calls() -> None:
    """Phase C1b: QdrantVectorStore is now a real implementation; the
    in-process unit + integration tests exercise it. Here we only
    verify the construction path remains importable + typeable
    against the abstract surface."""

    store = QdrantVectorStore(
        client=object(), collection="x", embedder_id="e", dimensions=4,
    )
    assert store.collection_name == "x"


# ---- GraphStore -----------------------------------------------------


async def test_graph_store_add_node_and_edge() -> None:
    g = InMemoryGraphStore()
    await g.add_node(GraphNode(node_id="a", label="Entity"))
    await g.add_node(GraphNode(node_id="b", label="Entity"))
    await g.add_edge(
        GraphEdge(edge_id="e1", source_id="a", target_id="b", predicate="r"),
    )
    nodes, edges = await g.count()
    assert (nodes, edges) == (2, 1)


async def test_graph_store_orphan_edge_raises() -> None:
    g = InMemoryGraphStore()
    await g.add_node(GraphNode(node_id="a"))
    with pytest.raises(GraphStoreError):
        await g.add_edge(
            GraphEdge(edge_id="e", source_id="a", target_id="missing", predicate="r"),
        )


async def test_graph_store_add_claim() -> None:
    g = InMemoryGraphStore()
    claim = Claim(
        subject="MAST-U",
        predicate="is_a",
        object="spherical tokamak",
        citation=CitationSpan(source_uri="book:wesson", section_path="1"),
    )
    s, o, edge = await g.add_claim(claim)
    assert s.node_id == "mast_u"
    assert o.node_id == "spherical_tokamak"
    assert edge.predicate == "is_a"


async def test_graph_store_neighbours() -> None:
    g = InMemoryGraphStore()
    claim_a = Claim(
        subject="JET", predicate="requires", object="DT fuel",
        citation=CitationSpan(source_uri="x", section_path="1"),
    )
    claim_b = Claim(
        subject="DT fuel", predicate="contains", object="tritium",
        citation=CitationSpan(source_uri="x", section_path="1"),
    )
    await g.add_claim(claim_a)
    await g.add_claim(claim_b)
    result = await g.neighbours("jet", depth=2)
    ids = {n.node_id for n in result.nodes}
    assert {"jet", "dt_fuel", "tritium"} <= ids


async def test_graph_store_query_dsl() -> None:
    g = InMemoryGraphStore()
    await g.add_claim(
        Claim(
            subject="A", predicate="is_a", object="thing",
            citation=CitationSpan(source_uri="x", section_path="1"),
        ),
    )
    await g.add_claim(
        Claim(
            subject="B", predicate="is_a", object="thing",
            citation=CitationSpan(source_uri="x", section_path="1"),
        ),
    )
    await g.add_claim(
        Claim(
            subject="C", predicate="requires", object="thing",
            citation=CitationSpan(source_uri="x", section_path="1"),
        ),
    )
    result = await g.query("MATCH (s)-[r:is_a]->(o) RETURN s LIMIT 5")
    assert len(result.edges) == 2
    assert all(e.predicate == "is_a" for e in result.edges)


async def test_graph_store_query_invalid_dsl() -> None:
    g = InMemoryGraphStore()
    with pytest.raises(GraphStoreError):
        await g.query("DROP TABLE knowledge_graph;")


async def test_kuzu_constructible_with_injected_connection() -> None:
    """Phase C1b: KuzuGraphStore is now a real embedded-Kùzu store;
    the dedicated ``test_kuzu_graph_store.py`` file exercises it
    against a real database. Here we only verify the constructor
    surface remains importable + typeable against the ABC."""

    g = KuzuGraphStore(connection=object())
    assert g.connection is not None
