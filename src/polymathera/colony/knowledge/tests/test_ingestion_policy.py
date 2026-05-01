"""Tests for the ``IngestionPolicy`` idempotency contract.

Q-S0a contract (cps/phase_q_opm_meg_demo_plan.md): bulk-acquisition
pipelines and re-runs MUST be safe by default. The Ingestor's
``policy`` parameter governs the behaviour:

- ``SKIP_IF_PRESENT`` (default): if any chunks exist for the source
  URI, do nothing.
- ``UPGRADE_TIER``: bump persisted tier on existing chunks if and
  only if the new tier outranks the existing one (per
  ``tier_priority``); otherwise behave as ``SKIP_IF_PRESENT``.
- ``OVERWRITE``: delete all chunks for the source URI exactly, then
  ingest fresh.

These tests are framework-level — they exercise the Ingestor against
the ``InMemoryVectorStore`` so they don't depend on Qdrant / GROBID.
The Qdrant-backed integration test in ``tests/integration/`` covers
the same contract end-to-end on a live container.
"""

from __future__ import annotations

import pytest

from polymathera.colony.knowledge import (
    CorpusTier,
    InMemoryEmbedder,
    InMemoryGraphStore,
    InMemoryVectorStore,
    Ingestor,
    IngestionPolicy,
    IngestionStatus,
    KnowledgeFormat,
    tier_priority,
)


pytestmark = pytest.mark.asyncio


SAMPLE_TEXT = (
    "# Spin-Exchange Relaxation-Free Magnetometry\n\n"
    "## Cell parameters\n\n"
    "Rubidium-87 vapor cells operated at 150 °C produce sensitivities "
    "in the femtotesla regime.\n"
)
"""Lifted-from-real-OPM-MEG (Optically-Pumped Magnetometer
Magnetoencephalography) phrasing so the chunker emits multiple
chunks. The exact prose doesn't matter — only that re-ingestion
behaves correctly."""


SOURCE_A = "paper:opm:cell_2024"
SOURCE_B = "paper:opm:cell_2024_revised"


@pytest.fixture
def ingestor() -> Ingestor:
    """Fresh Ingestor over an in-memory vector store + graph store
    (no claim extractors, review-rate 0 so we don't perturb tests)."""

    return Ingestor(
        embedder=InMemoryEmbedder(),
        vector_store=InMemoryVectorStore(),
        graph_store=InMemoryGraphStore(),
        review_sample_rate=0.0,
    )


# ---------------------------------------------------------------------------
# tier_priority — total ordering invariant
# ---------------------------------------------------------------------------


async def test_tier_priority_strictly_ordered() -> None:
    """The priority order encodes the master §3.2 rule that
    foundational textbooks outrank standards outrank research outrank
    software-docs outrank untiered. (Async-marked only because the
    surrounding module sets ``pytestmark = pytest.mark.asyncio``;
    the body is synchronous.)"""

    assert (
        tier_priority(CorpusTier.TIER_1_FOUNDATIONS)
        > tier_priority(CorpusTier.TIER_2_STANDARDS)
        > tier_priority(CorpusTier.TIER_3_RESEARCH)
        > tier_priority(CorpusTier.TIER_4_SOFTWARE_DOCS)
        > tier_priority(CorpusTier.UNTIERED)
    )


# ---------------------------------------------------------------------------
# Default: SKIP_IF_PRESENT
# ---------------------------------------------------------------------------


async def test_default_policy_is_skip_if_present(ingestor: Ingestor) -> None:
    """A second ingestion of the same source URI must short-circuit
    without parsing / chunking / embedding. Verifies the *default* —
    callers should not have to remember the safety contract."""

    first = await ingestor.ingest_text(
        SAMPLE_TEXT,
        source_uri=SOURCE_A,
        fmt=KnowledgeFormat.MARKDOWN,
        tier=CorpusTier.TIER_3_RESEARCH,
    )
    assert first.status is IngestionStatus.COMPLETED
    assert first.chunks_produced > 0

    initial_count = await ingestor._vector_store.count()
    second = await ingestor.ingest_text(
        SAMPLE_TEXT,
        source_uri=SOURCE_A,
        fmt=KnowledgeFormat.MARKDOWN,
        tier=CorpusTier.TIER_3_RESEARCH,
    )
    assert second.status is IngestionStatus.SKIPPED_ALREADY_PRESENT
    assert second.chunks_produced == 0
    # Vector store unchanged.
    assert await ingestor._vector_store.count() == initial_count


async def test_skip_if_present_preserves_existing_tier(
    ingestor: Ingestor,
) -> None:
    """Re-ingest with a *different* tier under the default policy must
    NOT change the persisted tier — this is the user-seed safety
    property: bulk-acquire cannot demote a Tier 1 textbook to
    untiered."""

    await ingestor.ingest_text(
        SAMPLE_TEXT,
        source_uri=SOURCE_A,
        fmt=KnowledgeFormat.MARKDOWN,
        tier=CorpusTier.TIER_1_FOUNDATIONS,
    )
    await ingestor.ingest_text(
        SAMPLE_TEXT,
        source_uri=SOURCE_A,
        fmt=KnowledgeFormat.MARKDOWN,
        tier=CorpusTier.UNTIERED,
        policy=IngestionPolicy.SKIP_IF_PRESENT,
    )
    chunks = await ingestor._vector_store.list_chunks_for_source(SOURCE_A)
    assert chunks
    for ch in chunks:
        assert ch.chunk.tier is CorpusTier.TIER_1_FOUNDATIONS


async def test_skip_records_document_hash(ingestor: Ingestor) -> None:
    """The skipped record still surfaces the document hash so callers
    can detect content drift and choose to re-run with OVERWRITE."""

    first = await ingestor.ingest_text(
        SAMPLE_TEXT,
        source_uri=SOURCE_A,
        fmt=KnowledgeFormat.MARKDOWN,
        tier=CorpusTier.TIER_3_RESEARCH,
    )
    assert first.document_hash
    second = await ingestor.ingest_text(
        SAMPLE_TEXT + "\n\n## Addendum\n\nAdditional content.\n",
        source_uri=SOURCE_A,
        fmt=KnowledgeFormat.MARKDOWN,
        tier=CorpusTier.TIER_3_RESEARCH,
    )
    assert second.status is IngestionStatus.SKIPPED_ALREADY_PRESENT
    # Different content → different hash, even though we skipped.
    assert second.document_hash
    assert second.document_hash != first.document_hash


# ---------------------------------------------------------------------------
# UPGRADE_TIER — monotone in both directions
# ---------------------------------------------------------------------------


async def test_upgrade_tier_higher_priority_upgrades(
    ingestor: Ingestor,
) -> None:
    """Re-ingest with a *higher-priority* tier under UPGRADE_TIER
    bumps every existing chunk's tier in place."""

    await ingestor.ingest_text(
        SAMPLE_TEXT,
        source_uri=SOURCE_A,
        fmt=KnowledgeFormat.MARKDOWN,
        tier=CorpusTier.TIER_3_RESEARCH,
    )
    chunks_before = await ingestor._vector_store.list_chunks_for_source(SOURCE_A)
    n = len(chunks_before)
    assert n > 0
    assert all(
        c.chunk.tier is CorpusTier.TIER_3_RESEARCH for c in chunks_before
    )

    record = await ingestor.ingest_text(
        SAMPLE_TEXT,
        source_uri=SOURCE_A,
        fmt=KnowledgeFormat.MARKDOWN,
        tier=CorpusTier.TIER_1_FOUNDATIONS,
        policy=IngestionPolicy.UPGRADE_TIER,
    )
    assert record.status is IngestionStatus.TIER_UPGRADED
    assert record.chunks_produced == n  # same count, just re-tiered

    chunks_after = await ingestor._vector_store.list_chunks_for_source(SOURCE_A)
    assert len(chunks_after) == n
    assert all(
        c.chunk.tier is CorpusTier.TIER_1_FOUNDATIONS for c in chunks_after
    )


async def test_upgrade_tier_lower_priority_is_skip(
    ingestor: Ingestor,
) -> None:
    """Re-ingest with a *lower-priority* tier under UPGRADE_TIER
    behaves as SKIP_IF_PRESENT — the existing tier is preserved."""

    await ingestor.ingest_text(
        SAMPLE_TEXT,
        source_uri=SOURCE_A,
        fmt=KnowledgeFormat.MARKDOWN,
        tier=CorpusTier.TIER_1_FOUNDATIONS,
    )
    record = await ingestor.ingest_text(
        SAMPLE_TEXT,
        source_uri=SOURCE_A,
        fmt=KnowledgeFormat.MARKDOWN,
        tier=CorpusTier.TIER_3_RESEARCH,
        policy=IngestionPolicy.UPGRADE_TIER,
    )
    assert record.status is IngestionStatus.SKIPPED_ALREADY_PRESENT
    chunks = await ingestor._vector_store.list_chunks_for_source(SOURCE_A)
    assert all(
        c.chunk.tier is CorpusTier.TIER_1_FOUNDATIONS for c in chunks
    )


async def test_upgrade_tier_same_priority_is_skip(
    ingestor: Ingestor,
) -> None:
    """Re-ingest at the *same* tier is also a no-op — UPGRADE_TIER
    is strict (>), not loose (≥)."""

    await ingestor.ingest_text(
        SAMPLE_TEXT,
        source_uri=SOURCE_A,
        fmt=KnowledgeFormat.MARKDOWN,
        tier=CorpusTier.TIER_3_RESEARCH,
    )
    record = await ingestor.ingest_text(
        SAMPLE_TEXT,
        source_uri=SOURCE_A,
        fmt=KnowledgeFormat.MARKDOWN,
        tier=CorpusTier.TIER_3_RESEARCH,
        policy=IngestionPolicy.UPGRADE_TIER,
    )
    assert record.status is IngestionStatus.SKIPPED_ALREADY_PRESENT


async def test_upgrade_tier_on_unknown_source_runs_full_pipeline(
    ingestor: Ingestor,
) -> None:
    """If the source has never been ingested, UPGRADE_TIER must fall
    through to the full pipeline — there's nothing to upgrade *to*."""

    record = await ingestor.ingest_text(
        SAMPLE_TEXT,
        source_uri=SOURCE_A,
        fmt=KnowledgeFormat.MARKDOWN,
        tier=CorpusTier.TIER_2_STANDARDS,
        policy=IngestionPolicy.UPGRADE_TIER,
    )
    assert record.status is IngestionStatus.COMPLETED
    assert record.chunks_produced > 0


# ---------------------------------------------------------------------------
# OVERWRITE
# ---------------------------------------------------------------------------


async def test_overwrite_replaces_existing(ingestor: Ingestor) -> None:
    """OVERWRITE deletes the old chunks and re-runs the pipeline. The
    final chunk count corresponds to the *new* content, not the sum
    of old + new (which is what unguarded re-ingestion would
    produce)."""

    await ingestor.ingest_text(
        SAMPLE_TEXT,
        source_uri=SOURCE_A,
        fmt=KnowledgeFormat.MARKDOWN,
        tier=CorpusTier.TIER_3_RESEARCH,
    )
    initial_count = await ingestor._vector_store.count()

    longer = SAMPLE_TEXT + "\n\n## Heading B\n\nSecond paragraph.\n"
    record = await ingestor.ingest_text(
        longer,
        source_uri=SOURCE_A,
        fmt=KnowledgeFormat.MARKDOWN,
        tier=CorpusTier.TIER_1_FOUNDATIONS,
        policy=IngestionPolicy.OVERWRITE,
    )
    assert record.status is IngestionStatus.COMPLETED
    assert record.chunks_produced > 0

    final = await ingestor._vector_store.list_chunks_for_source(SOURCE_A)
    assert all(
        c.chunk.tier is CorpusTier.TIER_1_FOUNDATIONS for c in final
    )
    # No accumulation: total count equals count from the second ingest.
    assert await ingestor._vector_store.count() == record.chunks_produced
    # Sanity: longer content produced at least as many chunks as the
    # original (so the overwrite did re-chunk).
    assert record.chunks_produced >= initial_count


async def test_overwrite_does_not_touch_sibling_sources(
    ingestor: Ingestor,
) -> None:
    """OVERWRITE is exact-match. A source URI that happens to share a
    prefix with the target must not be deleted (e.g.,
    ``paper:opm:cell_2024_revised`` vs ``paper:opm:cell_2024``)."""

    await ingestor.ingest_text(
        SAMPLE_TEXT,
        source_uri=SOURCE_A,  # paper:opm:cell_2024
        fmt=KnowledgeFormat.MARKDOWN,
        tier=CorpusTier.TIER_3_RESEARCH,
    )
    await ingestor.ingest_text(
        SAMPLE_TEXT,
        source_uri=SOURCE_B,  # paper:opm:cell_2024_revised — shares prefix
        fmt=KnowledgeFormat.MARKDOWN,
        tier=CorpusTier.TIER_3_RESEARCH,
    )
    sibling_chunks_before = await ingestor._vector_store.list_chunks_for_source(SOURCE_B)
    assert sibling_chunks_before

    await ingestor.ingest_text(
        SAMPLE_TEXT,
        source_uri=SOURCE_A,
        fmt=KnowledgeFormat.MARKDOWN,
        tier=CorpusTier.TIER_2_STANDARDS,
        policy=IngestionPolicy.OVERWRITE,
    )

    sibling_chunks_after = await ingestor._vector_store.list_chunks_for_source(SOURCE_B)
    assert len(sibling_chunks_after) == len(sibling_chunks_before)


# ---------------------------------------------------------------------------
# IngestionRecord field surfaces
# ---------------------------------------------------------------------------


async def test_record_carries_policy_and_hash(ingestor: Ingestor) -> None:
    record = await ingestor.ingest_text(
        SAMPLE_TEXT,
        source_uri=SOURCE_A,
        fmt=KnowledgeFormat.MARKDOWN,
        tier=CorpusTier.TIER_3_RESEARCH,
        policy=IngestionPolicy.SKIP_IF_PRESENT,
    )
    assert record.policy is IngestionPolicy.SKIP_IF_PRESENT
    assert len(record.document_hash) == 64  # sha256 hex length


# ---------------------------------------------------------------------------
# VectorStore.list_chunks_for_source contract
# ---------------------------------------------------------------------------


async def test_list_chunks_for_source_exact_match() -> None:
    """``list_chunks_for_source`` is exact, not prefix. Distinct from
    ``delete_by_source(prefix)``."""

    store = InMemoryVectorStore()
    ing = Ingestor(
        embedder=InMemoryEmbedder(),
        vector_store=store,
        review_sample_rate=0.0,
    )
    await ing.ingest_text(
        SAMPLE_TEXT,
        source_uri="src:1",
        fmt=KnowledgeFormat.MARKDOWN,
    )
    await ing.ingest_text(
        SAMPLE_TEXT,
        source_uri="src:1:variant",
        fmt=KnowledgeFormat.MARKDOWN,
    )
    only_src1 = await store.list_chunks_for_source("src:1")
    only_variant = await store.list_chunks_for_source("src:1:variant")
    # Strict equality — none of the variant chunks leak into "src:1".
    assert all(c.chunk.source == "src:1" for c in only_src1)
    assert all(c.chunk.source == "src:1:variant" for c in only_variant)
    assert len(only_src1) > 0 and len(only_variant) > 0


async def test_update_tier_for_source_returns_count() -> None:
    store = InMemoryVectorStore()
    ing = Ingestor(
        embedder=InMemoryEmbedder(),
        vector_store=store,
        review_sample_rate=0.0,
    )
    record = await ing.ingest_text(
        SAMPLE_TEXT,
        source_uri=SOURCE_A,
        fmt=KnowledgeFormat.MARKDOWN,
        tier=CorpusTier.TIER_3_RESEARCH,
    )
    n = await store.update_tier_for_source(
        SOURCE_A, CorpusTier.TIER_1_FOUNDATIONS,
    )
    assert n == record.chunks_produced
    assert n > 0
    chunks = await store.list_chunks_for_source(SOURCE_A)
    assert all(
        c.chunk.tier is CorpusTier.TIER_1_FOUNDATIONS for c in chunks
    )


async def test_update_tier_for_unknown_source_is_zero() -> None:
    store = InMemoryVectorStore()
    n = await store.update_tier_for_source(
        "ghost:1", CorpusTier.TIER_1_FOUNDATIONS,
    )
    assert n == 0
