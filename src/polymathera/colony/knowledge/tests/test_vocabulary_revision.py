"""Revision-pass candidate generation + judged proposals."""

from __future__ import annotations

from collections import Counter

import pytest

from polymathera.colony.knowledge.persistence import KgFile, PersistedClaim
from polymathera.colony.knowledge.vocabulary import (
    VocabFile,
    VocabOperation,
    VocabOpType,
    VocabTerm,
    apply_operation,
)
from polymathera.colony.knowledge.vocabulary_revision import (
    CandidateCluster,
    ProposalJudgement,
    dedupe_clusters,
    judgement_to_operations,
    lexical_clusters,
    propose_operations,
    type_signature_clusters,
)


def _kg(*triples: tuple[str, str, str]) -> KgFile:
    return KgFile(claims=[
        PersistedClaim(
            subject=s, predicate=p, object=o,
            citation={"source_uri": "repo:kb/x.pdf"},
        )
        for s, p, o in triples
    ])


def test_lexical_clusters_group_normal_forms() -> None:
    usage = Counter({"published_in": 3, "Published-In": 1, "uses": 5})
    clusters = lexical_clusters(usage)
    assert len(clusters) == 1
    assert clusters[0].members == ["Published-In", "published_in"]
    assert clusters[0].signal == "lexical"


def test_type_signature_clusters_need_population_overlap() -> None:
    kg = _kg(
        ("a", "authored", "paper1"), ("a", "authored", "paper2"),
        ("a", "has_author", "paper1"), ("a", "has_author", "paper2"),
        ("x", "measures", "field"),
    )
    clusters = type_signature_clusters(kg)
    assert len(clusters) == 1
    assert clusters[0].members == ["authored", "has_author"]


def test_dedupe_combines_signals_as_evidence() -> None:
    a = CandidateCluster(members=["p", "q"], signal="lexical", evidence="e1")
    b = CandidateCluster(members=["p", "q"], signal="embedding", evidence="e2")
    merged = dedupe_clusters([a, b])
    assert len(merged) == 1
    assert merged[0].signal == "lexical+embedding"


def test_judgement_translates_to_unapproved_merge_ops() -> None:
    vocab = VocabFile(terms={
        "authored": VocabTerm(name="authored"),
        "has_author": VocabTerm(name="has_author"),
    })
    cluster = CandidateCluster(
        members=["authored", "has_author"], signal="lexical",
    )
    judgement = ProposalJudgement(
        should_merge=True, canonical="has_author",
        rationale="same relation", confidence=0.9,
    )
    ops = judgement_to_operations(cluster, judgement, vocab)
    assert len(ops) == 1
    op = ops[0]
    assert op.op_type is VocabOpType.MERGE
    assert (op.term, op.target) == ("authored", "has_author")
    assert op.approved_by == ""  # proposals arrive unapproved
    # And the gate holds: unapproved destructive op refuses to apply.
    from polymathera.colony.knowledge.vocabulary import VocabError
    with pytest.raises(VocabError, match="approver"):
        apply_operation(vocab, op)


@pytest.mark.asyncio
async def test_propose_operations_end_to_end_with_stub_judge() -> None:
    kg = _kg(
        ("a", "published_in", "j1"), ("b", "Published-In", "j1"),
        ("c", "uses", "tool"),
    )
    vocab = VocabFile(terms={
        p: VocabTerm(name=p) for p in
        ("published_in", "Published-In", "uses")
    })
    judged: list[str] = []

    async def _stub_llm(prompt: str, schema):
        judged.append(prompt)
        return ProposalJudgement(
            should_merge=True, canonical="published_in",
            rationale="variants", confidence=0.8,
        )

    proposals = await propose_operations(vocab, kg, _stub_llm)
    assert len(judged) == 1  # one cluster crossed MIN_CLUSTER_USAGE
    assert "published_in" in judged[0]  # prompt carries members + usage
    assert [
        (op.op_type, op.term, op.target) for op in proposals
    ] == [(VocabOpType.MERGE, "Published-In", "published_in")]


@pytest.mark.asyncio
async def test_embedding_clusters_vectorized_groups_identical_names() -> None:
    """Numpy path: identical vectors cluster, orthogonal ones don't.
    (The pure-Python pairwise loop took ~3 min on 4.6k predicates in
    the 2026-08-05 production pass — vectorization is load-bearing.)"""

    from polymathera.colony.knowledge.vocabulary_revision import (
        embedding_clusters,
    )

    class _StubEmbedder:
        async def embed(self, texts):
            vecs = {
                "has author": [1.0, 0.0, 0.0],
                "authored by": [1.0, 0.0, 0.0],
                "measures": [0.0, 1.0, 0.0],
            }
            return [vecs[t] for t in texts]

    usage = Counter({"has_author": 3, "authored_by": 2, "measures": 5})
    clusters = await embedding_clusters(usage, _StubEmbedder())
    assert len(clusters) == 1
    assert clusters[0].members == ["authored_by", "has_author"]
    assert clusters[0].signal == "embedding"


@pytest.mark.asyncio
async def test_propose_operations_reports_progress() -> None:
    """The progress callback sees the clustering phases and a
    per-cluster judging heartbeat — multi-minute passes must never be
    a black box (2026-08-05: 3 silent minutes in the dashboard)."""

    kg = _kg(
        ("a", "published_in", "j1"), ("b", "Published-In", "j1"),
    )
    vocab = VocabFile(terms={
        p: VocabTerm(name=p) for p in ("published_in", "Published-In")
    })

    async def _stub_llm(prompt: str, schema):
        return ProposalJudgement(should_merge=False)

    messages: list[str] = []
    await propose_operations(
        vocab, kg, _stub_llm, on_progress=messages.append,
    )
    assert any("Clustering" in m for m in messages)
    assert any(m.startswith("Judging ") for m in messages)
    assert any(m.startswith("Judged 1/") for m in messages)


@pytest.mark.asyncio
async def test_judging_runs_concurrently_under_semaphore() -> None:
    """Judge calls overlap up to JUDGE_CONCURRENCY — sequential
    judging ran ~10s/cluster in production (2026-08-05: 35/200 in
    ~6 minutes) while the deployment's concurrency budget sat idle."""

    import asyncio

    from polymathera.colony.knowledge.vocabulary_revision import (
        JUDGE_CONCURRENCY,
    )

    # 12 clusters, each with 2 claims so MIN_CLUSTER_USAGE passes.
    triples = []
    for n in range(12):
        triples += [
            (f"s{n}", f"pred_{n}", "o"), (f"t{n}", f"Pred-{n}", "o"),
        ]
    kg = _kg(*triples)
    vocab = VocabFile(terms={
        c.predicate: VocabTerm(name=c.predicate) for c in kg.claims
    })

    live = 0
    peak = 0

    async def _slow_llm(prompt: str, schema):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.02)
        live -= 1
        return ProposalJudgement(should_merge=False)

    await propose_operations(vocab, kg, _slow_llm)
    assert peak > 1, "judge calls never overlapped"
    assert peak <= JUDGE_CONCURRENCY
