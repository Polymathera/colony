"""Vocabulary revision — candidate generation + LLM-judged proposals
(``colony/predicate_vocabulary_plan.md`` §5).

Three independent candidate signals, per standard ontology-alignment
practice (lexical, semantic, extensional):

1. **Lexical**: normalization clusters — case / underscore / trivial
   morphology (``published_in`` / ``was_published_in``).
2. **Embedding**: cosine-similar predicate names via the process
   embedder (semantic similarity string matching can't see).
3. **Type-signature**: predicates whose subject/object populations
   overlap in the KG (extensional evidence of the same relation).

Clusters are then judged by the LLM (typed schema — same decoder-
enforced contract the claim extractor uses) into typed
:class:`VocabOperation` proposals with rationale + confidence.
Proposals change NOTHING: application is a separate, human-gated step
(:func:`..knowledge.vocabulary.apply_operation` refuses destructive
ops without an approver).
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .persistence import KgFile
from .vocabulary import (
    VocabFile,
    VocabOperation,
    VocabOpType,
    VocabTermStatus,
    normalize_predicate,
)


logger = logging.getLogger(__name__)

#: Cosine similarity above which two predicate embeddings cluster.
EMBEDDING_SIMILARITY_THRESHOLD = 0.90

#: Jaccard overlap of subject∪object populations above which two
#: predicates form an extensional candidate pair.
TYPE_SIGNATURE_JACCARD_THRESHOLD = 0.5

#: Don't judge clusters below this combined usage — merging two
#: singletons buys nothing; revision effort goes where claims are.
MIN_CLUSTER_USAGE = 2

#: Concurrent judge calls per pass. Mirrors the ingest fan-out's
#: rationale (materialize._INGEST_CONCURRENCY): the Anthropic
#: deployment's ``max_concurrent_requests=10`` is the actual
#: back-pressure; this semaphore just keeps the pass from queuing
#: hundreds of pending calls. Sequential judging ran ~10s/cluster in
#: production (2026-08-05: 35/200 in ~6 min) — an operator-facing
#: pass must saturate the deployment's existing concurrency budget.
JUDGE_CONCURRENCY = 10


class CandidateCluster(BaseModel):
    """A group of predicates one signal considers possibly-mergeable."""

    model_config = ConfigDict(extra="forbid")

    members: list[str]
    signal: str  # "lexical" | "embedding" | "type_signature"
    evidence: str = ""


class ProposalJudgement(BaseModel):
    """Schema the judge LLM must emit for one cluster (decoder-enforced).

    LLM-FACING SCHEMA: no ``ge``/``le``/length constraints — Anthropic
    structured outputs reject ``minimum``/``maximum`` etc. with a 400
    at request time (2026-08-05: the first revision pass died on
    exactly this). Value-range enforcement happens AFTER parse via the
    field_validator, per the :class:`ExtractedClaim` convention."""

    model_config = ConfigDict(extra="forbid")

    should_merge: bool
    canonical: str = ""
    """Which member (or better new name) survives, when merging."""
    add_broader: str = ""
    """Optional shared parent to attach to all members instead of (or
    in addition to) merging."""
    rationale: str = ""
    confidence: float = Field(default=0.5)

    @field_validator("confidence", mode="after")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        """Clamp instead of reject — one out-of-range judge value must
        not void the cluster's whole judgement."""

        return min(1.0, max(0.0, v))


def lexical_clusters(usage: Counter[str]) -> list[CandidateCluster]:
    groups: dict[str, list[str]] = defaultdict(list)
    for name in usage:
        groups[normalize_predicate(name)].append(name)
    return [
        CandidateCluster(
            members=sorted(members), signal="lexical",
            evidence=f"normalize to {key!r}",
        )
        for key, members in sorted(groups.items())
        if len(members) > 1
    ]


def type_signature_clusters(kg: KgFile) -> list[CandidateCluster]:
    """Predicates whose subject∪object populations overlap strongly.
    O(pairs) on distinct predicates — fine at 10^3-10^4 scale; revisit
    with minhashing if the vocabulary grows beyond that."""

    population: dict[str, set[str]] = defaultdict(set)
    for claim in kg.claims:
        population[claim.predicate].add(claim.subject)
        population[claim.predicate].add(claim.object_)
    names = sorted(population)
    clusters: list[CandidateCluster] = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = len(population[a] & population[b])
            if not inter:
                continue
            jaccard = inter / len(population[a] | population[b])
            if jaccard >= TYPE_SIGNATURE_JACCARD_THRESHOLD:
                clusters.append(CandidateCluster(
                    members=[a, b], signal="type_signature",
                    evidence=f"jaccard={jaccard:.2f} over {inter} shared entities",
                ))
    return clusters


async def embedding_clusters(
    usage: Counter[str], embedder,
) -> list[CandidateCluster]:
    """Single-link clusters over predicate-name embeddings. Uses the
    process embedder; O(n²) cosine at current scale."""

    names = sorted(usage)
    if len(names) < 2:
        return []
    vectors = await embedder.embed([n.replace("_", " ") for n in names])

    # Vectorized full cosine matrix: at 4.6k predicates the pure-Python
    # pairwise loop took ~3 minutes in production (2026-08-05 pass);
    # the matrix product is sub-second.
    import numpy as np

    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    unit = matrix / norms
    similar = (unit @ unit.T) >= EMBEDDING_SIMILARITY_THRESHOLD

    parent = list(range(len(names)))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    rows, cols = np.nonzero(np.triu(similar, k=1))
    for i, j in zip(rows.tolist(), cols.tolist()):
        parent[_find(i)] = _find(j)

    groups: dict[int, list[str]] = defaultdict(list)
    for i, name in enumerate(names):
        groups[_find(i)].append(name)
    return [
        CandidateCluster(
            members=sorted(members), signal="embedding",
            evidence=f"cosine ≥ {EMBEDDING_SIMILARITY_THRESHOLD}",
        )
        for members in groups.values()
        if len(members) > 1
    ]


def dedupe_clusters(
    clusters: Sequence[CandidateCluster],
) -> list[CandidateCluster]:
    """Same member-set found by several signals → one cluster with
    combined evidence (the multi-signal hit is itself evidence)."""

    by_members: dict[tuple[str, ...], CandidateCluster] = {}
    for cluster in clusters:
        key = tuple(cluster.members)
        if key in by_members:
            kept = by_members[key]
            kept.signal = f"{kept.signal}+{cluster.signal}"
            kept.evidence = f"{kept.evidence}; {cluster.evidence}"
        else:
            by_members[key] = cluster.model_copy()
    return list(by_members.values())


_JUDGE_PROMPT = (
    "You curate a scientific knowledge-graph predicate vocabulary.\n"
    "Candidate cluster (signal: {signal}; {evidence}):\n"
    "{members_block}\n\n"
    "Decide whether these predicates express the SAME relation and "
    "should merge into one canonical predicate (pick the clearest "
    "snake_case member, or a clearer new snake_case name). If they are "
    "genuinely distinct relations, do not merge; optionally suggest a "
    "shared broader parent predicate that would organize them. "
    "Prefer precision over aggressive merging: merging distinct "
    "relations destroys information, while a missed merge only costs "
    "a later pass."
)


async def judge_cluster(
    cluster: CandidateCluster,
    usage: Counter[str],
    llm,
) -> ProposalJudgement:
    """One typed judge call for one cluster. ``llm`` is the
    :data:`TypedLLMCallable` shape (prompt, schema) → validated model —
    the same decoder-enforced contract the claim extractor uses."""

    members_block = "\n".join(
        f"- {name} (used in {usage.get(name, 0)} claims)"
        for name in cluster.members
    )
    prompt = _JUDGE_PROMPT.format(
        signal=cluster.signal,
        evidence=cluster.evidence,
        members_block=members_block,
    )
    return await llm(prompt, ProposalJudgement)


def judgement_to_operations(
    cluster: CandidateCluster,
    judgement: ProposalJudgement,
    vocab: VocabFile,
) -> list[VocabOperation]:
    """Translate a judge verdict into typed, UNAPPROVED operations.
    Destructive ones will refuse to apply until an approver signs
    them (``apply_operation``'s gate)."""

    ops: list[VocabOperation] = []
    proposer = f"revision:{cluster.signal}"
    if judgement.should_merge and judgement.canonical:
        canonical = judgement.canonical
        for member in cluster.members:
            if member == canonical:
                continue
            op_type = VocabOpType.MERGE
            if canonical not in vocab.terms and member == cluster.members[0]:
                # New clearer name: first member renames into it, the
                # rest merge into the now-existing target.
                op_type = VocabOpType.RENAME
            ops.append(VocabOperation(
                op_type=op_type, term=member, target=canonical,
                rationale=judgement.rationale,
                confidence=judgement.confidence, proposed_by=proposer,
            ))
    if judgement.add_broader:
        for member in cluster.members:
            if member == judgement.add_broader:
                continue
            term = vocab.terms.get(member)
            if term is not None and term.status is VocabTermStatus.DEPRECATED:
                continue
            ops.append(VocabOperation(
                op_type=VocabOpType.ADD_BROADER,
                term=member, target=judgement.add_broader,
                rationale=judgement.rationale,
                confidence=judgement.confidence, proposed_by=proposer,
            ))
    return ops


async def propose_operations(
    vocab: VocabFile,
    kg: KgFile,
    llm,
    *,
    embedder=None,
    max_clusters: int | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> list[VocabOperation]:
    """Full candidate-generation + judging pass. Returns proposals
    ordered by judge confidence (desc); applies nothing.
    ``on_progress`` (optional) receives one-line phase/heartbeat
    messages so multi-minute passes are never a black box."""

    def _progress(message: str) -> None:
        if on_progress is not None:
            on_progress(message)

    usage: Counter[str] = Counter(c.predicate for c in kg.claims)
    _progress(
        f"Clustering {len(usage)} predicates (lexical + type-signature)...",
    )
    clusters = lexical_clusters(usage) + type_signature_clusters(kg)
    if embedder is not None:
        _progress(f"Clustering {len(usage)} predicates (embeddings)...")
        clusters += await embedding_clusters(usage, embedder)
    clusters = [
        c for c in dedupe_clusters(clusters)
        if sum(usage.get(m, 0) for m in c.members) >= MIN_CLUSTER_USAGE
    ]
    clusters.sort(
        key=lambda c: -sum(usage.get(m, 0) for m in c.members),
    )
    if max_clusters is not None:
        dropped = max(0, len(clusters) - max_clusters)
        if dropped:
            logger.info(
                "propose_operations: judging top %d clusters, deferring "
                "%d lower-usage ones to a later pass.",
                max_clusters, dropped,
            )
        clusters = clusters[:max_clusters]

    proposals: list[VocabOperation] = []
    total = len(clusters)
    completed = 0
    semaphore = asyncio.Semaphore(JUDGE_CONCURRENCY)
    _progress(
        f"Judging {total} clusters ({JUDGE_CONCURRENCY} in parallel)...",
    )

    async def _judge_one(cluster: CandidateCluster) -> list[VocabOperation]:
        nonlocal completed
        async with semaphore:
            judgement = await judge_cluster(cluster, usage, llm)
        ops = judgement_to_operations(cluster, judgement, vocab)
        completed += 1
        _progress(
            f"Judged {completed}/{total} clusters "
            f"({len(proposals)} operations proposed so far)...",
        )
        return ops

    # Fail-fast on a judge error (gather cancels the rest) — a broken
    # pass must surface, not limp to a partial proposal list. The SDK
    # already absorbs transient 429/529 with its own retries.
    for ops in await asyncio.gather(
        *(_judge_one(cluster) for cluster in clusters),
    ):
        proposals.extend(ops)
    proposals.sort(key=lambda op: -op.confidence)
    logger.info(
        "propose_operations: %d clusters judged → %d proposed operations.",
        len(clusters), len(proposals),
    )
    return proposals


__all__ = (
    "CandidateCluster",
    "EMBEDDING_SIMILARITY_THRESHOLD",
    "MIN_CLUSTER_USAGE",
    "ProposalJudgement",
    "TYPE_SIGNATURE_JACCARD_THRESHOLD",
    "dedupe_clusters",
    "embedding_clusters",
    "judge_cluster",
    "judgement_to_operations",
    "lexical_clusters",
    "propose_operations",
    "type_signature_clusters",
)
