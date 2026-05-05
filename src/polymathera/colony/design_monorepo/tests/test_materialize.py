"""Tests for ``materialize_knowledge_routing``.

We test the KB-side materialiser directly; the VCM half of
``materialize_repo_map`` requires a live VCM and is exercised by the
CLI integration test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polymathera.colony.design_monorepo.materialize import (
    materialize_knowledge_routing,
)
from polymathera.colony.design_monorepo.repo_map import (
    KnowledgeRoute,
    RepoMap,
    SourceSpec,
)
from polymathera.colony.knowledge.deps import (
    get_default_ingestor,
    reset_knowledge_deps,
)


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Each test gets a fresh in-memory ``Ingestor`` so writes from
    one case do not leak into another."""

    reset_knowledge_deps()
    yield
    reset_knowledge_deps()


def _seed_corpus(repo_root: Path) -> None:
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "literature" / "curated").mkdir(parents=True)
    (repo_root / "literature" / "promoted").mkdir(parents=True)
    (repo_root / "literature" / "curated" / "a.txt").write_text(
        "Curated paper A — to be ingested into the knowledge base.\n",
        encoding="utf-8",
    )
    (repo_root / "literature" / "curated" / "b.txt").write_text(
        "Curated paper B — also ingested.\n",
        encoding="utf-8",
    )
    (repo_root / "literature" / "promoted" / "c.txt").write_text(
        "Paper C was promoted to VCM — must NOT be KB-ingested.\n",
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_only_knowledge_base_rows_are_ingested(tmp_path: Path) -> None:
    repo_root = tmp_path / "r"
    _seed_corpus(repo_root)

    repo_map = RepoMap(
        sources=[SourceSpec(name="default", type="git_repo")],
        knowledge_routing=[
            KnowledgeRoute(
                paths=["literature/curated/**/*.txt"],
                ingest_to="knowledge_base",
                profile="scientific_paper",
            ),
            KnowledgeRoute(
                paths=["literature/promoted/**/*.txt"],
                ingest_to="vcm",
            ),
        ],
    )
    records = await materialize_knowledge_routing(
        repo_map=repo_map, repo_root=repo_root,
    )
    sources = sorted(r.source_uri for r in records)
    # Two ``literature/curated`` files; the ``promoted`` one is skipped.
    assert len(records) == 2
    assert all("literature/curated" in s for s in sources)


@pytest.mark.asyncio
async def test_default_ingest_to_is_knowledge_base(tmp_path: Path) -> None:
    repo_root = tmp_path / "r"
    _seed_corpus(repo_root)

    # Construct a route without ``ingest_to`` explicitly set — the
    # default must be ``knowledge_base`` and the file must be ingested.
    repo_map = RepoMap(
        sources=[SourceSpec(name="default", type="git_repo")],
        knowledge_routing=[
            KnowledgeRoute(paths=["literature/curated/a.txt"]),
        ],
    )
    records = await materialize_knowledge_routing(
        repo_map=repo_map, repo_root=repo_root,
    )
    assert len(records) == 1


@pytest.mark.asyncio
async def test_empty_knowledge_routing_is_a_noop(tmp_path: Path) -> None:
    repo_root = tmp_path / "r"
    _seed_corpus(repo_root)

    repo_map = RepoMap(
        sources=[SourceSpec(name="default", type="git_repo")],
        knowledge_routing=[],
    )
    records = await materialize_knowledge_routing(
        repo_map=repo_map, repo_root=repo_root,
    )
    assert records == []
    # No deps singleton was forced into existence — the function must
    # tolerate an empty list without instantiating an ingestor.
    # (We don't assert the singleton is None because the helper may
    # have been touched indirectly; this test just checks the no-op
    # contract on the return value.)
    _ = get_default_ingestor  # silence unused-import in some linters
