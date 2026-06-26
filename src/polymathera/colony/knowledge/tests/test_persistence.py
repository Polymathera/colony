"""Round-trip + branch-annotation tests for the KG persistence layer."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from polymathera.colony.knowledge.models import Claim, CitationSpan
from polymathera.colony.knowledge.persistence import (
    KG_FILE_RELATIVE_PATH,
    KgFile,
    PersistedClaim,
    SCHEMA_VERSION,
    SNAPSHOT_CALLBACK_NAME,
    load_branch_from_text,
    register_kg_snapshot_callback,
    snapshot_branch_to_file,
)
from polymathera.colony.knowledge.stores.graph import (
    InMemoryGraphStore,
    set_current_branch,
)


def _claim(subject: str, obj: str, *, predicate: str = "links",
           confidence: float = 0.9, source: str = "lit:1",
           extractor: str = "deterministic@v1") -> Claim:
    return Claim(
        subject=subject, predicate=predicate, object=obj,
        confidence=confidence,
        citation=CitationSpan(source_uri=source, section_path="3.1",
                              char_start=10, char_end=42),
        provenance={"extractor": extractor, "run_id": "r-1"},
    )


@pytest.fixture
def isolated_deps(monkeypatch):
    """Bind the process-wide knowledge deps to a fresh
    :class:`InMemoryGraphStore` for the duration of the test and
    reset the pre-commit registry so tests don't pollute each
    other."""

    from polymathera.colony.knowledge import deps as deps_mod
    from polymathera.colony.design_monorepo.commit_hooks import (
        reset_pre_commit_registry,
    )

    reset_pre_commit_registry()
    monkeypatch.setattr(deps_mod, "_deps", None)
    monkeypatch.setattr(deps_mod, "_ingestor", None)
    store = InMemoryGraphStore()
    deps_mod.set_knowledge_deps(graph_store=store)
    yield store
    monkeypatch.setattr(deps_mod, "_deps", None)
    monkeypatch.setattr(deps_mod, "_ingestor", None)
    reset_pre_commit_registry()


@pytest.mark.asyncio
async def test_persisted_claim_round_trip_preserves_provenance() -> None:
    c = _claim("A", "B")
    p = PersistedClaim.from_claim(c)
    back = p.to_claim()
    assert back.subject == c.subject
    assert back.predicate == c.predicate
    assert back.object_ == c.object_
    assert back.confidence == c.confidence
    assert back.citation.source_uri == c.citation.source_uri
    assert back.citation.section_path == c.citation.section_path
    assert back.provenance == c.provenance


@pytest.mark.asyncio
async def test_kg_file_json_is_stable_and_byte_identical(
    isolated_deps,
) -> None:
    store = isolated_deps
    with set_current_branch("feature"):
        for s, o in [("a", "b"), ("c", "d"), ("a", "c")]:
            await store.add_claim(_claim(s, o))
    claims = [c async for c in store.export_claims(branch="feature")]
    file1 = KgFile(
        claims=sorted(
            [PersistedClaim.from_claim(c) for c in claims],
            key=lambda x: (x.subject, x.predicate, x.object_),
        ),
    ).to_json()
    file2 = KgFile(
        claims=sorted(
            [PersistedClaim.from_claim(c) for c in claims],
            key=lambda x: (x.subject, x.predicate, x.object_),
        ),
    ).to_json()
    assert file1 == file2
    payload = json.loads(file1)
    assert payload["version"] == SCHEMA_VERSION


@pytest.mark.asyncio
async def test_snapshot_branch_to_file_skips_empty_branch(
    tmp_path, isolated_deps,
) -> None:
    path, count = await snapshot_branch_to_file(tmp_path, "empty-branch")
    assert count == 0
    assert not path.exists()


@pytest.mark.asyncio
async def test_snapshot_branch_to_file_writes_only_current_branch(
    tmp_path, isolated_deps,
) -> None:
    store = isolated_deps
    with set_current_branch("main"):
        await store.add_claim(_claim("a", "b"))
    with set_current_branch("dev"):
        await store.add_claim(_claim("c", "d"))
    path, count = await snapshot_branch_to_file(tmp_path, "main")
    assert path == tmp_path / KG_FILE_RELATIVE_PATH
    assert count == 1
    file = KgFile.from_json(path.read_text(encoding="utf-8"))
    subjects = {c.subject for c in file.claims}
    assert subjects == {"a"}


@pytest.mark.asyncio
async def test_load_branch_from_text_imports_with_tagging(
    isolated_deps,
) -> None:
    store = isolated_deps
    payload = KgFile(claims=[PersistedClaim.from_claim(_claim("x", "y"))])
    result = await load_branch_from_text(payload.to_json(), "main")
    assert result == {
        "claims_in_file": 1,
        "claims_newly_added": 1,
        "claims_newly_tagged": 0,
        "claims_already_present": 0,
    }
    nodes, edges = await store.count(branch_filter="main")
    assert (nodes, edges) == (2, 1)


@pytest.mark.asyncio
async def test_load_branch_from_text_is_idempotent_then_tags_on_second_branch(
    isolated_deps,
) -> None:
    payload = KgFile(claims=[PersistedClaim.from_claim(_claim("x", "y"))])
    text = payload.to_json()
    first = await load_branch_from_text(text, "main")
    assert first["claims_newly_added"] == 1
    same = await load_branch_from_text(text, "main")
    assert same == {
        "claims_in_file": 1,
        "claims_newly_added": 0,
        "claims_newly_tagged": 0,
        "claims_already_present": 1,
    }
    diff = await load_branch_from_text(text, "feature")
    assert diff == {
        "claims_in_file": 1,
        "claims_newly_added": 0,
        "claims_newly_tagged": 1,
        "claims_already_present": 0,
    }


@pytest.mark.asyncio
async def test_register_kg_snapshot_callback_is_idempotent(
    isolated_deps,
) -> None:
    from polymathera.colony.design_monorepo.commit_hooks import (
        get_pre_commit_registry,
    )

    register_kg_snapshot_callback()
    register_kg_snapshot_callback()
    assert get_pre_commit_registry().names().count(
        SNAPSHOT_CALLBACK_NAME,
    ) == 1


@pytest.mark.asyncio
async def test_snapshot_callback_writes_file_into_commit_path(
    tmp_path, isolated_deps,
) -> None:
    """End-to-end: simulate the callback firing as part of a commit;
    assert the .colony/colony.kg.json file is materialised in the
    working tree with the current branch's claims."""

    from polymathera.colony.design_monorepo.commit_hooks import (
        PreCommitContext, get_pre_commit_registry,
    )

    store = isolated_deps
    with set_current_branch("main"):
        await store.add_claim(_claim("alpha", "beta"))

    register_kg_snapshot_callback()
    ctx = PreCommitContext(
        client=None,
        identity=None,
        message="m",
        branch="main",
        paths=None,
        working_dir=tmp_path,
    )
    await get_pre_commit_registry().fire_all(ctx)
    assert (tmp_path / KG_FILE_RELATIVE_PATH).is_file()
    file = KgFile.from_json(
        (tmp_path / KG_FILE_RELATIVE_PATH).read_text(encoding="utf-8"),
    )
    assert len(file.claims) == 1
    assert file.claims[0].subject == "alpha"
