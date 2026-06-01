"""Tests for :class:`SystemDesignCapability` — Phase 1 surface
(``summarise_design_context`` + ``search_design_context`` with raw
path; kuzu/vcm paths return structured not-yet-available errors).

The capability inherits :class:`DesignMonorepoCapabilityBase`, which
wants a real on-disk git repo at ``working_dir``. We build a minimal
one per test rather than reaching for this directory's bootstrapped_repo
fixture (the design-monorepo bootstrap pulls in manifest / identity
scaffolding the SystemDesignCapability surface doesn't need).
"""

from __future__ import annotations

from pathlib import Path

import git
import pytest

from polymathera.colony.design_monorepo.capabilities import (
    SystemDesignCapability,
)


pytestmark = pytest.mark.asyncio


def _make_repo(root: Path) -> None:
    """Init a git repo with a no-op commit so ``.git/`` exists."""

    repo = git.Repo.init(root, initial_branch="main")
    repo.config_writer().set_value("user", "email", "t@t").release()
    repo.config_writer().set_value("user", "name", "t").release()
    sentinel = root / ".init"
    sentinel.write_text("init\n", encoding="utf-8")
    repo.index.add([str(sentinel)])
    repo.index.commit("init")


def _make_capability(tmp_path: Path) -> SystemDesignCapability:
    """Capability instance bound to a fresh on-disk repo, no agent."""

    _make_repo(tmp_path)
    return SystemDesignCapability(
        agent=None, scope_id="test", working_dir=tmp_path,
    )


# ---------------------------------------------------------------------------
# summarise_design_context
# ---------------------------------------------------------------------------


async def test_summarise_returns_helpful_message_when_no_rows(
    tmp_path: Path,
) -> None:
    """Empty / missing ``design_context_sources:`` block returns an
    empty report with a clear ``message`` telling the operator what
    to do."""

    cap = _make_capability(tmp_path)
    (tmp_path / ".colony").mkdir()
    (tmp_path / ".colony" / "repo_map.yaml").write_text(
        "schema_version: 3\n"
        "vcm_sources:\n"
        "  - { name: default, type: git_repo }\n",
        encoding="utf-8",
    )
    result = await cap.summarise_design_context()
    assert result["sources"] == []
    assert result["total_files"] == 0
    assert "No ``design_context_sources:``" in result["message"]


async def test_summarise_groups_files_per_source_with_heading_peeks(
    tmp_path: Path,
) -> None:
    cap = _make_capability(tmp_path)
    (tmp_path / ".colony").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "hypotheses").mkdir()
    (tmp_path / "docs" / "objectives.md").write_text(
        "# Objectives\n## Phase 1\nsome prose\n", encoding="utf-8",
    )
    (tmp_path / "docs" / "constraints.md").write_text(
        "# Constraints\n## Hard limits\n", encoding="utf-8",
    )
    (tmp_path / "hypotheses" / "h1.md").write_text(
        "# Hypothesis 1\nbody\n", encoding="utf-8",
    )
    (tmp_path / ".colony" / "repo_map.yaml").write_text(
        "schema_version: 3\n"
        "vcm_sources:\n"
        "  - { name: default, type: git_repo }\n"
        "design_context_sources:\n"
        "  - name: docs\n"
        "    paths: ['docs/**/*.md']\n"
        "    hint: 'mixed objectives + constraints'\n"
        "    pin_in_vcm: true\n"
        "  - name: hypos\n"
        "    paths: ['hypotheses/**/*.md']\n",
        encoding="utf-8",
    )

    result = await cap.summarise_design_context()
    by_name = {s["name"]: s for s in result["sources"]}
    assert set(by_name) == {"docs", "hypos"}
    assert result["total_files"] == 3

    docs = by_name["docs"]
    assert docs["file_count"] == 2
    assert docs["hint"] == "mixed objectives + constraints"
    assert docs["pin_in_vcm"] is True
    paths = sorted(f["path"] for f in docs["files"])
    assert paths == ["docs/constraints.md", "docs/objectives.md"]
    # Heading peek: both H1 + H2 are captured.
    objectives_file = next(f for f in docs["files"] if "objectives" in f["path"])
    assert objectives_file["headings"] == ["# Objectives", "## Phase 1"]

    hypos = by_name["hypos"]
    assert hypos["file_count"] == 1
    assert hypos["pin_in_vcm"] is False
    assert hypos["files"][0]["headings"] == ["# Hypothesis 1"]


async def test_summarise_filters_by_source_names(tmp_path: Path) -> None:
    cap = _make_capability(tmp_path)
    (tmp_path / ".colony").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "hypotheses").mkdir()
    (tmp_path / "docs" / "a.md").write_text("# a\n", encoding="utf-8")
    (tmp_path / "hypotheses" / "h.md").write_text("# h\n", encoding="utf-8")
    (tmp_path / ".colony" / "repo_map.yaml").write_text(
        "schema_version: 3\n"
        "vcm_sources:\n"
        "  - { name: default, type: git_repo }\n"
        "design_context_sources:\n"
        "  - name: docs\n"
        "    paths: ['docs/**/*.md']\n"
        "  - name: hypos\n"
        "    paths: ['hypotheses/**/*.md']\n",
        encoding="utf-8",
    )
    result = await cap.summarise_design_context(source_names=["docs"])
    assert [s["name"] for s in result["sources"]] == ["docs"]
    assert result["total_files"] == 1


async def test_summarise_caps_files_per_source(tmp_path: Path) -> None:
    """``max_files_per_source`` truncates the per-row file list and
    sets ``truncated_at`` so the planner can re-call with a narrower
    glob if it needs the long tail."""

    cap = _make_capability(tmp_path)
    (tmp_path / ".colony").mkdir()
    (tmp_path / "docs").mkdir()
    for i in range(10):
        (tmp_path / "docs" / f"f{i}.md").write_text(
            f"# file {i}\n", encoding="utf-8",
        )
    (tmp_path / ".colony" / "repo_map.yaml").write_text(
        "schema_version: 3\n"
        "vcm_sources:\n"
        "  - { name: default, type: git_repo }\n"
        "design_context_sources:\n"
        "  - name: docs\n"
        "    paths: ['docs/**/*.md']\n",
        encoding="utf-8",
    )
    result = await cap.summarise_design_context(max_files_per_source=3)
    assert result["sources"][0]["file_count"] == 10
    assert result["sources"][0]["truncated_at"] == 3
    assert len(result["sources"][0]["files"]) == 3


# ---------------------------------------------------------------------------
# search_design_context — raw / auto paths
# ---------------------------------------------------------------------------


def _seed_grep_fixture(tmp_path: Path) -> None:
    """Two design-context corpora with predictable hit locations."""

    (tmp_path / ".colony").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "hypotheses").mkdir()
    (tmp_path / "docs" / "objectives.md").write_text(
        "# Objectives\n"
        "Maximise sensitivity to magnetic fields.\n"
        "## Constraints\n"
        "Sensitivity ceiling is 1 fT/√Hz.\n",
        encoding="utf-8",
    )
    (tmp_path / "docs" / "constraints.md").write_text(
        "# Hard limits\n"
        "Operating temperature must be below 40 C.\n",
        encoding="utf-8",
    )
    (tmp_path / "hypotheses" / "h1.md").write_text(
        "# Hypothesis 1\n"
        "Sensitivity scales as 1/sqrt(N).\n",
        encoding="utf-8",
    )
    (tmp_path / ".colony" / "repo_map.yaml").write_text(
        "schema_version: 3\n"
        "vcm_sources:\n"
        "  - { name: default, type: git_repo }\n"
        "design_context_sources:\n"
        "  - name: docs\n"
        "    paths: ['docs/**/*.md']\n"
        "  - name: hypos\n"
        "    paths: ['hypotheses/**/*.md']\n",
        encoding="utf-8",
    )


async def test_search_raw_finds_matches_with_snippets(tmp_path: Path) -> None:
    cap = _make_capability(tmp_path)
    _seed_grep_fixture(tmp_path)
    result = await cap.search_design_context(query="sensitivity", path="raw")
    assert result["path_used"] == "raw"
    assert result["error"] == ""
    # 3 hits — 2 in objectives.md, 1 in hypotheses/h1.md.
    assert len(result["results"]) == 3
    files_hit = {r["file"] for r in result["results"]}
    assert files_hit == {"docs/objectives.md", "hypotheses/h1.md"}
    # Snippets carry surrounding lines for grounding.
    objectives_hits = [
        r for r in result["results"] if r["file"] == "docs/objectives.md"
    ]
    assert any("# Objectives" in r["snippet"] for r in objectives_hits)


async def test_search_raw_is_case_insensitive_by_default(tmp_path: Path) -> None:
    cap = _make_capability(tmp_path)
    _seed_grep_fixture(tmp_path)
    result = await cap.search_design_context(query="MAGNETIC", path="raw")
    assert len(result["results"]) == 1
    assert "magnetic" in result["results"][0]["snippet"]


async def test_search_raw_case_sensitive_respects_case(tmp_path: Path) -> None:
    cap = _make_capability(tmp_path)
    _seed_grep_fixture(tmp_path)
    result = await cap.search_design_context(
        query="MAGNETIC", path="raw", case_sensitive=True,
    )
    assert result["results"] == []
    assert result["truncated"] is False


async def test_search_raw_top_k_bounds_results(tmp_path: Path) -> None:
    cap = _make_capability(tmp_path)
    _seed_grep_fixture(tmp_path)
    result = await cap.search_design_context(
        query="sensitivity", path="raw", top_k=1,
    )
    assert len(result["results"]) == 1
    assert result["truncated"] is True


async def test_search_raw_filters_by_source_names(tmp_path: Path) -> None:
    cap = _make_capability(tmp_path)
    _seed_grep_fixture(tmp_path)
    result = await cap.search_design_context(
        query="sensitivity", path="raw", source_names=["hypos"],
    )
    assert len(result["results"]) == 1
    assert result["results"][0]["source_name"] == "hypos"


async def test_search_vcm_returns_not_yet_available(tmp_path: Path) -> None:
    """Path 'vcm' (VCM-paginated content search) is still a future
    enhancement — explicit error tells the planner to use 'auto' /
    'raw' until it lands."""

    cap = _make_capability(tmp_path)
    _seed_grep_fixture(tmp_path)
    result = await cap.search_design_context(query="x", path="vcm")
    assert result["path_used"] == "vcm"
    assert result["results"] == []
    assert "not yet wired" in result["error"]


# ---------------------------------------------------------------------------
# search_design_context — path='kuzu' (P3b) and auto-routing
# ---------------------------------------------------------------------------
#
# These tests populate an in-memory graph store with Claim instances
# whose ``citation.source_uri`` mirrors what materialize_design_context
# (P3a) would write (``design_context://<source_name>/<rel>``). They
# then exercise the kuzu path's filtering + auto-routing fallback.


async def _seed_kg_with_design_context_claims(tmp_path: Path) -> None:
    """Populate the process-singleton knowledge deps with an
    in-memory graph store carrying a handful of design-context
    claims. Tests use ``set_knowledge_deps(graph_store=…)`` so the
    capability's ``get_knowledge_deps().graph_store`` sees them."""

    from polymathera.colony.knowledge.deps import (
        reset_knowledge_deps, set_knowledge_deps,
    )
    from polymathera.colony.knowledge.models import (
        CitationSpan, Claim, KnowledgeFormat,
    )
    from polymathera.colony.knowledge.stores.graph import InMemoryGraphStore

    reset_knowledge_deps()
    store = InMemoryGraphStore()
    set_knowledge_deps(graph_store=store)

    def _claim(
        subject: str, predicate: str, object_: str,
        *, source_uri: str, confidence: float = 0.9,
    ) -> Claim:
        return Claim(
            subject=subject, predicate=predicate, object=object_,
            confidence=confidence,
            citation=CitationSpan(
                source_uri=source_uri,
                source_format=KnowledgeFormat.MARKDOWN,
                section_path="",
                char_start=0, char_end=10,
            ),
        )

    # Design-context claims (2 sources)
    await store.add_claim(_claim(
        "PMP-9", "is_a", "magnetometer",
        source_uri="design_context://hard-constraints/docs/constraints.md",
    ))
    await store.add_claim(_claim(
        "operating-temperature", "is_at_most", "40 C",
        source_uri="design_context://hard-constraints/docs/constraints.md",
    ))
    await store.add_claim(_claim(
        "sensitivity", "scales_as", "1/sqrt(N)",
        source_uri="design_context://hypotheses/hypotheses/h1.md",
    ))
    # Literature claim (different scheme — kuzu search MUST filter it out)
    await store.add_claim(_claim(
        "BGE-large", "is_a", "embedding model",
        source_uri="literature://allred-2002/foundational.pdf",
    ))


@pytest.fixture
def _reset_knowledge_deps_after():
    """Tests that mutate the knowledge-deps singleton restore the
    default afterwards so they don't bleed into other test files in
    the same process."""

    yield
    from polymathera.colony.knowledge.deps import reset_knowledge_deps

    reset_knowledge_deps()


async def test_search_kuzu_returns_claim_hits_filtered_by_design_context_uri(
    tmp_path: Path, _reset_knowledge_deps_after,
) -> None:
    """Path 'kuzu' returns subject/predicate/object hits drawn ONLY
    from edges whose citation_uri starts with ``design_context://``
    (literature / other corpora are correctly excluded)."""

    cap = _make_capability(tmp_path)
    _seed_grep_fixture(tmp_path)
    await _seed_kg_with_design_context_claims(tmp_path)

    result = await cap.search_design_context(
        query="magnetometer", path="kuzu",
    )
    assert result["path_used"] == "kuzu"
    assert result["error"] == ""
    assert len(result["results"]) == 1
    hit = result["results"][0]
    assert hit["subject"] == "PMP-9"
    assert hit["predicate"] == "is_a"
    assert hit["object"] == "magnetometer"
    assert hit["source_name"] == "hard-constraints"
    assert hit["file"] == "docs/constraints.md"
    assert hit["citation_uri"].startswith("design_context://")


async def test_search_kuzu_excludes_literature_corpus_claims(
    tmp_path: Path, _reset_knowledge_deps_after,
) -> None:
    """The literature claim (``BGE-large is_a embedding model``)
    matches the query text-wise but its citation_uri is the
    ``literature://`` scheme — must not surface in design-context
    kuzu search."""

    cap = _make_capability(tmp_path)
    _seed_grep_fixture(tmp_path)
    await _seed_kg_with_design_context_claims(tmp_path)

    result = await cap.search_design_context(query="BGE-large", path="kuzu")
    assert result["results"] == []


async def test_search_kuzu_filters_by_source_names(
    tmp_path: Path, _reset_knowledge_deps_after,
) -> None:
    cap = _make_capability(tmp_path)
    _seed_grep_fixture(tmp_path)
    await _seed_kg_with_design_context_claims(tmp_path)

    # "sensitivity" matches one claim in 'hypotheses'; filtering to
    # 'hard-constraints' should yield zero.
    result_constrained = await cap.search_design_context(
        query="sensitivity", path="kuzu", source_names=["hard-constraints"],
    )
    assert result_constrained["results"] == []
    result_all = await cap.search_design_context(
        query="sensitivity", path="kuzu",
    )
    assert len(result_all["results"]) == 1
    assert result_all["results"][0]["source_name"] == "hypotheses"


async def test_search_kuzu_top_k_bounds_results(
    tmp_path: Path, _reset_knowledge_deps_after,
) -> None:
    """``top_k`` caps the returned hits and sets ``truncated`` when
    the scan had more matches."""

    cap = _make_capability(tmp_path)
    _seed_grep_fixture(tmp_path)
    await _seed_kg_with_design_context_claims(tmp_path)

    # "_a" matches both ``is_a`` predicate (predicate match) and
    # is broad enough to hit multiple claims.
    result = await cap.search_design_context(
        query="is_a", path="kuzu", top_k=1,
    )
    assert len(result["results"]) == 1
    assert result["truncated"] is True


async def test_search_kuzu_empty_when_no_design_context_claims_ingested(
    tmp_path: Path, _reset_knowledge_deps_after,
) -> None:
    """Fresh deps (no path-1 ingestion run) → kuzu returns empty
    cleanly. No error — the planner is supposed to interpret an
    empty result as 'no KG hits' and fall through if it cares."""

    from polymathera.colony.knowledge.deps import (
        reset_knowledge_deps, set_knowledge_deps,
    )
    from polymathera.colony.knowledge.stores.graph import InMemoryGraphStore

    reset_knowledge_deps()
    set_knowledge_deps(graph_store=InMemoryGraphStore())

    cap = _make_capability(tmp_path)
    _seed_grep_fixture(tmp_path)
    result = await cap.search_design_context(
        query="anything", path="kuzu",
    )
    assert result["path_used"] == "kuzu"
    assert result["results"] == []
    assert result["error"] == ""


async def test_search_auto_returns_kuzu_hits_when_kg_has_matches(
    tmp_path: Path, _reset_knowledge_deps_after,
) -> None:
    """``path='auto'`` tries kuzu first; if hits exist, returns them
    (does NOT fall through to raw — the planner sees claims, not
    file-line snippets)."""

    cap = _make_capability(tmp_path)
    _seed_grep_fixture(tmp_path)
    await _seed_kg_with_design_context_claims(tmp_path)

    result = await cap.search_design_context(
        query="magnetometer", path="auto",
    )
    assert result["path_used"] == "kuzu"
    assert len(result["results"]) == 1
    # Shape is the claim-hit shape, not the file-line-snippet shape.
    assert "subject" in result["results"][0]
    assert "snippet" not in result["results"][0]


async def test_search_auto_falls_through_to_raw_when_kg_has_no_hits(
    tmp_path: Path, _reset_knowledge_deps_after,
) -> None:
    """``path='auto'`` falls through to ``raw`` when the KG query
    returns empty. ``path_used`` flips to ``'raw'`` so the planner
    knows where the answer came from."""

    cap = _make_capability(tmp_path)
    _seed_grep_fixture(tmp_path)
    await _seed_kg_with_design_context_claims(tmp_path)

    # "sensitivity" appears 3× in the grep fixture as raw text, but
    # only matches the hypotheses claim — make the query specific
    # enough to miss the KG entirely so we exercise the fallback.
    result = await cap.search_design_context(
        query="ceiling", path="auto",
    )
    assert result["path_used"] == "raw"
    # Raw grep finds "Sensitivity ceiling is 1 fT/√Hz." in
    # objectives.md per the seed.
    assert any("ceiling" in r["snippet"].lower() for r in result["results"])


async def test_search_auto_falls_through_to_raw_when_no_graph_store(
    tmp_path: Path, _reset_knowledge_deps_after,
) -> None:
    """Operator hasn't wired any graph store (or it raises on
    access) → auto-routing degrades gracefully to raw rather than
    surfacing the wiring error to the planner mid-search."""

    from polymathera.colony.knowledge.deps import (
        reset_knowledge_deps, set_knowledge_deps,
    )

    reset_knowledge_deps()
    # No graph_store passed — falls through to _default_graph_store
    # which returns InMemoryGraphStore (empty) per P3a wiring.
    set_knowledge_deps()

    cap = _make_capability(tmp_path)
    _seed_grep_fixture(tmp_path)
    result = await cap.search_design_context(
        query="sensitivity", path="auto",
    )
    # KG empty → falls through to raw → real grep hits.
    assert result["path_used"] == "raw"
    assert len(result["results"]) >= 1


async def test_search_empty_query_short_circuits(tmp_path: Path) -> None:
    cap = _make_capability(tmp_path)
    _seed_grep_fixture(tmp_path)
    result = await cap.search_design_context(query="", path="raw")
    assert result["results"] == []
    assert result["error"] == "empty query"


async def test_search_no_design_context_sources_returns_clear_error(
    tmp_path: Path,
) -> None:
    cap = _make_capability(tmp_path)
    (tmp_path / ".colony").mkdir()
    (tmp_path / ".colony" / "repo_map.yaml").write_text(
        "schema_version: 3\n"
        "vcm_sources:\n"
        "  - { name: default, type: git_repo }\n",
        encoding="utf-8",
    )
    result = await cap.search_design_context(query="x", path="raw")
    assert result["results"] == []
    assert "design_context_sources" in result["error"]


# ---------------------------------------------------------------------------
# find_inconsistencies — P3c
# ---------------------------------------------------------------------------


async def _seed_kg_with_contradictions_and_rules(tmp_path: Path) -> None:
    """KG fixture with a mix of explicit contradiction claims +
    consistency_rule claims (in both predicate-tagged and ``is_a``
    idiom forms) + non-design-context literature claims that
    contradict (must be excluded from results)."""

    from polymathera.colony.knowledge.deps import (
        reset_knowledge_deps, set_knowledge_deps,
    )
    from polymathera.colony.knowledge.models import (
        CitationSpan, Claim, KnowledgeFormat,
    )
    from polymathera.colony.knowledge.stores.graph import InMemoryGraphStore

    reset_knowledge_deps()
    store = InMemoryGraphStore()
    set_knowledge_deps(graph_store=store)

    def _claim(s, p, o, *, source_uri, confidence=0.9):
        return Claim(
            subject=s, predicate=p, object=o, confidence=confidence,
            citation=CitationSpan(
                source_uri=source_uri,
                source_format=KnowledgeFormat.MARKDOWN,
                section_path="", char_start=0, char_end=10,
            ),
        )

    # Contradiction in design context — must be surfaced.
    await store.add_claim(_claim(
        "decision-D-04", "contradicts", "constraint-C-11",
        source_uri="design_context://constraints/docs/contradictions.md",
    ))
    await store.add_claim(_claim(
        "design-alt-A", "conflicts_with", "design-alt-B",
        source_uri="design_context://alternatives/docs/alternatives.md",
    ))

    # Operator-authored consistency rules (two idioms).
    await store.add_claim(_claim(
        "rule:every-constraint-referenced",
        "defines_consistency_rule",
        "A constraint with no decision referencing it is an orphan.",
        source_uri="design_context://rules/docs/rules.md",
    ))
    await store.add_claim(_claim(
        "rule:contradictory-decisions",
        "is_a", "consistency rule",  # ``X is_a 'consistency rule'`` idiom
        source_uri="design_context://rules/docs/rules.md",
    ))

    # Non-contradiction, non-rule design-context claim — must NOT
    # appear in either list.
    await store.add_claim(_claim(
        "PMP-9", "is_a", "magnetometer",
        source_uri="design_context://hard-constraints/docs/constraints.md",
    ))

    # Literature contradiction — different scheme; must be excluded.
    await store.add_claim(_claim(
        "paper-A", "contradicts", "paper-B",
        source_uri="literature://review-2024/paper-a.pdf",
    ))


async def test_find_inconsistencies_surfaces_contradiction_claims(
    tmp_path: Path, _reset_knowledge_deps_after,
) -> None:
    cap = _make_capability(tmp_path)
    await _seed_kg_with_contradictions_and_rules(tmp_path)

    result = await cap.find_inconsistencies(
        emit_blackboard_events=False,
    )
    assert result["error"] == ""
    # Two contradictions, both from design-context URIs.
    contradictions = result["contradictions"]
    assert len(contradictions) == 2
    preds = {c["predicate"] for c in contradictions}
    assert preds == {"contradicts", "conflicts_with"}
    # Literature contradiction excluded.
    for c in contradictions:
        assert c["citation_uri"].startswith("design_context://")


async def test_find_inconsistencies_discovers_rule_claims_both_idioms(
    tmp_path: Path, _reset_knowledge_deps_after,
) -> None:
    """Both the ``defines_consistency_rule`` predicate AND the
    ``X is_a 'consistency rule'`` idiom surface as discovered rules."""

    cap = _make_capability(tmp_path)
    await _seed_kg_with_contradictions_and_rules(tmp_path)

    result = await cap.find_inconsistencies(
        emit_blackboard_events=False,
    )
    rule_subjects = {r["rule_id"] for r in result["rules_discovered"]}
    assert rule_subjects == {
        "rule:every-constraint-referenced",
        "rule:contradictory-decisions",
    }


async def test_find_inconsistencies_filters_by_source_names(
    tmp_path: Path, _reset_knowledge_deps_after,
) -> None:
    cap = _make_capability(tmp_path)
    await _seed_kg_with_contradictions_and_rules(tmp_path)

    result = await cap.find_inconsistencies(
        source_names=["constraints"],
        emit_blackboard_events=False,
    )
    assert len(result["contradictions"]) == 1
    assert result["contradictions"][0]["source_name"] == "constraints"
    # Rules live in a different source — filtered out.
    assert result["rules_discovered"] == []


async def test_find_inconsistencies_emits_blackboard_event_per_contradiction(
    tmp_path: Path, _reset_knowledge_deps_after,
) -> None:
    """One DesignInconsistencyProtocol event per contradiction with
    the right key shape + tag set."""

    from unittest.mock import AsyncMock, MagicMock

    cap = _make_capability(tmp_path)
    await _seed_kg_with_contradictions_and_rules(tmp_path)
    fake_bb = MagicMock()
    fake_bb.write = AsyncMock()
    cap._colony_blackboard = fake_bb

    result = await cap.find_inconsistencies(emit_blackboard_events=True)
    assert len(result["contradictions"]) == 2
    assert fake_bb.write.await_count == 2

    keys = [c.kwargs["key"] for c in fake_bb.write.await_args_list]
    assert all(k.startswith("design_inconsistency:") for k in keys)
    assert all(":contradiction:" in k for k in keys)
    for call in fake_bb.write.await_args_list:
        assert "contradiction" in call.kwargs["tags"]
        assert "design_context" in call.kwargs["tags"]
        # Payload mirrors the finding + adds detected_at.
        assert "detected_at" in call.kwargs["value"]
        assert call.kwargs["value"]["kind"] == "contradiction"


async def test_find_inconsistencies_empty_kg_returns_clean(
    tmp_path: Path, _reset_knowledge_deps_after,
) -> None:
    """No claims in the KG → empty result + no events, no error."""

    from unittest.mock import AsyncMock, MagicMock
    from polymathera.colony.knowledge.deps import (
        reset_knowledge_deps, set_knowledge_deps,
    )
    from polymathera.colony.knowledge.stores.graph import InMemoryGraphStore

    reset_knowledge_deps()
    set_knowledge_deps(graph_store=InMemoryGraphStore())

    cap = _make_capability(tmp_path)
    fake_bb = MagicMock()
    fake_bb.write = AsyncMock()
    cap._colony_blackboard = fake_bb

    result = await cap.find_inconsistencies()
    assert result == {
        "contradictions": [],
        "rules_discovered": [],
        "stats": {"scanned_claims": 0, "scan_cap_hit": False},
        "error": "",
    }
    fake_bb.write.assert_not_awaited()


async def test_find_inconsistencies_custom_contradict_predicates(
    tmp_path: Path, _reset_knowledge_deps_after,
) -> None:
    """Operator can override the default predicate set; the action
    matches against ONLY the supplied predicates."""

    cap = _make_capability(tmp_path)
    await _seed_kg_with_contradictions_and_rules(tmp_path)

    result = await cap.find_inconsistencies(
        contradict_predicates=["contradicts"],  # excludes 'conflicts_with'
        emit_blackboard_events=False,
    )
    assert len(result["contradictions"]) == 1
    assert result["contradictions"][0]["predicate"] == "contradicts"


# ---------------------------------------------------------------------------
# audit_hypothesis_coverage — P3c
# ---------------------------------------------------------------------------


async def _seed_kg_with_hypotheses(tmp_path: Path) -> None:
    """KG with hypotheses in both idioms + a mix of covered/uncovered
    + a literature hypothesis (must be excluded) + a coverage claim
    pointing at a non-hypothesis subject (must be ignored)."""

    from polymathera.colony.knowledge.deps import (
        reset_knowledge_deps, set_knowledge_deps,
    )
    from polymathera.colony.knowledge.models import (
        CitationSpan, Claim, KnowledgeFormat,
    )
    from polymathera.colony.knowledge.stores.graph import InMemoryGraphStore

    reset_knowledge_deps()
    store = InMemoryGraphStore()
    set_knowledge_deps(graph_store=store)

    def _claim(s, p, o, *, source_uri, confidence=0.9):
        return Claim(
            subject=s, predicate=p, object=o, confidence=confidence,
            citation=CitationSpan(
                source_uri=source_uri,
                source_format=KnowledgeFormat.MARKDOWN,
                section_path="", char_start=0, char_end=10,
            ),
        )

    # Two hypotheses in different idioms.
    await store.add_claim(_claim(
        "H-01-sensitivity-scales", "hypothesizes",
        "sensitivity scales as 1/sqrt(N)",
        source_uri="design_context://hypotheses/h1.md",
    ))
    await store.add_claim(_claim(
        "H-02-shielding-factor", "is_a", "hypothesis",
        source_uri="design_context://hypotheses/h2.md",
    ))
    # Third hypothesis: orphan.
    await store.add_claim(_claim(
        "H-03-orphan", "hypothesizes",
        "this orphan has no coverage",
        source_uri="design_context://hypotheses/h3.md",
    ))

    # Coverage for H-01 (object form) + H-02 (object form).
    await store.add_claim(_claim(
        "Test-7", "verifies", "H-01-sensitivity-scales",
        source_uri="design_context://hypotheses/tests.md",
    ))
    await store.add_claim(_claim(
        "experiment-shield", "falsifies", "H-02-shielding-factor",
        source_uri="design_context://hypotheses/tests.md",
    ))

    # Coverage claim pointing at a NON-hypothesis subject — must not
    # accidentally make that subject look hypothesis-covered.
    await store.add_claim(_claim(
        "random-test", "verifies", "some-other-subject",
        source_uri="design_context://hypotheses/tests.md",
    ))

    # Literature hypothesis — must be excluded.
    await store.add_claim(_claim(
        "paper-2024-hypothesis", "hypothesizes",
        "irrelevant",
        source_uri="literature://review-2024/paper.pdf",
    ))


async def test_audit_hypothesis_coverage_lists_all_design_context_hypotheses(
    tmp_path: Path, _reset_knowledge_deps_after,
) -> None:
    cap = _make_capability(tmp_path)
    await _seed_kg_with_hypotheses(tmp_path)

    result = await cap.audit_hypothesis_coverage(
        emit_blackboard_events=False,
    )
    assert result["error"] == ""
    subjects = {h["subject"] for h in result["hypotheses"]}
    assert subjects == {
        "H-01-sensitivity-scales",
        "H-02-shielding-factor",
        "H-03-orphan",
    }
    # Literature hypothesis excluded.
    assert "paper-2024-hypothesis" not in subjects


async def test_audit_hypothesis_coverage_identifies_orphans(
    tmp_path: Path, _reset_knowledge_deps_after,
) -> None:
    """An orphan is a hypothesis with no verify/falsify coverage."""

    cap = _make_capability(tmp_path)
    await _seed_kg_with_hypotheses(tmp_path)

    result = await cap.audit_hypothesis_coverage(
        emit_blackboard_events=False,
    )
    orphan_subjects = {o["subject"] for o in result["orphans"]}
    assert orphan_subjects == {"H-03-orphan"}
    # The two covered hypotheses report their coverage entries.
    by_subj = {h["subject"]: h for h in result["hypotheses"]}
    assert by_subj["H-01-sensitivity-scales"]["coverage_count"] == 1
    assert by_subj["H-01-sensitivity-scales"]["coverage"][0]["predicate"] == "verifies"
    assert by_subj["H-02-shielding-factor"]["coverage_count"] == 1
    assert by_subj["H-02-shielding-factor"]["coverage"][0]["predicate"] == "falsifies"
    assert by_subj["H-03-orphan"]["coverage_count"] == 0


async def test_audit_hypothesis_coverage_emits_blackboard_event_per_orphan(
    tmp_path: Path, _reset_knowledge_deps_after,
) -> None:
    from unittest.mock import AsyncMock, MagicMock

    cap = _make_capability(tmp_path)
    await _seed_kg_with_hypotheses(tmp_path)
    fake_bb = MagicMock()
    fake_bb.write = AsyncMock()
    cap._colony_blackboard = fake_bb

    result = await cap.audit_hypothesis_coverage(
        emit_blackboard_events=True,
    )
    assert len(result["orphans"]) == 1
    assert fake_bb.write.await_count == 1
    call = fake_bb.write.await_args
    assert call.kwargs["key"].startswith("design_suggestion:")
    assert ":hypothesis_orphan:" in call.kwargs["key"]
    assert "hypothesis_orphan" in call.kwargs["tags"]
    assert call.kwargs["value"]["kind"] == "hypothesis_orphan"
    assert call.kwargs["value"]["target_claim_type"] == "hypothesis"
    assert "H-03-orphan" in call.kwargs["value"]["summary"]


async def test_audit_hypothesis_coverage_filters_by_source_names(
    tmp_path: Path, _reset_knowledge_deps_after,
) -> None:
    """``source_names`` restricts both the hypothesis-scan AND
    the coverage-scan to claims from those rows."""

    cap = _make_capability(tmp_path)
    await _seed_kg_with_hypotheses(tmp_path)

    result = await cap.audit_hypothesis_coverage(
        source_names=["hypotheses"],
        emit_blackboard_events=False,
    )
    # All 3 hypotheses ARE in row 'hypotheses' — same row also
    # holds the coverage claims, so nothing changes vs unfiltered.
    assert len(result["hypotheses"]) == 3


async def test_audit_hypothesis_coverage_empty_kg_returns_clean(
    tmp_path: Path, _reset_knowledge_deps_after,
) -> None:
    from polymathera.colony.knowledge.deps import (
        reset_knowledge_deps, set_knowledge_deps,
    )
    from polymathera.colony.knowledge.stores.graph import InMemoryGraphStore

    reset_knowledge_deps()
    set_knowledge_deps(graph_store=InMemoryGraphStore())

    cap = _make_capability(tmp_path)
    result = await cap.audit_hypothesis_coverage()
    assert result["hypotheses"] == []
    assert result["orphans"] == []
    assert result["stats"]["hypothesis_count"] == 0
    assert result["error"] == ""


# ---------------------------------------------------------------------------
# Protocol key round-trips (P3c additions)
# ---------------------------------------------------------------------------


def test_design_inconsistency_protocol_round_trip() -> None:
    from polymathera.colony.agents.blackboard.protocol import (
        DesignInconsistencyProtocol,
    )

    key = DesignInconsistencyProtocol.event_key(
        source_name="hard-constraints", kind="contradiction",
        millis=1700000000123,
    )
    assert key.startswith("design_inconsistency:")
    parsed = DesignInconsistencyProtocol.parse_event_key(key)
    assert parsed == {
        "source_name": "hard-constraints",
        "kind": "contradiction",
        "millis": "1700000000123",
    }


def test_design_inconsistency_protocol_rejects_alien_key() -> None:
    from polymathera.colony.agents.blackboard.protocol import (
        DesignInconsistencyProtocol,
    )

    with pytest.raises(ValueError, match="Not a DesignInconsistency"):
        DesignInconsistencyProtocol.parse_event_key("monorepo_commit:foo:bar")


def test_design_inconsistency_protocol_patterns() -> None:
    from polymathera.colony.agents.blackboard.protocol import (
        DesignInconsistencyProtocol,
    )

    assert DesignInconsistencyProtocol.event_pattern() == (
        "design_inconsistency:*"
    )
    assert DesignInconsistencyProtocol.event_pattern_for_source("foo") == (
        "design_inconsistency:foo:*"
    )
    assert DesignInconsistencyProtocol.event_pattern_for_kind(
        "contradiction",
    ) == "design_inconsistency:*:contradiction:*"


def test_design_suggestion_protocol_round_trip() -> None:
    from polymathera.colony.agents.blackboard.protocol import (
        DesignSuggestionProtocol,
    )

    key = DesignSuggestionProtocol.event_key(
        source_name="hypotheses", kind="hypothesis_orphan",
        millis=42,
    )
    assert key.startswith("design_suggestion:")
    parsed = DesignSuggestionProtocol.parse_event_key(key)
    assert parsed == {
        "source_name": "hypotheses",
        "kind": "hypothesis_orphan",
        "millis": "42",
    }


def test_design_suggestion_protocol_rejects_alien_key() -> None:
    from polymathera.colony.agents.blackboard.protocol import (
        DesignSuggestionProtocol,
    )

    with pytest.raises(ValueError, match="Not a DesignSuggestion"):
        DesignSuggestionProtocol.parse_event_key("design_inconsistency:foo:c:1")
