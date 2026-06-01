"""Tests for :mod:`polymathera.colony.design_monorepo.repo_map`.

The materialiser (`design_monorepo.materialize`) needs a live VCM and
``polymathera`` system; that path is exercised by the CLI integration
test. Here we test the offline-only pieces: schema parsing, default
fallback, kwargs translation, and submodule resolution against a
hand-built fixture repo.
"""

from __future__ import annotations

from pathlib import Path

import git
import pytest

from polymathera.colony.design_monorepo.repo_map import (
    AcquirerSpec,
    DesignContextSource,
    KnowledgeSource,
    REPO_MAP_DIR,
    REPO_MAP_FILENAME,
    RepoMap,
    SCHEMA_VERSION,
    VcmSource,
)
from polymathera.colony.knowledge.models import CorpusTier


def _init_repo(root: Path) -> git.Repo:
    repo = git.Repo.init(root, initial_branch="main")
    repo.config_writer().set_value("user", "email", "t@t").release()
    repo.config_writer().set_value("user", "name", "t").release()
    return repo


def _commit_initial(repo: git.Repo, files: dict[str, bytes]) -> None:
    root = Path(repo.working_dir)
    for rel, data in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    repo.git.add(all=True)
    repo.index.commit("initial")


def test_default_for_unmapped_repo_returns_single_git_repo_source() -> None:
    rm = RepoMap.default_for_unmapped_repo()
    assert len(rm.vcm_sources) == 1
    assert rm.vcm_sources[0].name == "default"
    assert rm.vcm_sources[0].type == "git_repo"


def test_load_falls_back_to_default_when_file_absent(tmp_path: Path) -> None:
    repo_root = tmp_path / "r"
    repo_root.mkdir()
    rm = RepoMap.load(repo_root)
    assert len(rm.vcm_sources) == 1
    assert rm.vcm_sources[0].name == "default"


def test_load_parses_a_real_repo_map(tmp_path: Path) -> None:
    repo_root = tmp_path / "r"
    (repo_root / REPO_MAP_DIR).mkdir(parents=True)
    (repo_root / REPO_MAP_DIR / REPO_MAP_FILENAME).write_text(
        "schema_version: 3\n"
        "vcm_sources:\n"
        "  - name: code\n"
        "    type: git_repo\n"
        "    start_dir: tools/\n"
        "    exclude_globs: ['**/build/**']\n"
        "  - name: literature\n"
        "    type: literature\n"
        "    start_dir: literature/\n"
        "    chunk_target_tokens: 600\n",
        encoding="utf-8",
    )
    rm = RepoMap.load(repo_root)
    assert [s.name for s in rm.vcm_sources] == ["code", "literature"]
    assert rm.vcm_sources[0].start_dir == "tools/"
    assert rm.vcm_sources[0].exclude_globs == ["**/build/**"]
    assert rm.vcm_sources[1].chunk_target_tokens == 600


def test_load_rejects_unknown_schema_version(tmp_path: Path) -> None:
    repo_root = tmp_path / "r"
    (repo_root / REPO_MAP_DIR).mkdir(parents=True)
    (repo_root / REPO_MAP_DIR / REPO_MAP_FILENAME).write_text(
        "schema_version: 99\nvcm_sources: []\n", encoding="utf-8",
    )
    with pytest.raises(ValueError, match="schema_version"):
        RepoMap.load(repo_root)


def test_to_mmap_kwargs_passes_through_origin_url(tmp_path: Path) -> None:
    spec = VcmSource(
        name="code",
        type="git_repo",
        start_dir="tools/",
        exclude_globs=["**/build/**"],
        binary_policy="skip",
    )
    kwargs = spec.to_mmap_kwargs(
        repo_root=tmp_path,
        scope_id="repo:code",
        fallback_origin_url="https://x.test/repo.git",
        fallback_branch="main",
        fallback_commit="HEAD",
    )
    assert kwargs["scope_id"] == "repo:code"
    assert kwargs["source_type"] == "git_repo"
    assert kwargs["origin_url"] == "https://x.test/repo.git"
    assert kwargs["start_dir"] == "tools/"
    assert kwargs["exclude_globs"] == ["**/build/**"]
    assert kwargs["binary_policy"] == "skip"


def test_to_mmap_kwargs_resolves_submodule(tmp_path: Path) -> None:
    # Outer + submodule repos.
    outer_root = tmp_path / "outer"
    sub_root = tmp_path / "sub"
    outer_root.mkdir()
    sub_root.mkdir()

    sub_repo = _init_repo(sub_root)
    _commit_initial(sub_repo, {"a.py": b"x = 1\n"})
    sub_sha = sub_repo.head.commit.hexsha

    outer_repo = _init_repo(outer_root)
    _commit_initial(outer_repo, {"README.md": b"# outer\n"})
    # Add submodule by writing .gitmodules + ls-tree gitlink. We avoid
    # the actual ``git submodule add`` to keep the fixture self-contained.
    (outer_root / ".gitmodules").write_text(
        "[submodule \"third_party/foo\"]\n"
        "    path = third_party/foo\n"
        f"    url = file://{sub_root}\n",
        encoding="utf-8",
    )
    # Stage a gitlink to the submodule's HEAD.
    outer_repo.git.update_index(
        "--add", "--cacheinfo", f"160000,{sub_sha},third_party/foo",
    )
    outer_repo.index.commit("add submodule")

    spec = VcmSource(name="external-foo", type="git_repo", submodule="third_party/foo")
    kwargs = spec.to_mmap_kwargs(
        repo_root=outer_root,
        scope_id="repo:external-foo",
        fallback_origin_url="ignored",
        fallback_branch="ignored",
        fallback_commit="ignored",
    )
    assert kwargs["origin_url"] == f"file://{sub_root}"
    assert kwargs["commit"] == sub_sha


def test_to_mmap_kwargs_rejects_origin_and_submodule_both_set(tmp_path: Path) -> None:
    spec = VcmSource(
        name="bad",
        type="git_repo",
        origin_url="https://x.test/repo.git",
        submodule="third_party/foo",
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        spec.to_mmap_kwargs(
            repo_root=tmp_path,
            scope_id="x",
            fallback_origin_url="x",
            fallback_branch="main",
            fallback_commit="HEAD",
        )


def test_default_repo_map_has_empty_knowledge_sources() -> None:
    rm = RepoMap.default_for_unmapped_repo()
    assert rm.knowledge_sources == []


def test_load_parses_knowledge_sources_block(tmp_path: Path) -> None:
    repo_root = tmp_path / "r"
    (repo_root / REPO_MAP_DIR).mkdir(parents=True)
    (repo_root / REPO_MAP_DIR / REPO_MAP_FILENAME).write_text(
        "schema_version: 3\n"
        "vcm_sources:\n"
        "  - name: code\n"
        "    type: git_repo\n"
        "knowledge_sources:\n"
        "  - name: curated\n"
        "    paths: ['literature/curated/**/*.pdf']\n"
        "    profile: scientific_paper\n"
        "  - name: standards\n"
        "    paths: ['standards/**/*.pdf']\n",
        encoding="utf-8",
    )
    rm = RepoMap.load(repo_root)
    assert len(rm.knowledge_sources) == 2
    assert rm.knowledge_sources[0].name == "curated"
    assert rm.knowledge_sources[0].paths == ["literature/curated/**/*.pdf"]
    assert rm.knowledge_sources[0].profile == "scientific_paper"
    assert rm.knowledge_sources[1].name == "standards"
    assert rm.knowledge_sources[1].profile is None


def test_knowledge_source_requires_name(tmp_path: Path) -> None:
    repo_root = tmp_path / "r"
    (repo_root / REPO_MAP_DIR).mkdir(parents=True)
    (repo_root / REPO_MAP_DIR / REPO_MAP_FILENAME).write_text(
        "schema_version: 3\n"
        "vcm_sources: []\n"
        "knowledge_sources:\n"
        "  - paths: ['x/*']\n",  # missing required ``name``
        encoding="utf-8",
    )
    with pytest.raises(Exception):
        RepoMap.load(repo_root)


def test_per_source_paging_overrides_round_trip(tmp_path: Path) -> None:
    """``flush_threshold`` / ``flush_token_budget`` / ``pinned`` are
    optional per-source overrides. ``to_mmap_config_overrides`` returns
    only the fields the user actually set so the materialiser can
    layer them onto a base ``MmapConfig`` without clobbering defaults
    on rows that don't override.
    """

    none_set = VcmSource(name="bare", type="git_repo")
    assert none_set.to_mmap_config_overrides() == {}

    partial = VcmSource(
        name="code", type="git_repo", flush_threshold=8,
    )
    assert partial.to_mmap_config_overrides() == {"flush_threshold": 8}

    full = VcmSource(
        name="lit", type="literature",
        flush_threshold=2, flush_token_budget=512, pinned=True,
    )
    assert full.to_mmap_config_overrides() == {
        "flush_threshold": 2,
        "flush_token_budget": 512,
        "pinned": True,
    }


def test_load_parses_per_source_paging_block(tmp_path: Path) -> None:
    repo_root = tmp_path / "r"
    (repo_root / REPO_MAP_DIR).mkdir(parents=True)
    (repo_root / REPO_MAP_DIR / REPO_MAP_FILENAME).write_text(
        "schema_version: 3\n"
        "vcm_sources:\n"
        "  - name: code\n"
        "    type: git_repo\n"
        "    flush_threshold: 50\n"
        "    flush_token_budget: 8192\n"
        "    pinned: true\n",
        encoding="utf-8",
    )
    rm = RepoMap.load(repo_root)
    spec = rm.vcm_sources[0]
    assert spec.flush_threshold == 50
    assert spec.flush_token_budget == 8192
    assert spec.pinned is True


def test_knowledge_source_matches_glob() -> None:
    src = KnowledgeSource(name="k", paths=["lit/**/*.pdf"])
    assert src.matches("lit/curated/a.pdf")
    assert src.matches("lit/promoted/sub/b.pdf")
    assert not src.matches("standards/x.pdf")


# ---- Schema v2: acquirer-shaped knowledge_sources -------------------


def test_knowledge_source_paths_xor_acquirer() -> None:
    """Exactly one of ``paths`` / ``acquirer`` must be set."""

    with pytest.raises(ValueError, match="exactly one"):
        KnowledgeSource(name="bad-neither")

    with pytest.raises(ValueError, match="exactly one"):
        KnowledgeSource(
            name="bad-both",
            paths=["x/*"],
            acquirer=AcquirerSpec(method="doi", args={}),
            destination="x/",
        )


def test_knowledge_source_acquirer_requires_destination() -> None:
    with pytest.raises(ValueError, match="destination"):
        KnowledgeSource(
            name="bad",
            acquirer=AcquirerSpec(method="doi", args={"doi": "x"}),
        )


def test_knowledge_source_destination_forbidden_without_acquirer() -> None:
    with pytest.raises(ValueError, match="destination"):
        KnowledgeSource(
            name="bad", paths=["x/*"], destination="elsewhere/",
        )


def test_knowledge_source_destination_must_be_relative() -> None:
    with pytest.raises(ValueError, match="relative"):
        KnowledgeSource(
            name="bad",
            acquirer=AcquirerSpec(method="doi", args={}),
            destination="/abs/path",
        )


def test_load_parses_acquirer_row(tmp_path: Path) -> None:
    repo_root = tmp_path / "r"
    (repo_root / REPO_MAP_DIR).mkdir(parents=True)
    (repo_root / REPO_MAP_DIR / REPO_MAP_FILENAME).write_text(
        "schema_version: 3\n"
        "vcm_sources: []\n"
        "knowledge_sources:\n"
        "  - name: allred_2002\n"
        "    acquirer:\n"
        "      method: arxiv_id\n"
        "      args:\n"
        "        arxiv_id: 'physics/0205063'\n"
        "    destination: kb/literature/atomic-physics/\n"
        "    profile: scientific_paper\n"
        "    tier: tier_3_research\n",
        encoding="utf-8",
    )
    rm = RepoMap.load(repo_root)
    row = rm.knowledge_sources[0]
    assert row.acquirer is not None
    assert row.acquirer.method == "arxiv_id"
    assert row.acquirer.args == {"arxiv_id": "physics/0205063"}
    assert row.destination == "kb/literature/atomic-physics/"
    assert row.tier is CorpusTier.TIER_3_RESEARCH
    assert row.paths is None


def test_load_rejects_v1_schema(tmp_path: Path) -> None:
    """Per user answer A — no backward compatibility for v1."""

    repo_root = tmp_path / "r"
    (repo_root / REPO_MAP_DIR).mkdir(parents=True)
    (repo_root / REPO_MAP_DIR / REPO_MAP_FILENAME).write_text(
        "schema_version: 1\nvcm_sources: []\n", encoding="utf-8",
    )
    with pytest.raises(ValueError, match="schema_version"):
        RepoMap.load(repo_root)


def test_load_rejects_v2_schema(tmp_path: Path) -> None:
    """Schema v3 (design-context section) is the current version. v2
    is the previous one; operators must bump explicitly (the
    materialiser-side schema upgrade tool is out of scope here)."""

    repo_root = tmp_path / "r"
    (repo_root / REPO_MAP_DIR).mkdir(parents=True)
    (repo_root / REPO_MAP_DIR / REPO_MAP_FILENAME).write_text(
        "schema_version: 2\nvcm_sources: []\n", encoding="utf-8",
    )
    with pytest.raises(ValueError, match="schema_version"):
        RepoMap.load(repo_root)


def test_current_schema_version_is_3() -> None:
    """Locks the constant — bumping it again should force a conscious
    test update + design-doc note, not a silent change."""

    assert SCHEMA_VERSION == 3


# ---------------------------------------------------------------------------
# design_context_sources — schema v3 addition
# ---------------------------------------------------------------------------


def test_design_context_source_defaults() -> None:
    src = DesignContextSource(name="dc", paths=["docs/**/*.md"])
    assert src.hint is None
    assert src.pin_in_vcm is False
    assert src.pin_lock_duration_days == 7


def test_design_context_source_requires_non_empty_paths() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        DesignContextSource(name="dc", paths=[])


def test_design_context_source_pin_lock_duration_must_be_positive() -> None:
    with pytest.raises(ValueError):
        DesignContextSource(
            name="dc", paths=["x"], pin_lock_duration_days=0,
        )


def test_design_context_source_matches_glob() -> None:
    src = DesignContextSource(name="dc", paths=["docs/**/*.md"])
    assert src.matches("docs/objectives.md") is True
    assert src.matches("docs/sub/dir/decisions.md") is True
    assert src.matches("src/code.py") is False


def test_design_context_source_extra_fields_rejected() -> None:
    """The model uses ``extra='forbid'`` — typos must raise."""

    with pytest.raises(ValueError):
        DesignContextSource(
            name="dc", paths=["x"], role="objectives",  # type: ignore[call-arg]
        )


def test_repo_map_default_has_empty_design_context_sources() -> None:
    rm = RepoMap.default_for_unmapped_repo()
    assert rm.design_context_sources == []


def test_repo_map_rejects_duplicate_design_context_source_names() -> None:
    with pytest.raises(ValueError, match="duplicate design_context_sources"):
        RepoMap(
            vcm_sources=[VcmSource(name="default", type="git_repo")],
            design_context_sources=[
                DesignContextSource(name="dup", paths=["a"]),
                DesignContextSource(name="dup", paths=["b"]),
            ],
        )


def test_load_parses_design_context_sources_block(tmp_path: Path) -> None:
    repo_root = tmp_path / "r"
    (repo_root / REPO_MAP_DIR).mkdir(parents=True)
    (repo_root / REPO_MAP_DIR / REPO_MAP_FILENAME).write_text(
        "schema_version: 3\n"
        "vcm_sources:\n"
        "  - name: default\n"
        "    type: git_repo\n"
        "design_context_sources:\n"
        "  - name: system-design-docs\n"
        "    paths:\n"
        "      - 'kb/design/**/*.md'\n"
        "      - 'docs/**/*.md'\n"
        "    hint: 'mixed objectives + constraints + decisions'\n"
        "    pin_in_vcm: true\n"
        "    pin_lock_duration_days: 5\n"
        "  - name: hypotheses\n"
        "    paths: ['**/hypotheses/*.md']\n",
        encoding="utf-8",
    )
    rm = RepoMap.load(repo_root)
    assert [s.name for s in rm.design_context_sources] == [
        "system-design-docs", "hypotheses",
    ]
    pinned = rm.design_context_sources[0]
    assert pinned.pin_in_vcm is True
    assert pinned.pin_lock_duration_days == 5
    assert pinned.hint is not None and "mixed" in pinned.hint
    unpinned = rm.design_context_sources[1]
    assert unpinned.pin_in_vcm is False
    assert unpinned.pin_lock_duration_days == 7  # default
    assert unpinned.hint is None


def test_load_treats_design_context_sources_as_optional(tmp_path: Path) -> None:
    """A v3 file with no design_context_sources block is legal — the
    field defaults to []."""

    repo_root = tmp_path / "r"
    (repo_root / REPO_MAP_DIR).mkdir(parents=True)
    (repo_root / REPO_MAP_DIR / REPO_MAP_FILENAME).write_text(
        "schema_version: 3\n"
        "vcm_sources:\n"
        "  - name: default\n"
        "    type: git_repo\n",
        encoding="utf-8",
    )
    rm = RepoMap.load(repo_root)
    assert rm.design_context_sources == []


# ---------------------------------------------------------------------------
# DesignContextMappedProtocol — key/parse round-trip
# ---------------------------------------------------------------------------


def test_design_context_mapped_protocol_round_trip() -> None:
    from polymathera.colony.agents.blackboard.protocol import (
        DesignContextMappedProtocol,
    )

    key = DesignContextMappedProtocol.event_key(
        source_name="hard-constraints", path="vcm", millis=1700000000123,
    )
    assert key.startswith("design_context_mapped:")
    parsed = DesignContextMappedProtocol.parse_event_key(key)
    assert parsed == {
        "source_name": "hard-constraints",
        "path": "vcm",
        "millis": "1700000000123",
    }


def test_design_context_mapped_protocol_round_trip_with_colon_in_name() -> None:
    """Source names with ``:`` collide with the blackboard separator
    so the protocol substitutes ``/`` and reverses on parse."""

    from polymathera.colony.agents.blackboard.protocol import (
        DesignContextMappedProtocol,
    )

    key = DesignContextMappedProtocol.event_key(
        source_name="ns:foo", path="vcm", millis=1,
    )
    # The on-wire key has the substituted form; the parsed name is
    # the original.
    assert ":foo" not in key.split("design_context_mapped:")[1].split(":")[0]
    parsed = DesignContextMappedProtocol.parse_event_key(key)
    assert parsed["source_name"] == "ns:foo"
    assert parsed["path"] == "vcm"


def test_design_context_mapped_protocol_patterns() -> None:
    from polymathera.colony.agents.blackboard.protocol import (
        DesignContextMappedProtocol,
    )

    assert DesignContextMappedProtocol.event_pattern() == (
        "design_context_mapped:*"
    )
    assert DesignContextMappedProtocol.event_pattern_for_source("foo") == (
        "design_context_mapped:foo:*"
    )
    assert DesignContextMappedProtocol.event_pattern_for_path("vcm") == (
        "design_context_mapped:*:vcm:*"
    )


def test_design_context_mapped_protocol_rejects_alien_key() -> None:
    from polymathera.colony.agents.blackboard.protocol import (
        DesignContextMappedProtocol,
    )

    with pytest.raises(ValueError, match="Not a DesignContextMapped"):
        DesignContextMappedProtocol.parse_event_key("monorepo_commit:foo:bar")
