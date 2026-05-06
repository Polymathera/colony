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
    KnowledgeRoute,
    REPO_MAP_DIR,
    REPO_MAP_FILENAME,
    RepoMap,
    SourceSpec,
)


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
    assert len(rm.sources) == 1
    assert rm.sources[0].name == "default"
    assert rm.sources[0].type == "git_repo"


def test_load_falls_back_to_default_when_file_absent(tmp_path: Path) -> None:
    repo_root = tmp_path / "r"
    repo_root.mkdir()
    rm = RepoMap.load(repo_root)
    assert len(rm.sources) == 1
    assert rm.sources[0].name == "default"


def test_load_parses_a_real_repo_map(tmp_path: Path) -> None:
    repo_root = tmp_path / "r"
    (repo_root / REPO_MAP_DIR).mkdir(parents=True)
    (repo_root / REPO_MAP_DIR / REPO_MAP_FILENAME).write_text(
        "schema_version: 1\n"
        "sources:\n"
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
    assert [s.name for s in rm.sources] == ["code", "literature"]
    assert rm.sources[0].start_dir == "tools/"
    assert rm.sources[0].exclude_globs == ["**/build/**"]
    assert rm.sources[1].chunk_target_tokens == 600


def test_load_rejects_unknown_schema_version(tmp_path: Path) -> None:
    repo_root = tmp_path / "r"
    (repo_root / REPO_MAP_DIR).mkdir(parents=True)
    (repo_root / REPO_MAP_DIR / REPO_MAP_FILENAME).write_text(
        "schema_version: 99\nsources: []\n", encoding="utf-8",
    )
    with pytest.raises(ValueError, match="schema_version"):
        RepoMap.load(repo_root)


def test_to_mmap_kwargs_passes_through_origin_url(tmp_path: Path) -> None:
    spec = SourceSpec(
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

    spec = SourceSpec(name="external-foo", type="git_repo", submodule="third_party/foo")
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
    spec = SourceSpec(
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


def test_default_repo_map_has_empty_knowledge_routing() -> None:
    rm = RepoMap.default_for_unmapped_repo()
    assert rm.knowledge_routing == []


def test_load_parses_knowledge_routing_block(tmp_path: Path) -> None:
    repo_root = tmp_path / "r"
    (repo_root / REPO_MAP_DIR).mkdir(parents=True)
    (repo_root / REPO_MAP_DIR / REPO_MAP_FILENAME).write_text(
        "schema_version: 1\n"
        "sources:\n"
        "  - name: code\n"
        "    type: git_repo\n"
        "knowledge_routing:\n"
        "  - paths: ['literature/curated/**/*.pdf']\n"
        "    profile: scientific_paper\n"
        "  - paths: ['standards/**/*.pdf']\n"
        "  - paths: ['literature/promoted/**/*.pdf']\n"
        "    ingest_to: vcm\n",
        encoding="utf-8",
    )
    rm = RepoMap.load(repo_root)
    assert len(rm.knowledge_routing) == 3
    assert rm.knowledge_routing[0].paths == ["literature/curated/**/*.pdf"]
    assert rm.knowledge_routing[0].profile == "scientific_paper"
    # ``ingest_to`` defaults to ``knowledge_base`` when omitted.
    assert rm.knowledge_routing[0].ingest_to == "knowledge_base"
    assert rm.knowledge_routing[1].profile is None
    assert rm.knowledge_routing[1].ingest_to == "knowledge_base"
    # Promoted-to-VCM rows are documentation-only on the KB side.
    assert rm.knowledge_routing[2].ingest_to == "vcm"


def test_invalid_ingest_to_value_is_rejected(tmp_path: Path) -> None:
    repo_root = tmp_path / "r"
    (repo_root / REPO_MAP_DIR).mkdir(parents=True)
    (repo_root / REPO_MAP_DIR / REPO_MAP_FILENAME).write_text(
        "schema_version: 1\n"
        "sources: []\n"
        "knowledge_routing:\n"
        "  - paths: ['x/*']\n"
        "    ingest_to: bogus\n",
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

    none_set = SourceSpec(name="bare", type="git_repo")
    assert none_set.to_mmap_config_overrides() == {}

    partial = SourceSpec(
        name="code", type="git_repo", flush_threshold=8,
    )
    assert partial.to_mmap_config_overrides() == {"flush_threshold": 8}

    full = SourceSpec(
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
        "schema_version: 1\n"
        "sources:\n"
        "  - name: code\n"
        "    type: git_repo\n"
        "    flush_threshold: 50\n"
        "    flush_token_budget: 8192\n"
        "    pinned: true\n",
        encoding="utf-8",
    )
    rm = RepoMap.load(repo_root)
    spec = rm.sources[0]
    assert spec.flush_threshold == 50
    assert spec.flush_token_budget == 8192
    assert spec.pinned is True


def test_knowledge_route_matches_glob_relative_to_repo_root() -> None:
    route = KnowledgeRoute(paths=["literature/curated/**/*.pdf"])
    assert route.matches("literature/curated/2024/seminal.pdf")
    assert route.matches("literature/curated/x.pdf")
    assert not route.matches("tools/main.py")
    assert not route.matches("literature/notes/x.md")


def test_extra_fields_are_rejected(tmp_path: Path) -> None:
    repo_root = tmp_path / "r"
    (repo_root / REPO_MAP_DIR).mkdir(parents=True)
    (repo_root / REPO_MAP_DIR / REPO_MAP_FILENAME).write_text(
        "schema_version: 1\n"
        "sources:\n"
        "  - name: x\n"
        "    type: git_repo\n"
        "    bogus_field: 123\n",
        encoding="utf-8",
    )
    with pytest.raises(Exception):
        RepoMap.load(repo_root)
