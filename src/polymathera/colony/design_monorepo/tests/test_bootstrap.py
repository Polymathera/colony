"""Tests for ``bootstrap_design_monorepo``."""

from __future__ import annotations

from pathlib import Path

import pytest

from polymathera.colony.design_monorepo import (
    DECISIONS_DIR,
    DesignMonorepoClient,
    DesignMonorepoError,
    DesignMonorepoManifest,
    MANIFEST_RELATIVE_PATH,
    REGISTRY_RELATIVE_PATH,
    SCAFFOLD_DIRS,
    bootstrap_design_monorepo,
)


def test_bootstrap_creates_layout(bootstrapped_repo: DesignMonorepoClient) -> None:
    wd = bootstrapped_repo.working_dir
    for sub in SCAFFOLD_DIRS:
        assert (wd / sub).is_dir(), sub
    assert (wd / MANIFEST_RELATIVE_PATH).is_file()
    assert (wd / REGISTRY_RELATIVE_PATH).is_file()
    assert (wd / ".gitattributes").is_file()
    assert (wd / ".gitignore").is_file()
    assert (wd / "README.md").is_file()
    assert (wd / DECISIONS_DIR).is_dir()
    assert (wd / ".colony" / "checkpoints.log").is_file()
    assert (wd / ".colony" / "bootstrap_at").is_file()


def test_bootstrap_yields_one_commit(bootstrapped_repo: DesignMonorepoClient) -> None:
    state = bootstrapped_repo.current_state()
    assert state.is_fresh is True
    assert state.uncommitted_changes is False
    assert state.current_branch == "main"


def test_bootstrap_refuses_nonempty_dir(
    manifest: DesignMonorepoManifest, identity, tmp_path: Path,
) -> None:
    target = tmp_path / "repo"
    target.mkdir()
    (target / "stub").write_text("x")
    with pytest.raises(DesignMonorepoError):
        bootstrap_design_monorepo(manifest, target, identity=identity)


def test_bootstrap_registers_merge_drivers(
    bootstrapped_repo: DesignMonorepoClient,
) -> None:
    repo_config_path = (
        Path(bootstrapped_repo.repo.git_dir) / "config"
    )
    cw_str = repo_config_path.read_text("utf-8")
    for driver in (
        "kg-merge",
        "decisions-merge",
        "budget-merge",
        "page-graph-merge",
        "reqif-merge",
    ):
        assert f'[merge "{driver}"]' in cw_str, driver


def test_bootstrap_gitattributes_lists_merge_drivers(
    bootstrapped_repo: DesignMonorepoClient,
) -> None:
    text = (bootstrapped_repo.working_dir / ".gitattributes").read_text("utf-8")
    for driver in ("kg-merge", "decisions-merge", "budget-merge", "page-graph-merge", "reqif-merge"):
        assert f"merge={driver}" in text, driver
    assert "filter=lfs" in text


def test_open_after_bootstrap(
    bootstrapped_repo: DesignMonorepoClient, manifest: DesignMonorepoManifest,
) -> None:
    wd = bootstrapped_repo.working_dir
    re = DesignMonorepoClient.open(wd)
    assert re.manifest.program == manifest.program
