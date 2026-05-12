"""Tests for L1-F: ``ProjectAuthoringCapability`` and the L1-F
validator pipeline.

Covers:
- All seven action_kinds.
- Path safety: relative-only, no ``..``, no ``.colony/``, no ``.git/``.
- Validator dispatch: AST on ``.py`` under src/ tests/; pytest collect
  on tests/; markdown on dossier/*.md; ipynb cell AST; non-empty
  stubs on CAD/FEA/ReqIF.
- Atomic rollback when validation fails.
- Audit event shape.
- Read-side companions on RepoStateProvider.
- Risk #9 session lint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from polymathera.colony.agents.blackboard.protocol import DesignMonorepoEventProtocol
from polymathera.colony.design_monorepo import (
    PROJECT_ACTION_KINDS,
    DesignMonorepoClient,
    DesignMonorepoError,
    ProjectArtifactAuthoredPayload,
    ProjectArtifactValidationResult,
    ProjectAuthoringCapability,
    RepoStateProvider,
)
from polymathera.colony.design_monorepo import artifact_validators
from polymathera.colony.design_monorepo.session_lint import (
    DESIGN_ARTIFACT_TOP_LEVELS,
    LintFinding,
    SHELL_CAPABILITY_NAME,
    lint_session_actions,
)


@pytest.fixture
def authoring(bootstrapped_repo: DesignMonorepoClient) -> ProjectAuthoringCapability:
    cap = ProjectAuthoringCapability(
        agent=None, scope_id="test", working_dir=bootstrapped_repo.working_dir,
    )
    cap._client = bootstrapped_repo
    return cap


@pytest.fixture
def state_provider(bootstrapped_repo: DesignMonorepoClient) -> RepoStateProvider:
    cap = RepoStateProvider(
        agent=None, scope_id="test", working_dir=bootstrapped_repo.working_dir,
    )
    cap._client = bootstrapped_repo
    return cap


@pytest.fixture(autouse=True)
def _validators_reset():
    """Ensure each test starts with the default validator set —
    tests that ``register_validator(...)`` cannot leak across."""
    yield
    artifact_validators.reset_to_defaults()


# ---------------------------------------------------------------------------
# Single-source-of-truth audit
# ---------------------------------------------------------------------------


def test_payload_literal_matches_project_action_kinds() -> None:
    """``ProjectArtifactAuthoredPayload.action_kind`` Literal must
    equal :data:`PROJECT_ACTION_KINDS` — the type system requires
    literal strings, but the tuple is the canonical set. Drift
    fails this test."""
    from typing import get_args

    annotation = ProjectArtifactAuthoredPayload.model_fields["action_kind"].annotation
    assert set(get_args(annotation)) == set(PROJECT_ACTION_KINDS)


def test_seven_action_kinds() -> None:
    assert len(PROJECT_ACTION_KINDS) == 7
    # Spot-check the kinds the alignment plan calls out.
    for required in (
        "write_file", "edit_file", "delete_file", "move_file",
        "insert_lines", "delete_lines", "replace_lines",
    ):
        assert required in PROJECT_ACTION_KINDS


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_absolute_path_rejected(authoring: ProjectAuthoringCapability) -> None:
    with pytest.raises(DesignMonorepoError, match="relative"):
        await authoring.write_file("/etc/passwd", "x")


@pytest.mark.asyncio
async def test_traversal_rejected(authoring: ProjectAuthoringCapability) -> None:
    with pytest.raises(DesignMonorepoError, match="escapes"):
        await authoring.write_file("../escape.txt", "x")


@pytest.mark.asyncio
async def test_colony_path_rejected(authoring: ProjectAuthoringCapability) -> None:
    with pytest.raises(DesignMonorepoError, match="ToolBuilder"):
        await authoring.write_file(".colony/agents/whatever.py", "x")


@pytest.mark.asyncio
async def test_git_path_rejected(authoring: ProjectAuthoringCapability) -> None:
    with pytest.raises(DesignMonorepoError, match=".git"):
        await authoring.write_file(".git/hooks/post-commit", "x")


@pytest.mark.asyncio
async def test_empty_path_rejected(authoring: ProjectAuthoringCapability) -> None:
    with pytest.raises(DesignMonorepoError):
        await authoring.write_file("", "x")


# ---------------------------------------------------------------------------
# write_file + AST validator (round trip)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_file_under_src_passes_ast_and_commits(
    authoring: ProjectAuthoringCapability,
    bootstrapped_repo: DesignMonorepoClient,
) -> None:
    payload = await authoring.write_file(
        "src/opm_meg/calibration.py",
        "def calibrate() -> int:\n    return 42\n",
    )
    assert payload.action_kind == "write_file"
    assert payload.affected_paths == ("src/opm_meg/calibration.py",)
    abs_p = bootstrapped_repo.working_dir / "src" / "opm_meg" / "calibration.py"
    assert abs_p.read_text() == "def calibrate() -> int:\n    return 42\n"
    # AST validator must have run and passed.
    names = {r.validator for r in payload.pre_commit_validation_results}
    assert "ast_allow_list" in names
    assert all(r.ok for r in payload.pre_commit_validation_results)
    # Commit landed.
    head = bootstrapped_repo._repo.head.commit
    assert head.hexsha == payload.commit_sha
    assert "L1-F write_file: src/opm_meg/calibration.py" in head.message


@pytest.mark.asyncio
async def test_write_file_rejected_by_ast_rolls_back(
    authoring: ProjectAuthoringCapability,
    bootstrapped_repo: DesignMonorepoClient,
) -> None:
    abs_p = bootstrapped_repo.working_dir / "src" / "evil.py"
    assert not abs_p.exists()
    with pytest.raises(DesignMonorepoError, match="ast_allow_list"):
        await authoring.write_file(
            "src/evil.py",
            "import subprocess\nsubprocess.run(['rm', '-rf', '/'])\n",
        )
    # Rollback: the file must NOT exist after the failed action.
    assert not abs_p.exists()


@pytest.mark.asyncio
async def test_write_file_overwrite_rolls_back_on_validation_fail(
    authoring: ProjectAuthoringCapability,
    bootstrapped_repo: DesignMonorepoClient,
) -> None:
    await authoring.write_file("src/pkg/mod.py", "def good() -> None:\n    pass\n")
    abs_p = bootstrapped_repo.working_dir / "src" / "pkg" / "mod.py"
    good = abs_p.read_text()
    with pytest.raises(DesignMonorepoError):
        await authoring.write_file(
            "src/pkg/mod.py",
            "from os import system\nsystem('id')\n",
        )
    assert abs_p.read_text() == good  # rolled back


@pytest.mark.asyncio
async def test_write_file_outside_src_or_tests_skips_ast(
    authoring: ProjectAuthoringCapability,
    bootstrapped_repo: DesignMonorepoClient,
) -> None:
    """A ``.py`` at the repo root (not under src/ or tests/) is NOT a
    project-substance python module per the matcher — the AST gate
    only fires under src/ and tests/."""
    payload = await authoring.write_file(
        "scratch.py", "import subprocess\n",
    )
    names = {r.validator for r in payload.pre_commit_validation_results}
    assert "ast_allow_list" not in names


# ---------------------------------------------------------------------------
# edit_file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_edit_file_unique_match(
    authoring: ProjectAuthoringCapability,
    bootstrapped_repo: DesignMonorepoClient,
) -> None:
    await authoring.write_file(
        "src/a/m.py", "def f() -> int:\n    return 1\n",
    )
    payload = await authoring.edit_file(
        "src/a/m.py", "return 1", "return 2",
    )
    assert payload.action_kind == "edit_file"
    abs_p = bootstrapped_repo.working_dir / "src" / "a" / "m.py"
    assert "return 2" in abs_p.read_text()


@pytest.mark.asyncio
async def test_edit_file_zero_match_rejected(
    authoring: ProjectAuthoringCapability,
) -> None:
    await authoring.write_file("src/a/m.py", "x = 1\n")
    with pytest.raises(DesignMonorepoError, match="not found"):
        await authoring.edit_file("src/a/m.py", "y = 1", "y = 2")


@pytest.mark.asyncio
async def test_edit_file_multiple_match_rejected(
    authoring: ProjectAuthoringCapability,
) -> None:
    await authoring.write_file("src/a/m.py", "x = 1\nx = 1\n")
    with pytest.raises(DesignMonorepoError, match="matches 2 times"):
        await authoring.edit_file("src/a/m.py", "x = 1", "x = 2")


@pytest.mark.asyncio
async def test_edit_file_missing_file_rejected(
    authoring: ProjectAuthoringCapability,
) -> None:
    with pytest.raises(DesignMonorepoError, match="does not exist"):
        await authoring.edit_file("src/missing.py", "a", "b")


# ---------------------------------------------------------------------------
# delete_file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_file_succeeds(
    authoring: ProjectAuthoringCapability,
    bootstrapped_repo: DesignMonorepoClient,
) -> None:
    await authoring.write_file("src/junk.py", "pass\n")
    payload = await authoring.delete_file("src/junk.py")
    assert payload.action_kind == "delete_file"
    abs_p = bootstrapped_repo.working_dir / "src" / "junk.py"
    assert not abs_p.exists()


@pytest.mark.asyncio
async def test_delete_file_missing_rejected(
    authoring: ProjectAuthoringCapability,
) -> None:
    with pytest.raises(DesignMonorepoError, match="does not exist"):
        await authoring.delete_file("src/never_existed.py")


# ---------------------------------------------------------------------------
# move_file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_move_file_succeeds(
    authoring: ProjectAuthoringCapability,
    bootstrapped_repo: DesignMonorepoClient,
) -> None:
    await authoring.write_file("src/old/m.py", "x = 1\n")
    payload = await authoring.move_file("src/old/m.py", "src/new/m.py")
    assert payload.action_kind == "move_file"
    assert payload.affected_paths == ("src/old/m.py", "src/new/m.py")
    root = bootstrapped_repo.working_dir
    assert not (root / "src" / "old" / "m.py").exists()
    assert (root / "src" / "new" / "m.py").read_text() == "x = 1\n"


@pytest.mark.asyncio
async def test_move_file_dst_exists_rejected(
    authoring: ProjectAuthoringCapability,
) -> None:
    await authoring.write_file("src/a.py", "pass\n")
    await authoring.write_file("src/b.py", "pass\n")
    with pytest.raises(DesignMonorepoError, match="already exists"):
        await authoring.move_file("src/a.py", "src/b.py")


# ---------------------------------------------------------------------------
# Line-level ops
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_insert_lines_at_end(
    authoring: ProjectAuthoringCapability,
    bootstrapped_repo: DesignMonorepoClient,
) -> None:
    await authoring.write_file("src/m.py", "a = 1\nb = 2\n")
    # File has 2 lines (b=2 then trailing newline closes line 2). Append.
    payload = await authoring.insert_lines("src/m.py", after_line=2, content="c = 3\n")
    assert payload.action_kind == "insert_lines"
    body = (bootstrapped_repo.working_dir / "src" / "m.py").read_text()
    assert body == "a = 1\nb = 2\nc = 3\n"


@pytest.mark.asyncio
async def test_insert_lines_at_start(
    authoring: ProjectAuthoringCapability,
    bootstrapped_repo: DesignMonorepoClient,
) -> None:
    await authoring.write_file("src/m.py", "a = 1\n")
    await authoring.insert_lines("src/m.py", after_line=0, content="z = 0\n")
    body = (bootstrapped_repo.working_dir / "src" / "m.py").read_text()
    assert body == "z = 0\na = 1\n"


@pytest.mark.asyncio
async def test_insert_lines_out_of_range(
    authoring: ProjectAuthoringCapability,
) -> None:
    await authoring.write_file("src/m.py", "a = 1\n")
    with pytest.raises(DesignMonorepoError, match="out of range"):
        await authoring.insert_lines("src/m.py", after_line=99, content="x\n")


@pytest.mark.asyncio
async def test_delete_lines(
    authoring: ProjectAuthoringCapability,
    bootstrapped_repo: DesignMonorepoClient,
) -> None:
    await authoring.write_file("src/m.py", "a = 1\nb = 2\nc = 3\n")
    await authoring.delete_lines("src/m.py", start_line=2, end_line=2)
    body = (bootstrapped_repo.working_dir / "src" / "m.py").read_text()
    assert body == "a = 1\nc = 3\n"


@pytest.mark.asyncio
async def test_delete_lines_range_validated(
    authoring: ProjectAuthoringCapability,
) -> None:
    await authoring.write_file("src/m.py", "x = 1\n")
    with pytest.raises(DesignMonorepoError, match="subset of"):
        await authoring.delete_lines("src/m.py", start_line=0, end_line=1)
    with pytest.raises(DesignMonorepoError):
        await authoring.delete_lines("src/m.py", start_line=1, end_line=99)


@pytest.mark.asyncio
async def test_replace_lines(
    authoring: ProjectAuthoringCapability,
    bootstrapped_repo: DesignMonorepoClient,
) -> None:
    await authoring.write_file(
        "src/m.py", "def f() -> int:\n    return 1\n",
    )
    await authoring.replace_lines(
        "src/m.py", start_line=2, end_line=2, content="    return 2\n",
    )
    body = (bootstrapped_repo.working_dir / "src" / "m.py").read_text()
    assert body == "def f() -> int:\n    return 2\n"


# ---------------------------------------------------------------------------
# Validator pipeline — dossier markdown, ipynb, stubs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dossier_markdown_requires_heading(
    authoring: ProjectAuthoringCapability,
) -> None:
    with pytest.raises(DesignMonorepoError, match="dossier_markdown"):
        await authoring.write_file(
            "dossier/510k/section_4.md", "no heading here at all\n",
        )


@pytest.mark.asyncio
async def test_dossier_markdown_with_heading_passes(
    authoring: ProjectAuthoringCapability,
) -> None:
    payload = await authoring.write_file(
        "dossier/510k/section_4.md", "# Section 4\n\nContent.\n",
    )
    names = {r.validator for r in payload.pre_commit_validation_results}
    assert "dossier_markdown" in names


@pytest.mark.asyncio
async def test_ipynb_validator_runs_ast_on_python_cells(
    authoring: ProjectAuthoringCapability,
) -> None:
    nb = (
        '{"cells": [{"cell_type": "code", "source": "import subprocess\\n"}], '
        '"metadata": {"kernelspec": {"language": "python"}}, '
        '"nbformat": 4, "nbformat_minor": 5}'
    )
    with pytest.raises(DesignMonorepoError, match="ipynb_ast"):
        await authoring.write_file("notebooks/analysis.ipynb", nb)


@pytest.mark.asyncio
async def test_ipynb_validator_passes_clean_notebook(
    authoring: ProjectAuthoringCapability,
) -> None:
    nb = (
        '{"cells": [{"cell_type": "code", "source": "x = 1\\n"}], '
        '"metadata": {"kernelspec": {"language": "python"}}, '
        '"nbformat": 4, "nbformat_minor": 5}'
    )
    payload = await authoring.write_file("notebooks/a.ipynb", nb)
    names = {r.validator for r in payload.pre_commit_validation_results}
    assert "ipynb_ast" in names


@pytest.mark.asyncio
async def test_cad_non_empty_validator(
    authoring: ProjectAuthoringCapability,
) -> None:
    payload = await authoring.write_file(
        "models/cad/widget.step", "DUMMY STEP CONTENT\n",
    )
    names = {r.validator for r in payload.pre_commit_validation_results}
    assert "cad_non_empty" in names


@pytest.mark.asyncio
async def test_register_validator_extension_hook(
    authoring: ProjectAuthoringCapability,
) -> None:
    """CPS PR 6 plugs in domain validators via this hook; verify it
    composes correctly with the built-ins."""
    calls: list[Path] = []

    def _domain_validator(repo_root: Path, rel: Path) -> ProjectArtifactValidationResult:
        calls.append(rel)
        return ProjectArtifactValidationResult(validator="my_domain", ok=True)

    artifact_validators.register_validator(
        artifact_validators.ArtifactValidator(
            name="my_domain",
            matcher=lambda p: p.suffix == ".py" and p.parts[:1] == ("src",),
            run=_domain_validator,
        ),
    )
    payload = await authoring.write_file("src/x.py", "x = 1\n")
    assert calls == [Path("src/x.py")]
    names = {r.validator for r in payload.pre_commit_validation_results}
    assert {"ast_allow_list", "my_domain"} <= names


@pytest.mark.asyncio
async def test_failing_extension_validator_blocks_commit(
    authoring: ProjectAuthoringCapability,
    bootstrapped_repo: DesignMonorepoClient,
) -> None:
    artifact_validators.register_validator(
        artifact_validators.ArtifactValidator(
            name="picky",
            matcher=lambda p: p.suffix == ".txt",
            run=lambda repo, rel: ProjectArtifactValidationResult(
                validator="picky", ok=False, detail="nope",
            ),
        ),
    )
    with pytest.raises(DesignMonorepoError, match="picky: nope"):
        await authoring.write_file("data/x.txt", "content\n")
    assert not (bootstrapped_repo.working_dir / "data" / "x.txt").exists()


# ---------------------------------------------------------------------------
# Read-side: list_packages, list_design_artifacts, summarize_project_layout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_packages(
    authoring: ProjectAuthoringCapability,
    state_provider: RepoStateProvider,
) -> None:
    await authoring.write_file("src/alpha/__init__.py", "")
    await authoring.write_file("src/beta/__init__.py", "")
    # Non-package directory (no __init__.py) is excluded.
    await authoring.write_file("src/scripts/run.py", "x = 1\n")
    pkgs = await state_provider.list_packages()
    assert pkgs == ["alpha", "beta"]


@pytest.mark.asyncio
async def test_list_design_artifacts(
    authoring: ProjectAuthoringCapability,
    state_provider: RepoStateProvider,
) -> None:
    await authoring.write_file("src/a/__init__.py", "")
    await authoring.write_file("src/a/m.py", "x = 1\n")
    await authoring.write_file("tests/test_a.py", "def test_x() -> None:\n    pass\n")
    await authoring.write_file(
        "dossier/510k/section.md", "# Section\n",
    )
    await authoring.write_file("models/cad/p.step", "STEP\n")
    all_a = await state_provider.list_design_artifacts()
    assert "src/a/m.py" in all_a
    assert "tests/test_a.py" in all_a
    assert "dossier/510k/section.md" in all_a
    assert "models/cad/p.step" in all_a

    cad = await state_provider.list_design_artifacts(kind="cad")
    assert cad == ["models/cad/p.step"]

    tests = await state_provider.list_design_artifacts(kind="test")
    assert tests == ["tests/test_a.py"]


@pytest.mark.asyncio
async def test_list_design_artifacts_rejects_unknown_kind(
    state_provider: RepoStateProvider,
) -> None:
    with pytest.raises(ValueError, match="unknown kind"):
        await state_provider.list_design_artifacts(kind="bogus")


@pytest.mark.asyncio
async def test_summarize_project_layout(
    authoring: ProjectAuthoringCapability,
    state_provider: RepoStateProvider,
) -> None:
    await authoring.write_file("src/a/__init__.py", "")
    await authoring.write_file("tests/test_a.py", "pass\n")
    summary = await state_provider.summarize_project_layout()
    assert ".colony" not in summary["top_level"]
    assert ".git" not in summary["top_level"]
    assert "src" in summary["top_level"]
    assert "tests" in summary["top_level"]
    assert summary["counts"]["python_module"] >= 1
    assert summary["counts"]["test"] >= 1


# ---------------------------------------------------------------------------
# Audit event protocol
# ---------------------------------------------------------------------------


def test_project_artifact_authored_key_round_trips() -> None:
    key = DesignMonorepoEventProtocol.project_artifact_authored_key(
        "write_file", "src/a.py",
    )
    kind, path = DesignMonorepoEventProtocol.parse_project_artifact_authored_key(key)
    assert kind == "write_file"
    assert path == "src/a.py"


def test_project_artifact_authored_pattern() -> None:
    pattern = DesignMonorepoEventProtocol.project_artifact_authored_pattern()
    assert pattern.endswith("*")
    assert pattern.startswith("project_artifact_authored:")


def test_project_artifact_authored_parse_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        DesignMonorepoEventProtocol.parse_project_artifact_authored_key(
            "extension_authored:agents:foo",
        )


# ---------------------------------------------------------------------------
# Session lint (Risk #9)
# ---------------------------------------------------------------------------


def test_session_lint_design_artifact_top_levels_are_canonical() -> None:
    """The lint's target set must reflect the actual project-substance
    top-level dirs L1-F manages. If we ever rename one (e.g. ``docs/``
    → ``documentation/``), update this set."""
    assert "src" in DESIGN_ARTIFACT_TOP_LEVELS
    assert "tests" in DESIGN_ARTIFACT_TOP_LEVELS
    assert "dossier" in DESIGN_ARTIFACT_TOP_LEVELS
    assert ".colony" not in DESIGN_ARTIFACT_TOP_LEVELS


def test_session_lint_flags_shell_write_to_src() -> None:
    findings = lint_session_actions([
        {
            "capability": SHELL_CAPABILITY_NAME,
            "action_kind": "write_file",
            "kwargs": {"path": "src/opm_meg/calibration.py", "content": "..."},
        },
    ])
    assert len(findings) == 1
    f = findings[0]
    assert isinstance(f, LintFinding)
    assert f.capability == SHELL_CAPABILITY_NAME
    assert f.path == "src/opm_meg/calibration.py"


def test_session_lint_ignores_shell_write_to_scratch() -> None:
    findings = lint_session_actions([
        {
            "capability": SHELL_CAPABILITY_NAME,
            "action_kind": "write_file",
            "kwargs": {"path": "/tmp/debug.log", "content": "..."},
        },
    ])
    assert findings == []


def test_session_lint_ignores_project_authoring_writes() -> None:
    findings = lint_session_actions([
        {
            "capability": "ProjectAuthoringCapability",
            "action_kind": "write_file",
            "kwargs": {"path": "src/opm_meg/calibration.py", "content": "..."},
        },
    ])
    assert findings == []


def test_session_lint_skips_malformed_records() -> None:
    findings = lint_session_actions([
        "not-a-dict",
        {},
        {"capability": SHELL_CAPABILITY_NAME},
        {"capability": SHELL_CAPABILITY_NAME, "action_kind": "exec"},
        {"capability": SHELL_CAPABILITY_NAME, "action_kind": "write_file"},
        {
            "capability": SHELL_CAPABILITY_NAME,
            "action_kind": "write_file",
            "kwargs": {"path": None},
        },
    ])
    assert findings == []


def test_session_lint_uses_arguments_alias() -> None:
    """Some transcript schemas use ``arguments`` instead of ``kwargs``."""
    findings = lint_session_actions([
        {
            "capability": SHELL_CAPABILITY_NAME,
            "action_kind": "edit_file",
            "arguments": {"path": "dossier/510k/section.md"},
        },
    ])
    assert len(findings) == 1
    assert findings[0].action_kind == "edit_file"
