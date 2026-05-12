"""L1-F per-extension validator registry — Risk #5 enforcement on the
project-substance write surface.

One registry of ``(matcher, validator, name)`` triples. The
:class:`ProjectAuthoringCapability` queries
:func:`validators_for` with the path of the file the action just
wrote, runs every matcher that applies, and aborts the commit on the
first failure.

Built-in validators ship as ``DEFAULT_VALIDATORS`` (registered at
module import). CPS PR 6 (L2-G) calls :func:`register_validator` to
plug in domain-specific checks (CAD parser, FEA solver-input parser,
ReqIF schema, dossier markdown lint, per-domain notebook checks).

The registry is intentionally simple — predicate+function pairs in a
list, evaluated in order, all matching pairs run. No priorities, no
inheritance, no opt-out flags. CPS overrides built-ins by registering
a stricter validator with the same matcher.

The AST allow-list reuses
:mod:`polymathera.colony.design_monorepo.ast_validator` — the SAME
module L1-E goes through. One uniform pipeline regardless of which
half of L4 is being authored.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess  # noqa: S404 — pytest --collect-only is the explicit purpose here
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .ast_validator import validate_python_source
from .models import ProjectArtifactValidationResult


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


PathMatcher = Callable[[Path], bool]
"""Predicate on a working-tree-relative :class:`Path`."""


ValidatorFn = Callable[[Path, Path], ProjectArtifactValidationResult]
"""Validator signature: ``(repo_root, relative_path) -> result``.

``repo_root`` is needed by validators that shell out (e.g. pytest
collect) and need the working directory; ``relative_path`` is the
path inside the repo of the file to validate. The validator reads
the file's *current* on-disk content — L1-F has already written the
new content by the time the validator runs.
"""


@dataclass(frozen=True)
class ArtifactValidator:
    """One registered validator. The capability's dispatch loop runs
    every entry whose ``matcher`` returns ``True`` for the changed path."""

    name: str
    matcher: PathMatcher
    run: ValidatorFn


# ---------------------------------------------------------------------------
# Path matchers — simple suffix / prefix predicates kept inline so the
# matcher set lives next to the registration calls (single source of
# truth for "what counts as a Python file under src/ or tests/").
# ---------------------------------------------------------------------------


def _is_python_under_src_or_tests(p: Path) -> bool:
    if p.suffix != ".py":
        return False
    parts = p.parts
    return bool(parts) and parts[0] in {"src", "tests"}


def _is_test_python(p: Path) -> bool:
    return p.suffix == ".py" and bool(p.parts) and p.parts[0] == "tests"


def _is_dossier_markdown(p: Path) -> bool:
    return p.suffix == ".md" and bool(p.parts) and p.parts[0] == "dossier"


def _is_ipynb(p: Path) -> bool:
    return p.suffix == ".ipynb"


def _is_reqif(p: Path) -> bool:
    return p.suffix == ".reqif"


def _is_cad(p: Path) -> bool:
    return p.suffix in {".step", ".stp", ".iges", ".igs"}


def _is_fea_input(p: Path) -> bool:
    return p.suffix in {".inp", ".med"}


# ---------------------------------------------------------------------------
# Built-in validators
# ---------------------------------------------------------------------------


def _validate_ast(repo_root: Path, rel: Path) -> ProjectArtifactValidationResult:
    """AST allow-list — identical to L1-E's gate. Rejects shell-spawn,
    eval/exec, disallowed imports."""
    abs_path = repo_root / rel
    try:
        source = abs_path.read_text(encoding="utf-8")
    except OSError as exc:
        return ProjectArtifactValidationResult(
            validator="ast_allow_list", ok=False, detail=f"read failed: {exc}",
        )
    report = validate_python_source(source, path=abs_path)
    if report.ok:
        return ProjectArtifactValidationResult(validator="ast_allow_list", ok=True)
    detail = (
        report.syntax_error
        or "; ".join(f"line {iss.line}: {iss.detail}" for iss in report.issues)
    )
    return ProjectArtifactValidationResult(
        validator="ast_allow_list", ok=False, detail=detail,
    )


_PYTEST_COLLECT_TIMEOUT_SECONDS = 60


def _validate_pytest_collect(
    repo_root: Path, rel: Path,
) -> ProjectArtifactValidationResult:
    """``pytest --collect-only`` against the just-written test module.

    Best-effort: when ``pytest`` is not installed the validator passes
    with an explanatory detail rather than blocking the action. A
    collection error (syntax error in fixtures, import failure)
    blocks; a "no tests collected" result passes (an empty test stub
    is a legitimate intermediate state during authoring).
    """
    if shutil.which("pytest") is None:
        return ProjectArtifactValidationResult(
            validator="pytest_collect",
            ok=True,
            detail="pytest not installed in this environment; skipped",
        )
    try:
        proc = subprocess.run(  # noqa: S603 — args are explicit, no shell
            ["pytest", "--collect-only", "-q", str(rel)],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=_PYTEST_COLLECT_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ProjectArtifactValidationResult(
            validator="pytest_collect", ok=False,
            detail=f"pytest collect timed out after {_PYTEST_COLLECT_TIMEOUT_SECONDS}s",
        )
    except OSError as exc:
        return ProjectArtifactValidationResult(
            validator="pytest_collect", ok=False, detail=f"spawn failed: {exc}",
        )
    # pytest exits 0 (collected) or 5 (no tests collected) on success.
    if proc.returncode in (0, 5):
        return ProjectArtifactValidationResult(
            validator="pytest_collect", ok=True,
            detail=_truncate(proc.stdout.strip(), 240),
        )
    return ProjectArtifactValidationResult(
        validator="pytest_collect", ok=False,
        detail=_truncate(
            (proc.stderr or proc.stdout).strip()
            or f"pytest exited {proc.returncode}",
            480,
        ),
    )


def _validate_dossier_markdown(
    repo_root: Path, rel: Path,
) -> ProjectArtifactValidationResult:
    """Minimal markdown sanity for ``dossier/**/*.md`` — UTF-8, non-empty,
    at least one heading. Real lint / cross-reference checks live in
    CPS L2-G (PR 6)."""
    abs_path = repo_root / rel
    try:
        source = abs_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return ProjectArtifactValidationResult(
            validator="dossier_markdown", ok=False,
            detail=f"read failed: {exc}",
        )
    if not source.strip():
        return ProjectArtifactValidationResult(
            validator="dossier_markdown", ok=False, detail="empty document",
        )
    has_heading = any(
        line.lstrip().startswith("#") for line in source.splitlines()
    )
    if not has_heading:
        return ProjectArtifactValidationResult(
            validator="dossier_markdown", ok=False,
            detail="no markdown heading (lines starting with '#')",
        )
    return ProjectArtifactValidationResult(validator="dossier_markdown", ok=True)


def _validate_ipynb(repo_root: Path, rel: Path) -> ProjectArtifactValidationResult:
    """Parse the notebook as JSON and AST-check every Python code cell.

    The notebook format guarantees ``cells`` is a list; each cell with
    ``cell_type=='code'`` has a ``source`` (str or list of str). Non-
    Python kernels are not validated for now — CPS PR 6 wires per-
    kernel checks.
    """
    abs_path = repo_root / rel
    try:
        payload = json.loads(abs_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        return ProjectArtifactValidationResult(
            validator="ipynb_ast", ok=False, detail=f"read failed: {exc}",
        )
    except json.JSONDecodeError as exc:
        return ProjectArtifactValidationResult(
            validator="ipynb_ast", ok=False, detail=f"not valid JSON: {exc}",
        )
    if not isinstance(payload, dict) or "cells" not in payload:
        return ProjectArtifactValidationResult(
            validator="ipynb_ast", ok=False,
            detail="missing top-level 'cells'",
        )
    cells = payload.get("cells", [])
    if not isinstance(cells, list):
        return ProjectArtifactValidationResult(
            validator="ipynb_ast", ok=False, detail="'cells' is not a list",
        )
    lang = (
        payload.get("metadata", {})
        .get("kernelspec", {})
        .get("language", "python")
    )
    if lang != "python":
        return ProjectArtifactValidationResult(
            validator="ipynb_ast", ok=True,
            detail=f"non-python kernel {lang!r}; AST check skipped",
        )
    for i, cell in enumerate(cells):
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        raw = cell.get("source", "")
        source = "".join(raw) if isinstance(raw, list) else raw
        if not isinstance(source, str) or not source.strip():
            continue
        report = validate_python_source(source, path=f"{rel}#cell{i}")
        if not report.ok:
            detail = (
                report.syntax_error
                or "; ".join(f"line {iss.line}: {iss.detail}" for iss in report.issues)
            )
            return ProjectArtifactValidationResult(
                validator="ipynb_ast", ok=False,
                detail=f"cell {i}: {detail}",
            )
    return ProjectArtifactValidationResult(validator="ipynb_ast", ok=True)


def _validate_non_empty(name: str) -> ValidatorFn:
    """Factory for the CAD / FEA / ReqIF stubs — every L1-F-authored
    binary artifact must be non-empty. CPS PR 6 replaces these with
    real format validators."""
    def _run(repo_root: Path, rel: Path) -> ProjectArtifactValidationResult:
        abs_path = repo_root / rel
        try:
            size = abs_path.stat().st_size
        except OSError as exc:
            return ProjectArtifactValidationResult(
                validator=name, ok=False, detail=f"stat failed: {exc}",
            )
        if size == 0:
            return ProjectArtifactValidationResult(
                validator=name, ok=False, detail="empty file",
            )
        return ProjectArtifactValidationResult(
            validator=name, ok=True, detail=f"{size} bytes (stub validator)",
        )
    return _run


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


DEFAULT_VALIDATORS: tuple[ArtifactValidator, ...] = (
    ArtifactValidator(
        name="ast_allow_list",
        matcher=_is_python_under_src_or_tests,
        run=_validate_ast,
    ),
    ArtifactValidator(
        name="pytest_collect",
        matcher=_is_test_python,
        run=_validate_pytest_collect,
    ),
    ArtifactValidator(
        name="dossier_markdown",
        matcher=_is_dossier_markdown,
        run=_validate_dossier_markdown,
    ),
    ArtifactValidator(
        name="ipynb_ast",
        matcher=_is_ipynb,
        run=_validate_ipynb,
    ),
    ArtifactValidator(
        name="cad_non_empty",
        matcher=_is_cad,
        run=_validate_non_empty("cad_non_empty"),
    ),
    ArtifactValidator(
        name="fea_input_non_empty",
        matcher=_is_fea_input,
        run=_validate_non_empty("fea_input_non_empty"),
    ),
    ArtifactValidator(
        name="reqif_non_empty",
        matcher=_is_reqif,
        run=_validate_non_empty("reqif_non_empty"),
    ),
)


_REGISTRY: list[ArtifactValidator] = list(DEFAULT_VALIDATORS)


def register_validator(validator: ArtifactValidator) -> None:
    """Add ``validator`` to the registry.

    Validators are evaluated in registration order; multiple validators
    whose matcher returns ``True`` for a given path all run, and the
    action commits only when every one returns ``ok=True``. CPS PR 6
    (L2-G) registers per-domain validators via this hook from its
    ``register_components()`` entry-point handler.
    """
    _REGISTRY.append(validator)


def reset_to_defaults() -> None:
    """Restore the registry to :data:`DEFAULT_VALIDATORS` — for tests."""
    _REGISTRY[:] = list(DEFAULT_VALIDATORS)


def validators_for(rel_path: Path) -> Sequence[ArtifactValidator]:
    """Return every registered validator whose matcher accepts
    ``rel_path``. Order matches registration order."""
    return tuple(v for v in _REGISTRY if v.matcher(rel_path))


def run_validators(
    repo_root: Path, rel_path: Path,
) -> tuple[ProjectArtifactValidationResult, ...]:
    """Run every applicable validator on ``rel_path`` (relative to
    ``repo_root``). Returns the full result list — caller decides
    whether any failure aborts the action."""
    results: list[ProjectArtifactValidationResult] = []
    for v in validators_for(rel_path):
        try:
            results.append(v.run(repo_root, rel_path))
        except Exception as exc:  # noqa: BLE001 - validator must never crash the action
            logger.exception(
                "L1-F validator %r raised on %s", v.name, rel_path,
            )
            results.append(ProjectArtifactValidationResult(
                validator=v.name, ok=False,
                detail=f"validator raised {type(exc).__name__}: {exc}",
            ))
    return tuple(results)


def all_failed(
    results: Iterable[ProjectArtifactValidationResult],
) -> tuple[ProjectArtifactValidationResult, ...]:
    """Return only the failing results — empty tuple means "everyone passed"."""
    return tuple(r for r in results if not r.ok)


__all__ = (
    "ArtifactValidator",
    "DEFAULT_VALIDATORS",
    "PathMatcher",
    "ValidatorFn",
    "all_failed",
    "register_validator",
    "reset_to_defaults",
    "run_validators",
    "validators_for",
)
