"""Risk #9 lint: detect ``SandboxedShellCapability`` writes that should
have gone through :class:`ProjectAuthoringCapability` (L1-F).

The two halves of L4 must stay in audit-discipline parity. The shell
capability's same-named actions (``write_file``, ``edit_file``)
remain available as **escape hatches** for genuinely throwaway edits
(debug prints, temporary fixtures), but a planner that routes a
design-artifact mutation through them bypasses both the
``DesignCheckpointer`` audit trail and the L1-F validator pipeline.

This linter is a CI check, not a runtime gate. Given a list of action
records from a session transcript, it reports findings where:

- ``capability == "SandboxedShellCapability"``
- ``action_kind in {"write_file", "edit_file"}``
- the action's ``path`` lands under a design-artifact directory
  (``src/``, ``tests/``, ``dossier/``, ``data/``, ``models/``,
  ``notebooks/``, ``docs/``).

Findings are pure data — the caller decides whether to fail the
build or report them as warnings.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


#: Top-level directories that count as "design artifacts" — anything
#: under these paths should go through L1-F, not the shell capability.
#: Source of truth for the "what is a design artifact path" question.
DESIGN_ARTIFACT_TOP_LEVELS: frozenset[str] = frozenset({
    "src", "tests", "dossier", "data", "models", "notebooks", "docs",
})


SHELL_CAPABILITY_NAME: str = "SandboxedShellCapability"
LINTED_ACTION_KINDS: frozenset[str] = frozenset({"write_file", "edit_file"})


@dataclass(frozen=True)
class LintFinding:
    """One Risk #9 hit — a shell-capability mutation against a design-
    artifact path. ``action_record`` is the original dict for context;
    ``reason`` is a short human-readable explanation."""

    capability: str
    action_kind: str
    path: str
    reason: str
    action_record: dict


def _is_design_artifact_path(path: str) -> bool:
    if not path:
        return False
    # Use forward-slash as the canonical separator since session
    # transcripts originate in containers (Linux).
    normalized = path.lstrip("./")
    head = normalized.split("/", 1)[0] if "/" in normalized else normalized
    return head in DESIGN_ARTIFACT_TOP_LEVELS


def lint_session_actions(
    actions: Iterable[dict],
) -> list[LintFinding]:
    """Return Risk #9 findings for ``actions``.

    Each action is a dict with at minimum:

    - ``capability``  — capability class name
    - ``action_kind`` — method name on that capability
    - ``kwargs`` (or ``arguments``) — dict of keyword arguments;
      we look for ``path`` (write_file / edit_file)

    Unknown / malformed entries are skipped — this is a lint pass,
    not a validator. Conservative by design: we only flag what we
    can identify with confidence.
    """
    findings: list[LintFinding] = []
    for record in actions:
        if not isinstance(record, dict):
            continue
        capability = record.get("capability")
        kind = record.get("action_kind") or record.get("action")
        if capability != SHELL_CAPABILITY_NAME or kind not in LINTED_ACTION_KINDS:
            continue
        kwargs = record.get("kwargs") or record.get("arguments") or {}
        if not isinstance(kwargs, dict):
            continue
        path = kwargs.get("path")
        if not isinstance(path, str) or not _is_design_artifact_path(path):
            continue
        findings.append(LintFinding(
            capability=capability,
            action_kind=kind,
            path=path,
            reason=(
                f"{capability}.{kind} targeted design-artifact path "
                f"{path!r}; route design-substance mutations through "
                f"ProjectAuthoringCapability (L1-F) for audit + validation."
            ),
            action_record=record,
        ))
    return findings


__all__ = (
    "DESIGN_ARTIFACT_TOP_LEVELS",
    "LINTED_ACTION_KINDS",
    "LintFinding",
    "SHELL_CAPABILITY_NAME",
    "lint_session_actions",
)
