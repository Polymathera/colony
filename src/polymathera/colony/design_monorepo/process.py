

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

from ..agents.base import Agent
from ..agents.blackboard.protocol import (
    BottleneckDetectedProtocol,
    RoadmapSyncProtocol,
)
from ..agents.patterns.actions import action_executor

from .repo_map import RepoMap, REPO_MAP_DIR, REPO_MAP_FILENAME
from ._internal import (
    SYSDES_KUZU_SCAN_LIMIT,
    DESIGN_CONTEXT_URI_SCHEME,
    SYSDES_MAX_FILES_PER_SOURCE_IN_SUMMARY,
    parse_design_context_uri,
    sysdes_list_files,
    sysdes_peek_headings,
)
from .capabilities import DesignMonorepoCapabilityBase

logger = logging.getLogger(__name__)




# ---------------------------------------------------------------------------
# Roadmap-bootstrap helpers (Phase P5b — bootstrap_roadmap_from_objectives)
# ---------------------------------------------------------------------------
#
# HTML-comment marker the action stamps onto every issue body it
# creates, and the GitHub-inbound sync (Phase P5c) parses to join
# inbound issue state changes back to their ROADMAP.md task entries.
# The format is deliberate:
#
#   <!-- colony:roadmap-task: <stable-id> -->
#
# - ``colony:`` namespace so we don't collide with other tools' HTML
#   comments in the same body.
# - ``roadmap-task`` distinguishes from other future colony markers
#   (e.g. ``colony:author=...`` in P4's comment_as_session_agent).
# - ``<stable-id>`` is the first 12 hex chars of
#   ``sha256(f"{milestone_title}::{task_title}")``, per design doc
#   open question Q4's resolution (user picked content-hash).
#
# These two constants must stay synchronised; the parser regex
# anchors on ``_ROADMAP_TASK_MARKER_PREFIX``.

_ROADMAP_TASK_MARKER_PREFIX = "<!-- colony:roadmap-task:"
_ROADMAP_TASK_MARKER_SUFFIX = "-->"
_ROADMAP_TASK_MARKER_RE = re.compile(
    r"<!--\s*colony:roadmap-task:\s*(?P<stable_id>[0-9a-f]{12})\s*-->",
    re.IGNORECASE,
)

# Optional per-task assignee override (P5d). Either the operator stamps
# this into the roadmap task line / issue body to pre-decide who owns
# the task, or :meth:`DesignProcessCapability.propose_task_assignments`
# falls back to LLM classification. Two values only:
# - ``colony`` — Colony executes the task with its own action surface.
# - ``user``   — the human session author owns the task; Colony cannot
#                or should not execute it (e.g. CAD, lab work).
_ASSIGNEE_MARKER_PREFIX = "<!-- colony:assignee:"
_ASSIGNEE_MARKER_RE = re.compile(
    r"<!--\s*colony:assignee:\s*(?P<assignee>colony|user)\s*-->",
    re.IGNORECASE,
)

ASSIGNEE_TARGETS: tuple[str, str] = ("colony", "user")


def parse_owner_repo_from_url(url: str) -> str | None:
    """Extract ``owner/repo`` from a github.com clone URL.

    Handles the common shapes:

    - ``https://github.com/owner/repo.git``
    - ``https://github.com/owner/repo``
    - ``git@github.com:owner/repo.git``

    Returns ``None`` for non-github URLs (gitlab, internal forges) or
    malformed input — caller surfaces a clean error rather than guess.
    Pure; no IO. Tested in isolation.
    """

    if not url:
        return None
    s = url.strip()
    # SSH form: ``git@github.com:owner/repo[.git]``
    if s.startswith("git@github.com:"):
        path = s[len("git@github.com:"):]
    elif "github.com/" in s:
        path = s.split("github.com/", 1)[1]
    else:
        return None
    if path.endswith(".git"):
        path = path[:-4]
    path = path.strip("/")
    parts = path.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    return f"{parts[0]}/{parts[1]}"


def _stable_task_id(milestone_title: str, task_title: str) -> str:
    """Generate the stable-id stamped into the roadmap-task marker
    (resolution of design-doc Q4). Content-hash so:

    - It survives renaming the milestone or task title — the operator
      keeps the same id by editing the marker manually, which
      :func:`_extract_roadmap_task_marker` can pick up regardless.
    - The same ``(milestone, task)`` pair always hashes to the same
      id across machines, branches, and rebuilds. Makes the
      bidirectional sync (P5c) free of session-coupled id state.
    """

    digest = hashlib.sha256(
        f"{milestone_title}::{task_title}".encode("utf-8"),
    ).hexdigest()
    return digest[:12]


def _extract_roadmap_task_marker(body: str | None) -> str | None:
    """Pull the stable-id out of a GitHub issue body, or ``None`` if
    the body has no marker. Idempotent + tolerant of surrounding
    text — operators may add their own prose around the marker
    without breaking sync.

    Forward-compat for Phase P5c sync (bidirectional reconciliation)
    + Phase P5d's task-assignment proposals (which join via the same
    marker).
    """

    if not body:
        return None
    m = _ROADMAP_TASK_MARKER_RE.search(body)
    return m.group("stable_id") if m else None


def _extract_assignee_marker(body: str | None) -> str | None:
    """Pull the per-task assignee override out of a roadmap task line
    or GitHub issue body. Returns ``"colony"`` / ``"user"`` / ``None``
    (no marker present).

    Used by :meth:`DesignProcessCapability.propose_task_assignments`
    to honour an operator's explicit choice before falling back to
    LLM classification.
    """

    if not body:
        return None
    m = _ASSIGNEE_MARKER_RE.search(body)
    return m.group("assignee").lower() if m else None


# Prompt template the LLM extractor uses to propose a roadmap.
# Kept module-level so it's unit-testable in isolation + the operator
# can patch via monkeypatch in test environments.
_ROADMAP_PROPOSAL_SYSTEM = (
    "You are a project-planning assistant. Given a project's design "
    "context (objectives, constraints, requirements, optional "
    "existing roadmap, and any existing open GitHub issues), produce "
    "a proposed roadmap as a JSON object. The roadmap should be a "
    "reasonable, conservative plan: prefer 2-5 milestones, each "
    "with 2-8 tasks, ordered by dependency / urgency.\n\n"
    "Output ONLY valid JSON, no prose, no code fences. Schema:\n\n"
    "{\n"
    '  "milestones": [\n'
    "    {\n"
    '      "title": "<short milestone title>",\n'
    '      "description": "<1-3 sentence rationale>",\n'
    '      "tasks": [\n'
    "        {\n"
    '          "title": "<short task title>",\n'
    '          "description": "<1-3 sentence scope statement>",\n'
    '          "labels": ["<optional GitHub labels>"]\n'
    "        }\n"
    "      ]\n"
    "    }\n"
    "  ]\n"
    "}\n\n"
    "Constraints on your output:\n"
    "- Every milestone MUST have at least one task.\n"
    "- Task titles MUST be unique within a milestone.\n"
    "- Do NOT duplicate existing open GitHub issues — when an open "
    "issue already covers a task, omit it from the proposal.\n"
    "- Do NOT invent constraints / objectives not present in the "
    "design context. Stay grounded.\n"
)


def _build_roadmap_proposal_prompt(
    *,
    design_context_summary: dict[str, Any],
    existing_roadmap: str,
    roadmap_path: str,
    existing_issues: list[dict[str, Any]],
    max_chars_design_context: int = 12000,
    max_chars_existing_roadmap: int = 4000,
) -> str:
    """Assemble the LLM-facing prompt for roadmap proposal.

    Caps the per-section character count so the prompt fits the
    typical 8-16k context window even for big design corpora; the
    operator can adjust by passing different ``max_chars_*``."""

    # Design-context summary — collapse the per-source file inventory
    # to a compact textual representation. ``summarise_design_context``
    # returns ``{sources: [{name, hint, file_count, files: [{path,
    # size, headings}]}, ...]}``; we keep the path + first heading per
    # file so the LLM sees what's in the corpus without us shovelling
    # whole files in.
    dc_lines: list[str] = ["## Design context\n"]
    for src in design_context_summary.get("sources") or []:
        dc_lines.append(
            f"### Source: {src.get('name')!r} "
            f"(files: {src.get('file_count')})\n",
        )
        if src.get("hint"):
            dc_lines.append(f"Operator hint: {src['hint']}\n")
        for f in src.get("files") or []:
            headings = ", ".join(f.get("headings") or [])
            dc_lines.append(
                f"- `{f.get('path')}`"
                + (f" — headings: {headings}" if headings else "")
                + "\n",
            )
    dc_block = "".join(dc_lines)[:max_chars_design_context]

    roadmap_heading = f"## Existing roadmap (`{roadmap_path}`)"
    roadmap_block = (
        f"{roadmap_heading}\n```\n{existing_roadmap[:max_chars_existing_roadmap]}\n```\n"
        if existing_roadmap
        else f"{roadmap_heading}\n(none — first-time bootstrap)\n"
    )

    issues_lines = ["## Existing open GitHub issues\n"]
    for issue in existing_issues[:50]:
        title = issue.get("title", "")
        number = issue.get("number")
        milestone = issue.get("milestone")
        labels = ", ".join(issue.get("labels") or [])
        issues_lines.append(
            f"- #{number} {title!r}"
            + (f" [milestone: {milestone}]" if milestone else "")
            + (f" [labels: {labels}]" if labels else "")
            + "\n",
        )
    if not existing_issues:
        issues_lines.append("(none)\n")
    issues_block = "".join(issues_lines)

    return (
        f"{_ROADMAP_PROPOSAL_SYSTEM}\n\n"
        f"{dc_block}\n\n"
        f"{roadmap_block}\n\n"
        f"{issues_block}\n\n"
        "Now produce the JSON proposal."
    )


def _parse_roadmap_proposal(raw: str) -> dict[str, Any] | None:
    """Tolerant parser for the LLM's JSON output. Mirrors the
    code-fence stripping in :class:`LLMClaimExtractor._parse`.

    Returns ``None`` (with a logged WARN) on:
    - malformed JSON,
    - JSON that isn't an object,
    - missing ``milestones`` key,
    - ``milestones`` not a list,
    - any milestone missing ``title`` or ``tasks``,
    - any task missing ``title``.
    """

    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = text.removesuffix("```").strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning(
            "bootstrap_roadmap_from_objectives: LLM produced malformed "
            "JSON: %s (first 200 chars: %r)",
            exc, text[:200],
        )
        return None
    if not isinstance(payload, dict):
        return None
    milestones_raw = payload.get("milestones")
    if not isinstance(milestones_raw, list):
        return None

    out_milestones: list[dict[str, Any]] = []
    for m in milestones_raw:
        if not isinstance(m, dict):
            continue
        m_title = str(m.get("title") or "").strip()
        if not m_title:
            continue
        m_desc = str(m.get("description") or "").strip()
        tasks_raw = m.get("tasks")
        if not isinstance(tasks_raw, list) or not tasks_raw:
            continue
        out_tasks: list[dict[str, Any]] = []
        seen_titles: set[str] = set()
        for t in tasks_raw:
            if not isinstance(t, dict):
                continue
            t_title = str(t.get("title") or "").strip()
            if not t_title or t_title in seen_titles:
                continue
            seen_titles.add(t_title)
            t_desc = str(t.get("description") or "").strip()
            labels_raw = t.get("labels") or []
            labels = [
                str(l).strip() for l in labels_raw
                if isinstance(l, (str, int)) and str(l).strip()
            ]
            out_tasks.append({
                "title": t_title,
                "description": t_desc,
                "labels": labels,
                "stable_id": _stable_task_id(m_title, t_title),
            })
        if not out_tasks:
            continue
        out_milestones.append({
            "title": m_title,
            "description": m_desc,
            "tasks": out_tasks,
        })

    if not out_milestones:
        return None
    return {"milestones": out_milestones}


def _build_design_context_summary(
    repo_root: Path, repo_map: RepoMap,
) -> dict[str, Any]:
    """Inline compact summary of design_context_sources — same shape
    :meth:`SystemDesignCapability.summarise_design_context` produces,
    re-implemented here so
    :meth:`DesignProcessCapability.bootstrap_roadmap_from_objectives`
    doesn't need a sibling-capability lookup (which would force the
    agent to mount both capabilities for the bootstrap action to
    work).

    Sync (intended to run inside ``asyncio.to_thread`` because the
    file walks block on disk I/O)."""

    sources_payload: list[dict[str, Any]] = []
    for src in repo_map.design_context_sources:
        matched = sysdes_list_files(repo_root, src)
        files_info: list[dict[str, Any]] = []
        for f in matched[:SYSDES_MAX_FILES_PER_SOURCE_IN_SUMMARY]:
            try:
                rel = f.relative_to(repo_root).as_posix()
            except ValueError:
                rel = str(f)
            files_info.append({
                "path": rel,
                "headings": sysdes_peek_headings(f),
            })
        sources_payload.append({
            "name": src.name,
            "hint": src.hint,
            "pin_in_vcm": src.pin_in_vcm,
            "file_count": len(matched),
            "files": files_info,
        })
    return {"sources": sources_payload}


def _render_issue_body_for_task(
    *, milestone: dict[str, Any], task: dict[str, Any],
) -> str:
    """Render a GitHub issue body for a roadmap task. Stamps the
    ``<!-- colony:roadmap-task: <stable-id> -->`` marker as the
    LAST line so :func:`_extract_roadmap_task_marker` finds it
    reliably even when the operator adds prose above."""

    description = task.get("description") or ""
    stable_id = task["stable_id"]
    marker = (
        f"{_ROADMAP_TASK_MARKER_PREFIX} {stable_id} "
        f"{_ROADMAP_TASK_MARKER_SUFFIX}"
    )
    parts = [
        description,
        "",
        f"_Part of milestone: **{milestone['title']}**_",
        "",
        marker,
    ]
    return "\n".join(parts)


class _CommitFailed(RuntimeError):
    """Local commit step failed — nothing landed on the remote."""


class _PushFailed(RuntimeError):
    """Local commit succeeded but the push to origin did not.

    ``commit_sha`` is populated so the caller can surface the local
    SHA in the response and the operator can recover manually."""

    def __init__(self, message: str, *, commit_sha: str) -> None:
        super().__init__(message)
        self.commit_sha = commit_sha


def _commit_and_push_roadmap_file(
    *, repo_root: Path, roadmap_relpath: str, message: str,
) -> str:
    """Stage, commit, and push the roadmap file on the per-agent clone.
    Returns the new commit SHA.

    Bundles commit + push in a single helper so the two call sites
    (``bootstrap_roadmap_from_objectives``, ``sync_roadmap_with_
    github``) can't drift apart — the roadmap is only useful on the
    remote (the per-agent clone is ephemeral), so a "commit but don't
    push" outcome is a bug, not a valid state.

    Uses ``gitpython`` directly (rather than going through
    ``DesignCheckpointer.fork_design`` → ``merge_design``) because
    the bootstrap is a single conceptual change — the operator
    invoked it explicitly and reviews the resulting commit via
    normal git workflow.

    Raises :class:`_CommitFailed` for staging/commit errors (nothing
    on the remote) and :class:`_PushFailed` (carrying ``commit_sha``)
    for push errors. The two failure modes have different operator
    remediation paths, so the caller surfaces them as distinct
    ``error`` categories in the action return.

    Authentication: pushing to github.com is wired through the
    per-tenant App installation token via
    :func:`ensure_git_credentials_from_agent_metadata` — already
    invoked by ``DesignMonorepoCapabilityBase.initialize()`` for
    non-read-only mounts, so no extra plumbing is needed here.

    Sync (designed for ``asyncio.to_thread``).
    """

    import git

    try:
        repo = git.Repo(str(repo_root))
        repo.index.add([roadmap_relpath])
        if not repo.is_dirty(index=True, working_tree=False):
            # Nothing staged → no new commit. Still push HEAD so a
            # previous in-clone commit that never reached the remote
            # gets a second chance. ``commit_sha`` falls back to HEAD.
            commit_sha = repo.head.commit.hexsha
        else:
            commit_sha = repo.index.commit(message).hexsha
    except Exception as exc:  # noqa: BLE001
        raise _CommitFailed(str(exc)) from exc

    try:
        try:
            branch_name = repo.active_branch.name
        except TypeError as exc:
            # Detached HEAD — cannot push a symbolic ref.
            raise _PushFailed(
                "HEAD is detached; cannot push the roadmap commit. "
                "Switch to a branch and re-run.",
                commit_sha=commit_sha,
            ) from exc

        try:
            origin = repo.remote("origin")
        except ValueError as exc:
            raise _PushFailed(
                f"no ``origin`` remote configured on {repo_root}: {exc}",
                commit_sha=commit_sha,
            ) from exc

        push_info = origin.push(
            refspec=f"HEAD:refs/heads/{branch_name}",
        )
        # ``gitpython`` returns a list of PushInfo; any ERROR or
        # REJECTED flag means the push didn't land even when the call
        # returned normally (no Python exception).
        from git import PushInfo
        bad = [
            pi for pi in push_info
            if pi.flags & (PushInfo.ERROR | PushInfo.REJECTED)
        ]
        if bad:
            raise _PushFailed(
                "; ".join(
                    pi.summary.strip() or repr(pi.flags) for pi in bad
                ),
                commit_sha=commit_sha,
            )
    except _PushFailed:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _PushFailed(str(exc), commit_sha=commit_sha) from exc

    return commit_sha


def _render_roadmap_markdown(proposal: dict[str, Any]) -> str:
    """Render a parsed proposal as ROADMAP.md markdown.

    Each task line ends with the same stable-id marker stamped into
    the GitHub issue body so the operator (and the future P5c sync)
    can match roadmap rows to issues at a glance.
    """

    lines: list[str] = [
        "# Roadmap",
        "",
        "_Generated by `DesignProcessCapability.bootstrap_roadmap_from_objectives`. "
        "Edit freely; the next `sync_roadmap_with_github` (P5c) "
        "reconciles changes with the GitHub issue tracker._",
        "",
    ]
    for m in proposal.get("milestones") or []:
        lines.append(f"## {m['title']}")
        lines.append("")
        if m.get("description"):
            lines.append(m["description"])
            lines.append("")
        for t in m.get("tasks") or []:
            stable_id = t.get("stable_id") or _stable_task_id(
                m["title"], t["title"],
            )
            marker = (
                f"{_ROADMAP_TASK_MARKER_PREFIX} {stable_id} "
                f"{_ROADMAP_TASK_MARKER_SUFFIX}"
            )
            lines.append(f"- [ ] **{t['title']}** {marker}")
            if t.get("description"):
                lines.append(f"      {t['description']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# Sentinel milestone the sync action appends ``github_only`` tasks to
# when the issue exists with a marker but ROADMAP.md has no matching
# task. Lets the operator move it to the right milestone manually.
_UNTRACKED_MILESTONE_TITLE = "Untracked"


# Task line: ``- [ ] **title** <!-- colony:roadmap-task: <id> -->``
# Anchors on the bold title + marker; tolerant of leading spaces.
_ROADMAP_TASK_LINE_RE = re.compile(
    r"^\s*-\s*\[[ x]\]\s*\*\*(?P<title>.+?)\*\*\s*"
    r"<!--\s*colony:roadmap-task:\s*(?P<stable_id>[0-9a-f]{12})\s*-->",
)


def _parse_roadmap_markdown(text: str) -> dict[str, Any]:
    """Reverse of :func:`_render_roadmap_markdown` — parse the rendered
    ROADMAP.md back into the structured ``{milestones: [{title,
    description, tasks: [{title, stable_id, description?, labels?}]}]}``
    shape so :meth:`DesignProcessCapability.sync_roadmap_with_github`
    can diff it against GitHub issues.

    Tolerant of operator edits: lines without the marker are skipped
    (the marker is the join key with GitHub), the description line
    after a task is collected when present (indented 4+ spaces or
    starting with ``      ``), milestone descriptions are collected
    until the first task line. Lines outside any ``## milestone``
    header (e.g. the preamble paragraph the renderer writes) are
    ignored.

    Returns ``{"milestones": []}`` for an empty / missing roadmap so
    the diff downstream is symmetric with the renderer's input shape.
    """

    milestones: list[dict[str, Any]] = []
    current_milestone: dict[str, Any] | None = None
    current_task: dict[str, Any] | None = None
    in_preamble = True
    if not text:
        return {"milestones": milestones}

    for raw_line in text.splitlines():
        # Strip a single trailing newline if present; don't strip
        # leading spaces because indentation distinguishes
        # task-description continuations from milestone prose.
        line = raw_line.rstrip("\n")

        # New milestone header: ``## <title>``
        if line.startswith("## "):
            current_milestone = {
                "title": line[3:].strip(),
                "description": "",
                "tasks": [],
            }
            milestones.append(current_milestone)
            current_task = None
            in_preamble = False
            continue

        # Skip the top-level ``# Roadmap`` header.
        if line.startswith("# "):
            in_preamble = True
            continue

        # Task line — anchored on the marker.
        m = _ROADMAP_TASK_LINE_RE.match(line)
        if m:
            if current_milestone is None:
                # A task line outside any milestone — synthesise an
                # "Untracked" milestone so we don't drop it.
                current_milestone = {
                    "title": _UNTRACKED_MILESTONE_TITLE,
                    "description": "",
                    "tasks": [],
                }
                milestones.append(current_milestone)
            current_task = {
                "title": m.group("title").strip(),
                "stable_id": m.group("stable_id"),
                "description": "",
                "labels": [],
            }
            current_milestone["tasks"].append(current_task)
            continue

        # Task description continuation: indented under the task line
        # (renderer uses 6-space indent; accept 2+ to be lenient).
        if (
            current_task is not None
            and (line.startswith("      ") or line.startswith("\t"))
            and line.strip()
        ):
            existing = current_task["description"]
            stripped = line.strip()
            current_task["description"] = (
                f"{existing} {stripped}" if existing else stripped
            )
            continue

        # Blank line breaks the task continuation but keeps the
        # current milestone live.
        if not line.strip():
            current_task = None
            continue

        # Milestone description (between header and first task).
        if (
            current_milestone is not None
            and current_task is None
            and not in_preamble
        ):
            existing = current_milestone["description"]
            stripped = line.strip()
            current_milestone["description"] = (
                f"{existing} {stripped}" if existing else stripped
            )

    return {"milestones": milestones}


def _diff_roadmap_vs_issues(
    roadmap: dict[str, Any],
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute the four-bucket diff for
    :meth:`DesignProcessCapability.sync_roadmap_with_github`.

    Buckets:

    - ``roadmap_only`` — tasks in ROADMAP.md whose ``stable_id`` does
      NOT appear as a marker on any issue. Apply action: create
      issue with the marker stamped.
    - ``github_only`` — issues whose body marker's ``stable_id`` does
      NOT appear in ROADMAP.md. Apply action: append to the
      ``Untracked`` milestone in ROADMAP.md.
    - ``divergent`` — both sides have an entry for the ``stable_id``
      but the titles differ. Apply action: surface the conflict; do
      NOT auto-resolve.
    - ``in_sync`` — both sides match. Apply action: nothing.

    Plus ``untracked_issues`` — issues with no marker at all (e.g.
    operator created them outside the bootstrap flow). Surfaced for
    the planner's awareness; the sync action does NOT auto-import
    these (operator chooses what to roadmap).
    """

    # Index roadmap tasks by stable_id, carrying the milestone context.
    roadmap_index: dict[str, dict[str, Any]] = {}
    for m in roadmap.get("milestones") or []:
        for t in m.get("tasks") or []:
            sid = t.get("stable_id")
            if not sid:
                continue
            roadmap_index[sid] = {
                "milestone_title": m.get("title", ""),
                "title": t.get("title", ""),
                "description": t.get("description", ""),
                "labels": t.get("labels") or [],
                "stable_id": sid,
            }

    # Index issues by marker stable_id.
    issues_with_marker: dict[str, dict[str, Any]] = {}
    issues_without_marker: list[dict[str, Any]] = []
    for issue in issues:
        sid = _extract_roadmap_task_marker(issue.get("body"))
        if sid:
            issues_with_marker[sid] = issue
        else:
            issues_without_marker.append(issue)

    roadmap_ids = set(roadmap_index.keys())
    issue_ids = set(issues_with_marker.keys())

    roadmap_only = [
        roadmap_index[sid] for sid in sorted(roadmap_ids - issue_ids)
    ]
    github_only = [
        {
            "stable_id": sid,
            "title": issues_with_marker[sid].get("title", ""),
            "issue_number": issues_with_marker[sid].get("number"),
            "issue_state": issues_with_marker[sid].get("state"),
            "issue_url": issues_with_marker[sid].get("url"),
        }
        for sid in sorted(issue_ids - roadmap_ids)
    ]
    divergent: list[dict[str, Any]] = []
    in_sync: list[dict[str, Any]] = []
    for sid in sorted(roadmap_ids & issue_ids):
        rd = roadmap_index[sid]
        iss = issues_with_marker[sid]
        rd_title = rd["title"].strip()
        iss_title = (iss.get("title") or "").strip()
        if rd_title == iss_title:
            in_sync.append({
                "stable_id": sid,
                "title": rd_title,
                "issue_number": iss.get("number"),
                "issue_state": iss.get("state"),
            })
        else:
            divergent.append({
                "stable_id": sid,
                "roadmap_title": rd_title,
                "issue_title": iss_title,
                "issue_number": iss.get("number"),
                "issue_state": iss.get("state"),
                "issue_url": iss.get("url"),
                "milestone_title": rd["milestone_title"],
            })

    untracked_issues = [
        {
            "number": iss.get("number"),
            "title": iss.get("title"),
            "state": iss.get("state"),
            "url": iss.get("url"),
        }
        for iss in issues_without_marker
    ]

    return {
        "roadmap_only": roadmap_only,
        "github_only": github_only,
        "divergent": divergent,
        "in_sync": in_sync,
        "untracked_issues": untracked_issues,
    }


def _sync_stats(diff: dict[str, Any]) -> dict[str, int]:
    """One-liner stats summary derived from
    :func:`_diff_roadmap_vs_issues` for the action's response."""

    return {
        "roadmap_only_count": len(diff.get("roadmap_only") or []),
        "github_only_count": len(diff.get("github_only") or []),
        "divergent_count": len(diff.get("divergent") or []),
        "in_sync_count": len(diff.get("in_sync") or []),
        "untracked_issue_count": len(diff.get("untracked_issues") or []),
    }


def _merge_github_only_into_roadmap(
    roadmap: dict[str, Any],
    github_only: list[dict[str, Any]],
) -> dict[str, Any]:
    """Append ``github_only`` entries to the ``Untracked`` milestone in
    ``roadmap`` (creating the milestone if needed) so the operator can
    later move them to the right milestone manually.

    Pure function — returns a NEW dict; ``roadmap`` is not mutated."""

    if not github_only:
        return roadmap

    # Deep-copy the milestone list so the caller's parsed roadmap
    # stays unchanged (tests rely on this).
    out_milestones = [
        {
            "title": m.get("title", ""),
            "description": m.get("description", ""),
            "tasks": list(m.get("tasks") or []),
        }
        for m in (roadmap.get("milestones") or [])
    ]

    # Find or create the untracked bucket.
    untracked = next(
        (m for m in out_milestones
         if m["title"] == _UNTRACKED_MILESTONE_TITLE),
        None,
    )
    if untracked is None:
        untracked = {
            "title": _UNTRACKED_MILESTONE_TITLE,
            "description": (
                "Imported from GitHub issues that had a roadmap-task "
                "marker but no matching task in ROADMAP.md. Move "
                "these to the right milestone manually + re-run sync."
            ),
            "tasks": [],
        }
        out_milestones.append(untracked)

    for entry in github_only:
        untracked["tasks"].append({
            "title": entry["title"],
            "description": (
                f"Imported from issue #{entry['issue_number']} "
                f"(state: {entry['issue_state']!s})."
            ),
            "labels": [],
            "stable_id": entry["stable_id"],
        })

    return {"milestones": out_milestones}


# ---------------------------------------------------------------------------
# P5d: task-assignment classification
# ---------------------------------------------------------------------------
#
# The assignment universe is intentionally binary:
#
# - ``colony`` — the task is something Colony can execute end-to-end
#   with its own action surface (planning, design-context analysis,
#   roadmap maintenance, KG queries, code edits routed through the
#   blackboard, etc.).
# - ``user``   — the task requires human skills, tools, or judgement
#   Colony does NOT have access to (CAD, lab measurements, vendor
#   procurement, physical assembly, irreversible business calls).
#   The user can then reassign their tasks to teammates outside the
#   scope of Colony — Colony tracks only the single "human" identity
#   it commits + comments as.

_ASSIGNMENT_PROMPT_SYSTEM = (
    "You are an assignment router for a project-planning system. "
    "Each task is owned by either:\n"
    "  - colony — an autonomous AI agent that can plan, query design "
    "context (objectives / constraints / requirements / hypotheses), "
    "ingest documents, maintain a roadmap, query and edit the "
    "knowledge graph, draft pull requests, post GitHub comments, run "
    "scripted analyses, and reason over the project's design context;\n"
    "  - user   — the human collaborator. They own tasks that need "
    "skills Colony does NOT have access to: CAD work (SolidWorks, "
    "FreeCAD), hardware assembly, lab measurements, vendor calls / "
    "procurement, photographs / videos / sketches, irreversible "
    "business decisions, regulatory submissions, and any task whose "
    "judgement must come from a human who knows the project.\n\n"
    "Given a single roadmap task (title + description), classify it. "
    "Output ONLY valid JSON, no prose, no code fences. Schema:\n\n"
    "{\n"
    '  "assignee": "colony" | "user",\n'
    '  "reason":   "<one short sentence — why this side>"\n'
    "}\n\n"
    "Rules:\n"
    "- If the task primarily needs analysis, search, drafting, or "
    "reasoning over design context — assign to colony.\n"
    "- If the task primarily needs hands, eyes, hardware, vendors, "
    "or human judgement — assign to user.\n"
    "- When in doubt between the two, prefer user — Colony will "
    "evaluate the user's work rather than do it itself.\n"
)


def _build_assignment_classification_prompt(
    *, milestone_title: str, task_title: str, task_description: str,
) -> str:
    """Render the prompt :meth:`DesignProcessCapability.propose_task_assignments`
    sends to the LLM for one task. Kept module-level so it's unit-
    testable in isolation."""

    return (
        f"{_ASSIGNMENT_PROMPT_SYSTEM}\n"
        "---\n"
        f"Milestone: {milestone_title}\n"
        f"Task: {task_title}\n\n"
        f"Description:\n{task_description or '(no description)'}\n"
        "---\n"
        "Return the JSON object."
    )


def _parse_assignment_classification(raw: str | None) -> dict[str, str] | None:
    """Parse the LLM output for a single classification.

    Returns ``{assignee, reason}`` on success, ``None`` on any parse
    failure (caller surfaces "llm_parse_failed" + falls back to
    leaving the task unassigned).
    """

    if not raw:
        return None
    text = raw.strip()
    # Strip optional ``` fences (the LLM sometimes ignores instructions).
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    assignee = obj.get("assignee")
    if assignee not in ASSIGNEE_TARGETS:
        return None
    reason = obj.get("reason")
    if not isinstance(reason, str):
        reason = ""
    return {"assignee": assignee, "reason": reason.strip()}





# ---------------------------------------------------------------------------
# DesignProcessCapability — workflow orchestration (top-level design plan §8)
# ---------------------------------------------------------------------------
#
# Co-located with the other design-monorepo capabilities (not in
# ``agents/patterns/capabilities/``) for the same reason as
# :class:`SystemDesignCapability`: inheriting
# :class:`DesignMonorepoCapabilityBase` would cycle via the
# human_approval import that ``design_monorepo.capabilities`` already
# pulls in.

# Heuristic defaults for ``identify_bottlenecks``. Operators override
# per-call; the markdown-rule discovery path (``bottleneck_rule``-typed
# claims extracted by P3d's LLMClaimExtractor) supplies the richer
# operator-defined rules atop these built-ins.
_DEFAULT_STALLED_ISSUE_NO_ACTIVITY_DAYS = 14
_DEFAULT_BOTTLENECK_RULE_PREDICATES = frozenset(
    {"defines_bottleneck_rule"},
)


class DesignProcessCapability(DesignMonorepoCapabilityBase):
    """Workflow orchestration over the design-process state — the
    *how* of the team's work, complementing :class:`SystemDesignCapability`
    which owns the *what*.

    Phase P5a ships three actions:

    - :meth:`load_design_context` — thin wrapper around the shared
      ``_load_design_context_impl``. Owns the design-context
      materialisation lifecycle on the ``DesignProcess`` side per
      top-level design plan §13.
    - :meth:`summarise_progress` — milestone-level progress snapshot
      from ``GitHubCapability.list_milestones`` + optional design-
      context KG roll-up.
    - :meth:`identify_bottlenecks` — built-in stalled-issue heuristic
      over ``GitHubCapability.list_issues`` + discovery of operator-
      authored ``bottleneck_rule`` claims (P3d-extracted). Emits one
      :class:`BottleneckDetectedProtocol` per finding.

    Phase P5b adds ``bootstrap_roadmap_from_objectives``; P5c adds
    ``sync_roadmap_with_github``; P5d adds ``propose_task_assignments``
    (binary colony-vs-user routing via marker override + LLM fallback).

    Pure action surface — no ``@event_handler`` methods; passes
    ``input_patterns=[]`` to opt out of the base class's wildcard
    fallback (same discipline as
    :class:`RepoStateProvider` / :class:`SystemDesignCapability`).
    """

    def __init__(
        self,
        agent: Agent | None = None,
        scope_id: str | None = None,
        *,
        working_dir: Path | str | None = None,
        clone_scope_id: str | None = None,
        read_only: bool = True,
        capability_key: str | None = None,
        app_name: str | None = None,
    ) -> None:
        super().__init__(
            agent=agent,
            scope_id=scope_id,
            working_dir=working_dir,
            clone_scope_id=clone_scope_id,
            read_only=read_only,
            input_patterns=[],
            capability_key=capability_key,
            app_name=app_name,
        )

    def get_capability_tags(self) -> frozenset[str]:
        return frozenset({"design_process", "workflow", "github"})

    def _resolve_github_repo(self) -> str | None:
        """Derive ``owner/repo`` from the agent's configured
        ``design_monorepo_url``. Returns ``None`` when:

        - the capability is detached (no agent);
        - the agent's metadata carries no ``design_monorepo_url`` (no
          monorepo configured for the colony);
        - the URL is non-github (gitlab, internal forge) or malformed.

        Callers in this class use this instead of accepting a ``repo``
        kwarg from the LLM planner — the design monorepo's GitHub
        identity is something the capability already knows, and
        burdening the planner with it just invites hallucinated values.
        """
        url = self.design_monorepo_url
        if not url:
            return None
        return parse_owner_repo_from_url(url)

    def _no_github_repo_error(self, **extras: Any) -> dict[str, Any]:
        """Standard error dict every action returns when
        :meth:`_resolve_github_repo` can't produce a GitHub
        ``owner/repo``. Single source of truth for the message so the
        planner sees a consistent "missing prerequisite" signal across
        every roadmap/issue action."""

        return {
            "error": "no_github_repo",
            "message": (
                "This action operates on the colony's design monorepo "
                "on github.com, but the colony either has no design "
                "monorepo configured or its URL is not on github.com. "
                "Configure ``design_monorepo_url`` on the Colonies "
                "panel (must be a github.com clone URL)."
            ),
            **extras,
        }

    def _resolve_roadmap_path(self, repo_map: RepoMap) -> str | None:
        """Return the configured roadmap path from the
        ``is_roadmap=true`` row in ``design_context_sources``, or
        ``None`` when no such row exists.

        The roadmap's location is operator configuration (declared
        once in ``.colony/repo_map.yaml``), not something the LLM
        planner specifies per-call — the planner would otherwise
        hallucinate plausible-but-wrong paths. The schema enforces
        at-most-one ``is_roadmap=true`` row + a single non-glob path
        (see :class:`DesignContextSource._check_paths` /
        :meth:`RepoMap._check_design_context_sources_unique`), so the
        return value is unambiguous when present.
        """

        for src in repo_map.design_context_sources:
            if src.is_roadmap:
                return src.paths[0]
        return None

    def _no_roadmap_declared_error(self, **extras: Any) -> dict[str, Any]:
        """Standard error dict for actions that require a roadmap row
        but the operator hasn't declared one. Single source of truth
        for the message so the planner sees a consistent missing-
        prerequisite signal."""

        return {
            "error": "no_roadmap_declared",
            "message": (
                "No design_context_sources row has is_roadmap=true in "
                f"{REPO_MAP_DIR}/{REPO_MAP_FILENAME}. Declare one row "
                "with a single literal path (e.g. ``is_roadmap: true, "
                "paths: ['ROADMAP.md']``) so this action knows where "
                "to read/write the roadmap."
            ),
            **extras,
        }

    @action_executor(
        planning_summary=(
            "Refresh the design context (re-read repo_map.yaml + re-"
            "materialise design_context_sources through paths 1 + 2). "
            "Idempotent; safe to call at session start."
        ),
    )
    async def load_design_context(
        self, *, refresh: bool = True, include_kuzu: bool = True,
    ) -> dict[str, Any]:
        """Refresh the design-context view: re-read ``repo_map.yaml``,
        re-materialise the design_context_sources rows through paths
        1 (Kuzu KG, when ``include_kuzu=True``) and 2 (VCM chunked
        pages — always). Idempotent.

        Top-level design plan §13's rationale for this action living
        on ``DesignProcessCapability`` (vs ``SessionOrchestratorCapability``):
        it has no session-specific state — runs the same regardless of
        which agent invokes it — so it belongs in the workflow-
        orchestration capability, not the session orchestrator. The
        session orchestrator's ``summarise_current_state`` /
        ``propose_next_steps`` actions DO read session-local state
        (the agent's own consciousness streams) and stay there.

        Returns the same response shape as
        :meth:`RepoStateProvider.materialize_design_context` —
        ``{mapped, pinned, ingested, total_claims, failed, count, rows}`` —
        because both call the same
        ``DesignMonorepoCapabilityBase._load_design_context_impl``
        body under the hood. Subscribers of
        :class:`DesignContextMappedProtocol` see one event per
        (source, path) tuple regardless of which action triggered
        the run.
        """

        return await self._load_design_context_impl(
            refresh=refresh, include_kuzu=include_kuzu,
        )

    @action_executor(
        planning_summary=(
            "Roadmap-level progress snapshot — milestones (open/closed "
            "issue counts), optional design-context KG roll-up. Used by "
            "the Colony Status panel (§15) for the 'current milestone' "
            "tile."
        ),
    )
    async def summarise_progress(
        self,
        *,
        milestone_state: str = "open",
        max_milestones: int = 50,
    ) -> dict[str, Any]:
        """Return a structured progress snapshot.

        Phase P5a sources milestone data from
        ``GitHubCapability.list_milestones`` (each milestone carries
        ``open_issues`` + ``closed_issues`` counts from the API
        directly — no per-milestone issue scan). For each milestone:

        - ``progress_pct`` = ``closed_issues / (open_issues + closed_issues)``
          (``None`` when the milestone has zero issues — distinguishes
          "not started" from "100% done with 0 of 0").
        - ``current_milestone`` is the soonest-due open milestone
          (or the first one when no due dates set).

        ``milestone_state`` filters which milestones the snapshot
        considers (``open`` / ``closed`` / ``all``).

        The GitHub repo is derived from the colony's
        ``design_monorepo_url`` (the capability already knows it);
        callers do not pass it. Requires a sibling ``GitHubCapability``
        on the same agent.
        """

        # Empty-totals shape — kept consistent across the no-sibling /
        # api-failure / no-milestones branches so the planner can
        # always read ``totals["milestone_count"]`` without branching.
        _empty_totals = {
            "open_issues": 0,
            "closed_issues": 0,
            "total_issues": 0,
            "milestone_count": 0,
        }

        github = self._sibling_github_capability()
        if github is None:
            return {
                "current_milestone": None,
                "milestones": [],
                "totals": dict(_empty_totals),
                "message": (
                    "No GitHubCapability sibling found on the agent; "
                    "mount one alongside DesignProcessCapability so "
                    "summarise_progress can call list_milestones."
                ),
                "error": "github_capability_missing",
            }

        repo = self._resolve_github_repo()
        if repo is None:
            return {
                "current_milestone": None,
                "milestones": [],
                "totals": dict(_empty_totals),
                **self._no_github_repo_error(),
            }

        result = await github.list_milestones(
            repo=repo, state=milestone_state, max_results=max_milestones,
        )
        if not result.get("ok"):
            return {
                "current_milestone": None,
                "milestones": [],
                "totals": dict(_empty_totals),
                "message": result.get("message", ""),
                "error": "list_milestones_failed",
            }
        raw_milestones = result.get("milestones") or []

        enriched: list[dict[str, Any]] = []
        total_open = 0
        total_closed = 0
        for m in raw_milestones:
            open_n = int(m.get("open_issues") or 0)
            closed_n = int(m.get("closed_issues") or 0)
            total_open += open_n
            total_closed += closed_n
            total_issues = open_n + closed_n
            progress_pct = (
                None if total_issues == 0
                else round(100.0 * closed_n / total_issues, 1)
            )
            enriched.append({
                **m,
                "total_issues": total_issues,
                "progress_pct": progress_pct,
            })

        # ``current_milestone`` — first non-zero open milestone by
        # due_on (ascending); falls back to the first item if no
        # due_on is set on any open milestone.
        open_milestones = [
            m for m in enriched if m["state"] == "open"
        ]
        with_due = [m for m in open_milestones if m.get("due_on")]
        if with_due:
            current = min(with_due, key=lambda m: m["due_on"])
        elif open_milestones:
            current = open_milestones[0]
        else:
            current = None

        return {
            "current_milestone": current,
            "milestones": enriched,
            "totals": {
                "open_issues": total_open,
                "closed_issues": total_closed,
                "total_issues": total_open + total_closed,
                "milestone_count": len(enriched),
            },
            "message": "",
            "error": "",
        }

    @action_executor(
        planning_summary=(
            "Identify workflow bottlenecks — stalled open issues + "
            "operator-authored bottleneck_rule claims. Emits one "
            "BottleneckDetectedProtocol per finding."
        ),
    )
    async def identify_bottlenecks(
        self,
        *,
        stalled_no_activity_days: int = (
            _DEFAULT_STALLED_ISSUE_NO_ACTIVITY_DAYS
        ),
        bottleneck_rule_predicates: list[str] | None = None,
        emit_blackboard_events: bool = True,
        max_issues_scanned: int = 200,
    ) -> dict[str, Any]:
        """Identify workflow bottlenecks.

        Two surfaces:

        - **Built-in stalled-issue heuristic** — fetches open issues
          via ``GitHubCapability.list_issues`` (capped at
          ``max_issues_scanned``) and flags any with no activity for
          longer than ``stalled_no_activity_days`` (default 14).
          GitHub's ``updated_at`` bumps on comments / label changes
          / commit references / state transitions, so "no
          ``updated_at`` change" is a reasonable proxy for "nobody's
          touched this in a while".
        - **Discovery of operator-authored bottleneck rules** —
          surfaces claims whose ``predicate`` matches
          ``bottleneck_rule_predicates`` (default:
          ``{defines_bottleneck_rule}``) OR the
          ``X is_a "bottleneck rule"`` idiom. The actual evaluation
          of these rules waits for richer agent logic (P5b+) /
          LLM-based rule application; in P5a they're surfaced so
          the planner can decide what to do with them.

        ``emit_blackboard_events=True`` (default) writes a
        :class:`BottleneckDetectedProtocol` per stalled issue so the
        Colony Status panel can react.
        """

        import time as _time
        from datetime import datetime, timezone

        # Repo is derived from the colony's design monorepo URL (the
        # capability already knows it); the LLM planner is not asked
        # to specify it.
        resolved_repo = self._resolve_github_repo() or ""

        # --- Built-in heuristic: stalled open issues -----------------
        github = self._sibling_github_capability()
        stalled: list[dict[str, Any]] = []
        if github is not None and resolved_repo:
            issues_result = await github.list_issues(
                repo=resolved_repo, state="open",
                max_results=max_issues_scanned,
            )
            if issues_result.get("ok"):
                threshold_s = stalled_no_activity_days * 86400.0
                now_ts = _time.time()
                for issue in issues_result.get("issues") or []:
                    updated_at = issue.get("updated_at")
                    if not updated_at:
                        continue
                    try:
                        updated_ts = datetime.fromisoformat(
                            updated_at.replace("Z", "+00:00"),
                        ).astimezone(timezone.utc).timestamp()
                    except (ValueError, TypeError):
                        continue
                    age_s = now_ts - updated_ts
                    if age_s < threshold_s:
                        continue
                    stalled.append({
                        "kind": "stalled_issue",
                        "severity": (
                            "high" if age_s > 2 * threshold_s
                            else "medium"
                        ),
                        "repo": resolved_repo,
                        "issue_number": issue.get("number"),
                        "title": issue.get("title"),
                        "url": issue.get("url"),
                        "updated_at": updated_at,
                        "stale_days": round(age_s / 86400.0, 1),
                        "summary": (
                            f"Issue #{issue.get('number')} "
                            f"({issue.get('title')!r}) has had no "
                            f"activity for "
                            f"{round(age_s / 86400.0, 1)} days "
                            f"(threshold {stalled_no_activity_days})."
                        ),
                        "suggested_remedies": [
                            "ping the assignee",
                            "comment with a status check",
                            "demote or close if no longer relevant",
                        ],
                    })

        # --- Discovery of operator-authored bottleneck rules ---------
        rule_set = set(
            bottleneck_rule_predicates
            or _DEFAULT_BOTTLENECK_RULE_PREDICATES,
        )
        rules_discovered = await self._discover_bottleneck_rules(
            rule_set=rule_set,
        )

        # --- Blackboard events --------------------------------------
        if emit_blackboard_events and stalled and resolved_repo:
            blackboard = await self._get_colony_blackboard()
            now = _time.time()
            millis_base = int(now * 1000)
            for idx, finding in enumerate(stalled):
                key = BottleneckDetectedProtocol.event_key(
                    repo=resolved_repo,
                    kind="stalled_issue",
                    millis=millis_base + idx,
                )
                await blackboard.write(
                    key=key,
                    value={**finding, "detected_at": now},
                    created_by=(
                        f"design_process.identify_bottlenecks:"
                        f"{self.capability_key}"
                    ),
                    tags={
                        "design_process",
                        "bottleneck",
                        "stalled_issue",
                        finding["severity"],
                    },
                )

        return {
            "stalled_issues": stalled,
            "rules_discovered": rules_discovered,
            "stats": {
                "stalled_count": len(stalled),
                "rule_count": len(rules_discovered),
                "github_capability_available": github is not None,
                "stalled_no_activity_days": stalled_no_activity_days,
            },
            "error": "" if github is not None else "github_capability_missing",
        }

    @action_executor(
        planning_summary=(
            "Bootstrap (or revise) the design roadmap: read design "
            "context, LLM-propose milestones + tasks, optionally apply "
            "by writing ROADMAP.md + creating one GitHub issue per "
            "task (each stamped with a stable-id marker for sync). "
            "Default ``dry_run=True`` returns the proposal without "
            "writing — operator reviews + re-calls with ``dry_run=False``."
        ),
    )
    async def bootstrap_roadmap_from_objectives(
        self,
        *,
        target_project_id: str | None = None,
        dry_run: bool = True,
        llm_max_tokens: int = 4096,
        llm_temperature: float = 0.2,
        llm_timeout_s: float = 60.0,
        commit_message: str | None = None,
    ) -> dict[str, Any]:
        """Propose (and optionally apply) a roadmap derived from the
        design context.

        Flow:

        1. Load ``repo_map.yaml``; iterate ``design_context_sources``
           rows. Build a compact summary (file paths + first H1/H2
           headings; bounded so the LLM prompt fits) — same shape
           :meth:`SystemDesignCapability.summarise_design_context`
           returns, inlined here to keep this action self-contained.
        2. Read the existing ``roadmap_path`` (if any) so the LLM can
           treat the bootstrap as an incremental revision rather than
           a from-scratch generation when a roadmap already exists.
        3. List the operator's open GitHub issues so the LLM can
           avoid proposing duplicates of work already tracked. Falls
           back to an empty list (with a warning logged) when no
           sibling :class:`GitHubCapability` is mounted.
        4. Call the cluster's LLM (same path P3d's :class:`LLMClaimExtractor`
           uses) with a structured-JSON prompt. The deterministic
           parser (:func:`_parse_roadmap_proposal`) tolerates code
           fences + validates the shape.
        5. Assign a stable id to every task
           (:func:`_stable_task_id` — content-hash, per design-doc Q4).

        With ``dry_run=True`` (the default), returns the proposal
        without touching git or GitHub — the operator reviews the
        proposed roadmap + issue list, then re-calls with
        ``dry_run=False`` to apply.

        With ``dry_run=False``:

        6. Write the rendered ROADMAP.md to ``roadmap_path``
           (creating parent dirs if needed) and commit it.
        7. Create one GitHub issue per task via the sibling
           :class:`GitHubCapability`'s ``create_issue`` (P4) — body
           stamped with the marker
           ``<!-- colony:roadmap-task: <stable-id> -->`` so P5c's
           bidirectional sync can join issues to roadmap rows.
           Auto-attaches to ``target_project_id`` (or the
           capability's ``default_project_id``) when present.

        Returns ``{dry_run, proposal, ...}`` always; when
        ``dry_run=False`` also includes ``roadmap_written``,
        ``commit_sha``, ``issues_created``, ``issue_failures``.
        """

        # ``repo`` is derived later, only when a GitHub sibling is
        # mounted (the local roadmap write succeeds without a sibling;
        # only the issue-creation pass needs the GitHub repo).

        # ---------- step 1: design-context summary -------------------
        repo_root = self._working_dir
        if not (repo_root / ".git").is_dir():
            self._lazy_clone_from_agent_metadata()
        if not (repo_root / ".git").is_dir():
            return {
                "dry_run": dry_run,
                "proposal": None,
                "error": "no_design_monorepo",
                "message": (
                    f"{repo_root} is not a git repository — set the "
                    "colony's design-monorepo URL or run "
                    "``initialize_repo_map`` first."
                ),
            }

        repo_map = await asyncio.to_thread(RepoMap.load, repo_root)
        if not repo_map.design_context_sources:
            return {
                "dry_run": dry_run,
                "proposal": None,
                "error": "no_design_context_sources",
                "message": (
                    f"No ``design_context_sources:`` rows declared in "
                    f"{REPO_MAP_DIR}/{REPO_MAP_FILENAME}. Bootstrap "
                    "requires at least one design-context source so "
                    "the LLM has objectives / constraints / "
                    "requirements to propose against."
                ),
            }
        roadmap_path = self._resolve_roadmap_path(repo_map)
        if roadmap_path is None:
            return {
                "dry_run": dry_run,
                "proposal": None,
                **self._no_roadmap_declared_error(),
            }
        design_context_summary = await asyncio.to_thread(
            _build_design_context_summary, repo_root, repo_map,
        )

        # ---------- step 2: existing roadmap -------------------------
        roadmap_file = repo_root / roadmap_path
        existing_roadmap = ""
        if roadmap_file.is_file():
            try:
                existing_roadmap = roadmap_file.read_text(encoding="utf-8")
            except OSError as exc:
                logger.warning(
                    "bootstrap_roadmap_from_objectives: failed to read "
                    "existing roadmap at %s: %s — treating as empty.",
                    roadmap_file, exc,
                )

        # ---------- step 3: open GitHub issues -----------------------
        # Resolve repo only when a sibling is mounted (the local-only
        # path doesn't need it). If sibling is mounted but no repo can
        # be derived → clean error before any github calls.
        github = self._sibling_github_capability()
        repo: str | None = None
        if github is not None:
            repo = self._resolve_github_repo()
            if repo is None:
                return {
                    "dry_run": dry_run,
                    "proposal": None,
                    **self._no_github_repo_error(),
                }
        existing_issues: list[dict[str, Any]] = []
        if github is not None:
            issues_result = await github.list_issues(
                repo=repo, state="open", max_results=200,
            )
            if issues_result.get("ok"):
                existing_issues = issues_result.get("issues") or []
            else:
                logger.warning(
                    "bootstrap_roadmap_from_objectives: list_issues "
                    "failed (%s) — proceeding without existing-issue "
                    "context.",
                    issues_result.get("message"),
                )

        # ---------- step 4: LLM proposal -----------------------------
        prompt = _build_roadmap_proposal_prompt(
            design_context_summary=design_context_summary,
            existing_roadmap=existing_roadmap,
            roadmap_path=roadmap_path,
            existing_issues=existing_issues,
        )
        try:
            from polymathera.colony.knowledge.deps import (
                build_default_llm_callable,
            )
            llm_callable = build_default_llm_callable(
                max_tokens=llm_max_tokens,
                temperature=llm_temperature,
            )
            raw = await asyncio.wait_for(
                llm_callable(prompt), timeout=llm_timeout_s,
            )
        except asyncio.TimeoutError:
            return {
                "dry_run": dry_run,
                "proposal": None,
                "error": "llm_timeout",
                "message": (
                    f"LLM call exceeded the {llm_timeout_s}s timeout; "
                    "consider raising ``llm_timeout_s`` or simplifying "
                    "the design context."
                ),
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "bootstrap_roadmap_from_objectives: LLM call failed",
            )
            return {
                "dry_run": dry_run,
                "proposal": None,
                "error": "llm_call_failed",
                "message": str(exc),
            }

        proposal = _parse_roadmap_proposal(raw)
        if proposal is None:
            return {
                "dry_run": dry_run,
                "proposal": None,
                "error": "llm_proposal_parse_failed",
                "message": (
                    "LLM output did not parse as a valid roadmap "
                    "proposal (see WARN logs for details)."
                ),
                "raw_excerpt": raw[:500] if raw else "",
            }

        # Stable ids are already injected by :func:`_parse_roadmap_proposal`.

        # ---------- early-return: dry_run ----------------------------
        if dry_run:
            task_count = sum(
                len(m["tasks"]) for m in proposal["milestones"]
            )
            return {
                "dry_run": True,
                "proposal": proposal,
                "stats": {
                    "milestone_count": len(proposal["milestones"]),
                    "task_count": task_count,
                    "existing_issue_count": len(existing_issues),
                },
                "message": (
                    f"Proposal generated ({len(proposal['milestones'])} "
                    f"milestones, {task_count} tasks). Re-call with "
                    f"``dry_run=False`` to write ROADMAP.md and "
                    f"create GitHub issues."
                ),
                "error": "",
            }

        # ---------- step 6: write ROADMAP.md + commit ----------------
        rendered = _render_roadmap_markdown(proposal)
        try:
            roadmap_file.parent.mkdir(parents=True, exist_ok=True)
            roadmap_file.write_text(rendered, encoding="utf-8")
        except OSError as exc:
            return {
                "dry_run": False,
                "proposal": proposal,
                "error": "roadmap_write_failed",
                "message": str(exc),
            }

        msg = (
            commit_message
            or "Bootstrap roadmap from design-context objectives "
            "(via DesignProcessCapability.bootstrap_roadmap_from_objectives)"
        )
        commit_sha = ""
        try:
            commit_sha = await asyncio.to_thread(
                _commit_and_push_roadmap_file,
                repo_root=repo_root,
                roadmap_relpath=roadmap_path,
                message=msg,
            )
        except _CommitFailed as exc:
            logger.exception(
                "bootstrap_roadmap_from_objectives: ROADMAP.md "
                "written to disk but git commit failed",
            )
            return {
                "dry_run": False,
                "proposal": proposal,
                "roadmap_written": roadmap_path,
                "error": "commit_failed",
                "message": str(exc),
            }
        except _PushFailed as exc:
            logger.exception(
                "bootstrap_roadmap_from_objectives: ROADMAP.md "
                "committed locally (sha=%s) but push to origin failed",
                exc.commit_sha,
            )
            return {
                "dry_run": False,
                "proposal": proposal,
                "roadmap_written": roadmap_path,
                "commit_sha": exc.commit_sha,
                "error": "push_failed",
                "message": str(exc),
            }

        # ---------- step 7: create one GitHub issue per task ---------
        issues_created: list[dict[str, Any]] = []
        issue_failures: list[dict[str, Any]] = []
        if github is None:
            logger.warning(
                "bootstrap_roadmap_from_objectives: no GitHubCapability "
                "sibling — ROADMAP.md written + committed but no "
                "issues created. Mount a GitHubCapability on the agent "
                "and re-run if you want the GitHub side too.",
            )
        else:
            for milestone in proposal["milestones"]:
                for task in milestone["tasks"]:
                    body = _render_issue_body_for_task(
                        milestone=milestone, task=task,
                    )
                    create_result = await github.create_issue(
                        title=task["title"],
                        body=body,
                        repo=repo,
                        labels=task.get("labels") or None,
                        project_id=target_project_id,
                    )
                    if create_result.get("ok"):
                        issues_created.append({
                            "stable_id": task["stable_id"],
                            "milestone": milestone["title"],
                            "task": task["title"],
                            "issue": create_result.get("issue", {}),
                            "project_item_id": create_result.get(
                                "project_item_id",
                            ),
                            "project_attach_error": create_result.get(
                                "project_attach_error",
                            ),
                        })
                    else:
                        issue_failures.append({
                            "stable_id": task["stable_id"],
                            "milestone": milestone["title"],
                            "task": task["title"],
                            "error": create_result.get("message", ""),
                        })

        return {
            "dry_run": False,
            "proposal": proposal,
            "roadmap_written": roadmap_path,
            "commit_sha": commit_sha,
            "issues_created": issues_created,
            "issue_failures": issue_failures,
            "stats": {
                "milestone_count": len(proposal["milestones"]),
                "task_count": sum(
                    len(m["tasks"]) for m in proposal["milestones"]
                ),
                "issues_created_count": len(issues_created),
                "issue_failure_count": len(issue_failures),
            },
            "error": "",
        }

    @action_executor(
        planning_summary=(
            "Reconcile ROADMAP.md with GitHub issues via the "
            "stable-id markers stamped by bootstrap. Three directions; "
            "default bidirectional. ``dry_run`` returns the diff "
            "without writing — operator reviews + re-calls with "
            "``dry_run=False`` to apply."
        ),
    )
    async def sync_roadmap_with_github(
        self,
        *,
        direction: str = "bidirectional",
        dry_run: bool = True,
        target_project_id: str | None = None,
        commit_message: str | None = None,
        emit_blackboard_events: bool = True,
        max_issues_scanned: int = 500,
    ) -> dict[str, Any]:
        """Reconcile ROADMAP.md against GitHub issues.

        The two sides are joined via the
        ``<!-- colony:roadmap-task: <stable-id> -->`` marker P5b's
        ``bootstrap_roadmap_from_objectives`` stamps onto issue
        bodies. :func:`_diff_roadmap_vs_issues` produces four
        buckets (``roadmap_only`` / ``github_only`` / ``divergent``
        / ``in_sync``) plus a ``untracked_issues`` list (issues
        without any marker — the operator chose to create them
        outside the bootstrap flow, the sync action leaves them
        alone).

        ``direction`` controls which side gets writes applied:

        - ``bidirectional`` (default) — create issues for
          roadmap-only tasks AND append github-only tasks to
          ROADMAP.md under an ``Untracked`` milestone.
        - ``roadmap_to_github`` — only create issues; ROADMAP.md
          stays read-only.
        - ``github_to_roadmap`` — only update ROADMAP.md; GitHub
          stays read-only.

        ``divergent`` tasks (both sides exist, titles differ) are
        NEVER auto-resolved — surfaced in the response for operator
        mediation. Real conflict resolution requires LLM judgement
        on which side to honour and that gets its own action.

        ``dry_run=True`` (default) returns the computed diff + the
        actions the apply pass WOULD take, without writing
        anything. ``dry_run=False`` applies per ``direction``.

        Emits one :class:`RoadmapSyncProtocol` event per run when
        ``emit_blackboard_events=True``, carrying the diff + the
        apply outcome so the Colony Status panel can render the
        latest sync state.
        """

        import time as _time

        if direction not in (
            "bidirectional", "roadmap_to_github", "github_to_roadmap",
        ):
            return {
                "dry_run": dry_run,
                "direction": direction,
                "error": "invalid_direction",
                "message": (
                    f"direction={direction!r} unknown; pick one of "
                    "bidirectional / roadmap_to_github / github_to_roadmap."
                ),
            }

        # ``repo`` is derived later (after the sibling check) so the
        # more-fundamental "no GitHub capability mounted" error fires
        # first when both are missing.

        # ---------- step 0: design-monorepo presence -----------------
        repo_root = self._working_dir
        if not (repo_root / ".git").is_dir():
            self._lazy_clone_from_agent_metadata()
        if not (repo_root / ".git").is_dir():
            return {
                "dry_run": dry_run,
                "direction": direction,
                "error": "no_design_monorepo",
                "message": (
                    f"{repo_root} is not a git repository — set the "
                    "colony's design-monorepo URL or run "
                    "``initialize_repo_map`` first."
                ),
            }

        repo_map = await asyncio.to_thread(RepoMap.load, repo_root)
        roadmap_path = self._resolve_roadmap_path(repo_map)
        if roadmap_path is None:
            return {
                "dry_run": dry_run,
                "direction": direction,
                **self._no_roadmap_declared_error(),
            }

        # ---------- step 1: read + parse the roadmap ----------------
        roadmap_file = repo_root / roadmap_path
        existing_roadmap_text = ""
        if roadmap_file.is_file():
            try:
                existing_roadmap_text = roadmap_file.read_text(
                    encoding="utf-8",
                )
            except OSError as exc:
                logger.warning(
                    "sync_roadmap_with_github: failed to read "
                    "roadmap at %s: %s — treating as empty.",
                    roadmap_file, exc,
                )
        roadmap_parsed = _parse_roadmap_markdown(existing_roadmap_text)

        # ---------- step 2: list ALL (open + closed) issues ---------
        github = self._sibling_github_capability()
        if github is None:
            return {
                "dry_run": dry_run,
                "direction": direction,
                "error": "github_capability_missing",
                "message": (
                    "No GitHubCapability sibling found on the agent; "
                    "mount one alongside DesignProcessCapability so "
                    "sync_roadmap_with_github can list issues."
                ),
            }
        repo = self._resolve_github_repo()
        if repo is None:
            return {
                "dry_run": dry_run,
                "direction": direction,
                **self._no_github_repo_error(),
            }
        # state="all" so a roadmap task whose issue is closed still
        # matches (otherwise we'd erroneously propose creating a new
        # one). Open + closed are both relevant to the diff.
        issues_result = await github.list_issues(
            repo=repo, state="all", max_results=max_issues_scanned,
        )
        if not issues_result.get("ok"):
            return {
                "dry_run": dry_run,
                "direction": direction,
                "error": "list_issues_failed",
                "message": issues_result.get("message", ""),
            }
        issues = issues_result.get("issues") or []

        # ---------- step 3: diff -------------------------------------
        diff = _diff_roadmap_vs_issues(roadmap_parsed, issues)

        # ---------- step 4: plan apply per direction ----------------
        will_create_issues = direction in (
            "bidirectional", "roadmap_to_github",
        )
        will_update_roadmap = direction in (
            "bidirectional", "github_to_roadmap",
        )

        if dry_run:
            planned_actions = {
                "issues_to_create": (
                    diff["roadmap_only"] if will_create_issues else []
                ),
                "roadmap_appends": (
                    diff["github_only"] if will_update_roadmap else []
                ),
            }
            await self._maybe_emit_roadmap_sync_event(
                repo=repo, direction=direction, diff=diff,
                applied=False, dry_run=True,
                roadmap_written=False, commit_sha="",
                issues_created=[], issue_failures=[],
                emit=emit_blackboard_events,
            )
            return {
                "dry_run": True,
                "direction": direction,
                "diff": diff,
                "planned_actions": planned_actions,
                "stats": _sync_stats(diff),
                "error": "",
            }

        # ---------- step 5: apply ------------------------------------
        issues_created: list[dict[str, Any]] = []
        issue_failures: list[dict[str, Any]] = []
        if will_create_issues:
            for entry in diff["roadmap_only"]:
                body = _render_issue_body_for_task(
                    milestone={"title": entry["milestone_title"]},
                    task={
                        "title": entry["title"],
                        "description": entry["description"],
                        "stable_id": entry["stable_id"],
                    },
                )
                result = await github.create_issue(
                    title=entry["title"], body=body, repo=repo,
                    labels=entry.get("labels") or None,
                    project_id=target_project_id,
                )
                if result.get("ok"):
                    issues_created.append({
                        "stable_id": entry["stable_id"],
                        "milestone": entry["milestone_title"],
                        "task": entry["title"],
                        "issue": result.get("issue", {}),
                        "project_item_id": result.get("project_item_id"),
                    })
                else:
                    issue_failures.append({
                        "stable_id": entry["stable_id"],
                        "task": entry["title"],
                        "error": result.get("message", ""),
                    })

        roadmap_written = False
        commit_sha = ""
        if will_update_roadmap and diff["github_only"]:
            roadmap_after = _merge_github_only_into_roadmap(
                roadmap_parsed, diff["github_only"],
            )
            rendered = _render_roadmap_markdown(roadmap_after)
            try:
                roadmap_file.parent.mkdir(parents=True, exist_ok=True)
                roadmap_file.write_text(rendered, encoding="utf-8")
                roadmap_written = True
            except OSError as exc:
                logger.exception(
                    "sync_roadmap_with_github: failed to write "
                    "ROADMAP.md at %s",
                    roadmap_file,
                )
                return {
                    "dry_run": False,
                    "direction": direction,
                    "diff": diff,
                    "issues_created": issues_created,
                    "issue_failures": issue_failures,
                    "error": "roadmap_write_failed",
                    "message": str(exc),
                }
            try:
                commit_sha = await asyncio.to_thread(
                    _commit_and_push_roadmap_file,
                    repo_root=repo_root,
                    roadmap_relpath=roadmap_path,
                    message=(
                        commit_message
                        or "Sync roadmap with GitHub issues "
                        "(via DesignProcessCapability."
                        "sync_roadmap_with_github)"
                    ),
                )
            except _CommitFailed as exc:
                logger.exception(
                    "sync_roadmap_with_github: ROADMAP.md written but "
                    "git commit failed",
                )
                return {
                    "dry_run": False,
                    "direction": direction,
                    "diff": diff,
                    "issues_created": issues_created,
                    "issue_failures": issue_failures,
                    "roadmap_written": True,
                    "error": "commit_failed",
                    "message": str(exc),
                }
            except _PushFailed as exc:
                logger.exception(
                    "sync_roadmap_with_github: ROADMAP.md committed "
                    "locally (sha=%s) but push to origin failed",
                    exc.commit_sha,
                )
                return {
                    "dry_run": False,
                    "direction": direction,
                    "diff": diff,
                    "issues_created": issues_created,
                    "issue_failures": issue_failures,
                    "roadmap_written": True,
                    "commit_sha": exc.commit_sha,
                    "error": "push_failed",
                    "message": str(exc),
                }

        await self._maybe_emit_roadmap_sync_event(
            repo=repo, direction=direction, diff=diff,
            applied=True, dry_run=False,
            roadmap_written=roadmap_written, commit_sha=commit_sha,
            issues_created=issues_created,
            issue_failures=issue_failures,
            emit=emit_blackboard_events,
        )

        return {
            "dry_run": False,
            "direction": direction,
            "diff": diff,
            "issues_created": issues_created,
            "issue_failures": issue_failures,
            "roadmap_written": roadmap_written,
            "commit_sha": commit_sha,
            "stats": {
                **_sync_stats(diff),
                "issues_created_count": len(issues_created),
                "issue_failure_count": len(issue_failures),
            },
            "error": "",
        }

    @action_executor(
        planning_summary=(
            "Propose colony-vs-user assignment for each open roadmap-"
            "linked issue. ``colony`` → the App-bot (per-tenant); "
            "``user`` → the human's verified GitHub login (when they've "
            "connected GitHub). Marker overrides LLM classification. "
            "``dry_run`` returns the proposal; ``dry_run=False`` applies "
            "via assign_issue (user proposals are skipped when the user "
            "hasn't connected GitHub)."
        ),
    )
    async def propose_task_assignments(
        self,
        *,
        dry_run: bool = True,
        reassign_existing: bool = False,
        max_issues_scanned: int = 500,
        llm_max_tokens: int = 256,
        llm_temperature: float = 0.0,
        llm_timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        """Classify each open roadmap-linked GitHub issue as colony-
        owned or user-owned and (optionally) apply the assignment.

        Identity resolution (P7 + P8 of
        ``colony/github_identity_fix_plan.md``):

        - ``colony`` → :meth:`GitHubCapability.whoami` returns the
          App-bot login (e.g. ``polymathera-colony[bot]``) for the
          tenant's installation.
        - ``user`` → the human's GitHub login from
          ``agent.metadata.parameters["github_identity"]
          ["user_github_login"]`` (populated by the OAuth callback
          on the user's profile). When the user hasn't connected
          GitHub yet, the proposal is marked ``user_unassignable=True``
          and skipped at apply time.

        Classification source (per task, in order):

        1. Explicit ``<!-- colony:assignee: colony|user -->`` marker
           in the roadmap task line OR the GitHub issue body.
        2. LLM classification using
           :func:`_build_assignment_classification_prompt`.

        ``reassign_existing=False`` (default) skips issues with any
        current assignee — operators set assignees by hand all the
        time and we don't override that without an explicit ask.
        ``reassign_existing=True`` proposes (and on apply,
        overwrites) for every roadmap-linked open issue.

        ``dry_run=True`` (default) returns the proposal without
        calling ``assign_issue``. ``dry_run=False`` calls
        :meth:`GitHubCapability.assign_issue` per proposal with
        ``replace=True`` so re-runs don't accumulate co-assignees;
        ``user_unassignable`` proposals are skipped (no apply attempt).
        """

        # P8 (github_identity_fix_plan): user-side login comes from
        # the per-user OAuth identity threaded into agent metadata
        # by session-create (P4). ``None`` when the user has not run
        # the Connect GitHub flow on their profile.
        user_login: str | None = None
        if self._agent is not None:
            from ..agents.patterns.capabilities.github import GitHubCapability
            gh_identity = self._agent.metadata.parameters.get(
                GitHubCapability.GITHUB_IDENTITY_KEY,
            ) or {}
            user_login = gh_identity.get("user_github_login")

        # ``repo`` is resolved after the sibling check so the
        # more-fundamental "no GitHub capability" error wins when both
        # are missing.

        # ---------- step 1: GitHub sibling --------------------------
        github = self._sibling_github_capability()
        if github is None:
            return {
                "dry_run": dry_run,
                "error": "github_capability_missing",
                "message": (
                    "No GitHubCapability sibling found on the agent; "
                    "mount one alongside DesignProcessCapability so "
                    "propose_task_assignments can list + assign issues."
                ),
            }
        repo = self._resolve_github_repo()
        if repo is None:
            return {
                "dry_run": dry_run,
                **self._no_github_repo_error(),
            }

        whoami_result = await github.whoami()
        if not whoami_result.get("ok"):
            return {
                "dry_run": dry_run,
                "error": "whoami_failed",
                "message": (
                    "Could not resolve Colony's GitHub bot login: "
                    f"{whoami_result.get('message', '')}"
                ),
            }
        colony_login = whoami_result["login"]

        # ---------- step 2: roadmap markers (for override hints) ----
        # The roadmap is OPTIONAL here — propose's primary classification
        # source is the GitHub issue body (markers + LLM). When a
        # roadmap row is declared and the file exists, we additionally
        # pick up marker hints stamped on the roadmap task line; when
        # not, we skip the file-read step and fall through to issue-
        # body markers + LLM classification.
        repo_root = self._working_dir
        if not (repo_root / ".git").is_dir():
            self._lazy_clone_from_agent_metadata()
        roadmap_markers_by_id: dict[str, str | None] = {}
        roadmap_text = ""
        if (repo_root / ".git").is_dir():
            repo_map = await asyncio.to_thread(RepoMap.load, repo_root)
            roadmap_path = self._resolve_roadmap_path(repo_map)
            if roadmap_path is not None:
                roadmap_file = repo_root / roadmap_path
                if roadmap_file.is_file():
                    try:
                        roadmap_text = roadmap_file.read_text(encoding="utf-8")
                    except OSError:
                        roadmap_text = ""
        if roadmap_text:
            # Walk task lines; for each stable-id, pull the assignee
            # marker present on the same line (operators stamp it as
            # ``... <!-- colony:roadmap-task: <id> --> <!-- colony:
            # assignee: user -->``). One pass; preserves None when
            # absent so the LLM path still fires.
            for line in roadmap_text.splitlines():
                m = _ROADMAP_TASK_LINE_RE.match(line.rstrip())
                if m is None:
                    continue
                sid = m.group("stable_id")
                roadmap_markers_by_id[sid] = (
                    _extract_assignee_marker(line)
                )

        # ---------- step 3: list open issues with markers -----------
        issues_result = await github.list_issues(
            repo=repo, state="open", max_results=max_issues_scanned,
        )
        if not issues_result.get("ok"):
            return {
                "dry_run": dry_run,
                "error": "list_issues_failed",
                "message": issues_result.get("message", ""),
            }
        issues = issues_result.get("issues") or []

        # Only act on roadmap-linked issues. Untracked issues
        # (operator-created, no marker) are surfaced separately so
        # the response is honest about scope.
        candidates: list[dict[str, Any]] = []
        untracked_issues: list[dict[str, Any]] = []
        for issue in issues:
            sid = _extract_roadmap_task_marker(issue.get("body"))
            if sid is None:
                untracked_issues.append({
                    "issue_number": issue.get("number"),
                    "title": issue.get("title"),
                    "current_assignees": issue.get("assignees") or [],
                })
                continue
            candidates.append({"stable_id": sid, "issue": issue})

        # ---------- step 4: classify --------------------------------
        proposals: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        llm_callable = None
        for candidate in candidates:
            sid = candidate["stable_id"]
            issue = candidate["issue"]
            issue_number = issue.get("number")
            title = issue.get("title") or ""
            body = issue.get("body") or ""
            milestone_title = issue.get("milestone") or ""
            current_assignees = issue.get("assignees") or []

            # Skip already-assigned issues unless re-asked.
            if current_assignees and not reassign_existing:
                skipped.append({
                    "stable_id": sid,
                    "issue_number": issue_number,
                    "title": title,
                    "current_assignees": current_assignees,
                    "reason": "already_assigned",
                })
                continue

            # Classification: marker override first.
            assignee = (
                _extract_assignee_marker(body)
                or roadmap_markers_by_id.get(sid)
            )
            source = "marker"
            reason = "explicit colony:assignee marker"
            if assignee is None:
                # LLM fallback.
                if llm_callable is None:
                    try:
                        from polymathera.colony.knowledge.deps import (
                            build_default_llm_callable,
                        )
                        llm_callable = build_default_llm_callable(
                            max_tokens=llm_max_tokens,
                            temperature=llm_temperature,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.exception(
                            "propose_task_assignments: failed to build "
                            "LLM callable",
                        )
                        skipped.append({
                            "stable_id": sid,
                            "issue_number": issue_number,
                            "title": title,
                            "current_assignees": current_assignees,
                            "reason": "llm_unavailable",
                            "detail": str(exc),
                        })
                        continue
                prompt = _build_assignment_classification_prompt(
                    milestone_title=milestone_title,
                    task_title=title,
                    task_description=body,
                )
                try:
                    raw = await asyncio.wait_for(
                        llm_callable(prompt), timeout=llm_timeout_s,
                    )
                except asyncio.TimeoutError:
                    skipped.append({
                        "stable_id": sid,
                        "issue_number": issue_number,
                        "title": title,
                        "current_assignees": current_assignees,
                        "reason": "llm_timeout",
                    })
                    continue
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "propose_task_assignments: LLM call failed for "
                        "issue #%s", issue_number,
                    )
                    skipped.append({
                        "stable_id": sid,
                        "issue_number": issue_number,
                        "title": title,
                        "current_assignees": current_assignees,
                        "reason": "llm_call_failed",
                        "detail": str(exc),
                    })
                    continue
                parsed = _parse_assignment_classification(raw)
                if parsed is None:
                    skipped.append({
                        "stable_id": sid,
                        "issue_number": issue_number,
                        "title": title,
                        "current_assignees": current_assignees,
                        "reason": "llm_parse_failed",
                        "raw_excerpt": (raw or "")[:200],
                    })
                    continue
                assignee = parsed["assignee"]
                source = "llm"
                reason = parsed["reason"] or "(no reason provided)"

            proposed_login = (
                colony_login if assignee == "colony" else user_login
            )
            # User-marked tasks need a connected GitHub identity. When
            # ``user_login`` is None, surface the gap honestly + skip
            # apply for this proposal (the user must click "Connect
            # GitHub" on their profile before this can land).
            user_unassignable = (
                assignee == "user" and not user_login
            )
            proposals.append({
                "stable_id": sid,
                "issue_number": issue_number,
                "milestone": milestone_title,
                "title": title,
                "current_assignees": current_assignees,
                "proposed_assignee": assignee,
                "proposed_login": proposed_login,
                "user_unassignable": user_unassignable,
                "source": source,
                "reason": reason,
            })

        # ---------- step 5: dry-run early-return --------------------
        if dry_run:
            return {
                "dry_run": True,
                "colony_login": colony_login,
                "user_login": user_login,
                "proposals": proposals,
                "skipped": skipped,
                "untracked_issues": untracked_issues,
                "stats": {
                    "proposed_count": len(proposals),
                    "skipped_count": len(skipped),
                    "untracked_count": len(untracked_issues),
                    "proposed_colony": sum(
                        1 for p in proposals
                        if p["proposed_assignee"] == "colony"
                    ),
                    "proposed_user": sum(
                        1 for p in proposals
                        if p["proposed_assignee"] == "user"
                    ),
                    "user_unassignable_count": sum(
                        1 for p in proposals if p["user_unassignable"]
                    ),
                    "marker_count": sum(
                        1 for p in proposals if p["source"] == "marker"
                    ),
                    "llm_count": sum(
                        1 for p in proposals if p["source"] == "llm"
                    ),
                },
                "error": "",
            }

        # ---------- step 6: apply -----------------------------------
        applied: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for p in proposals:
            if p["user_unassignable"]:
                # User hasn't connected GitHub yet — surfaced in the
                # proposal already; don't call ``assign_issue`` with
                # ``[None]``. The user can re-run after connecting.
                continue
            result = await github.assign_issue(
                p["issue_number"], [p["proposed_login"]],
                repo=repo, replace=True,
            )
            if result.get("ok"):
                applied.append({
                    "stable_id": p["stable_id"],
                    "issue_number": p["issue_number"],
                    "proposed_assignee": p["proposed_assignee"],
                    "assigned_login": p["proposed_login"],
                })
            else:
                errors.append({
                    "stable_id": p["stable_id"],
                    "issue_number": p["issue_number"],
                    "proposed_assignee": p["proposed_assignee"],
                    "error": result.get("message", ""),
                })

        return {
            "dry_run": False,
            "colony_login": colony_login,
            "user_login": user_login,
            "proposals": proposals,
            "applied": applied,
            "errors": errors,
            "skipped": skipped,
            "untracked_issues": untracked_issues,
            "stats": {
                "proposed_count": len(proposals),
                "applied_count": len(applied),
                "error_count": len(errors),
                "skipped_count": len(skipped),
                "untracked_count": len(untracked_issues),
                "user_unassignable_count": sum(
                    1 for p in proposals if p["user_unassignable"]
                ),
            },
            "error": "",
        }

    async def _maybe_emit_roadmap_sync_event(
        self,
        *,
        repo: str | None,
        direction: str,
        diff: dict[str, Any],
        applied: bool,
        dry_run: bool,
        roadmap_written: bool,
        commit_sha: str,
        issues_created: list[dict[str, Any]],
        issue_failures: list[dict[str, Any]],
        emit: bool,
    ) -> None:
        """One :class:`RoadmapSyncProtocol` event per sync run (dry
        or apply) so the Colony Status panel can render the latest
        sync state without re-querying the action."""

        if not emit:
            return
        github = self._sibling_github_capability()
        resolved_repo = (
            repo or (github._default_repo if github is not None else "")
            or "local"
        )
        import time as _time

        blackboard = await self._get_colony_blackboard()
        now = _time.time()
        await blackboard.write(
            key=RoadmapSyncProtocol.event_key(
                repo=resolved_repo,
                direction=direction,
                millis=int(now * 1000),
            ),
            value={
                "direction": direction,
                "applied": applied,
                "dry_run": dry_run,
                "diff": diff,
                "roadmap_written": roadmap_written,
                "commit_sha": commit_sha,
                "issues_created_count": len(issues_created),
                "issue_failure_count": len(issue_failures),
                "ran_at": now,
            },
            created_by=(
                f"design_process.sync_roadmap_with_github:"
                f"{self.capability_key}"
            ),
            tags={
                "design_process",
                "roadmap_sync",
                direction,
                *(("applied",) if applied else ("dry_run",)),
            },
        )

    def _sibling_github_capability(self) -> Any:
        """Look up the ``GitHubCapability`` mounted on the same agent.

        Returns the capability instance, or ``None`` when (a) the
        capability is detached (no agent), (b) no GitHubCapability is
        mounted, or (c) the agent's capability registry doesn't
        expose lookup-by-class. Callers must handle ``None`` —
        ``summarise_progress`` and ``identify_bottlenecks`` degrade
        gracefully.
        """

        if self._agent is None:
            return None
        # Try the standard capability registry. Different Agent /
        # AgentHandle implementations expose this differently; try
        # the most common shapes.
        for attr in ("get_capability", "capability_by_class"):
            getter = getattr(self._agent, attr, None)
            if getter is None:
                continue
            try:
                # Lazy import to avoid cycles + to keep this method
                # safe in test environments without the full agent
                # infrastructure.
                from ..agents.patterns.capabilities.github import (
                    GitHubCapability,
                )
                cap = getter(GitHubCapability)
                if cap is not None:
                    return cap
            except Exception:  # noqa: BLE001
                continue
        # Fall back to the ``_capabilities`` dict if exposed.
        caps = getattr(self._agent, "_capabilities", None)
        if isinstance(caps, dict):
            from ..agents.patterns.capabilities.github import (
                GitHubCapability,
            )
            for cap in caps.values():
                if isinstance(cap, GitHubCapability):
                    return cap
        return None

    async def _discover_bottleneck_rules(
        self, *, rule_set: set[str],
    ) -> list[dict[str, Any]]:
        """Scan the design-context corner of the KG for claims that
        define operator-authored bottleneck rules. Mirrors
        :meth:`SystemDesignCapability.find_inconsistencies`'s rule-
        discovery surface — same shape, different predicate set.

        Returns an empty list (without raising) when the KG is empty
        or no graph store is wired — the built-in stalled-issue
        heuristic still fires."""

        try:
            from polymathera.colony.knowledge.deps import get_knowledge_deps
            deps = get_knowledge_deps()
            graph_store = deps.graph_store
        except Exception:  # noqa: BLE001
            return []
        if graph_store is None:
            return []
        try:
            result = await graph_store.query(
                f"MATCH (s)-[r]->(o) LIMIT {SYSDES_KUZU_SCAN_LIMIT}",
            )
        except Exception:  # noqa: BLE001
            return []
        nodes_by_id = {n.node_id: n for n in result.nodes}
        rules: list[dict[str, Any]] = []
        for edge in result.edges:
            citation_uri = getattr(edge, "citation_uri", None) or ""
            if not citation_uri.startswith(
                f"{DESIGN_CONTEXT_URI_SCHEME}://",
            ):
                continue
            s_node = nodes_by_id.get(edge.source_id)
            o_node = nodes_by_id.get(edge.target_id)
            if s_node is None or o_node is None:
                continue
            s_surface = (
                str(s_node.properties.get("surface", ""))
                or s_node.node_id
            )
            o_surface = (
                str(o_node.properties.get("surface", ""))
                or o_node.node_id
            )
            is_rule = edge.predicate in rule_set or (
                edge.predicate == "is_a"
                and o_surface.replace("_", " ").lower()
                in {"bottleneck rule", "bottleneck_rule"}
            )
            if not is_rule:
                continue
            source_name, rel_path = parse_design_context_uri(citation_uri)
            rules.append({
                "rule_id": s_surface,
                "predicate": edge.predicate,
                "object": o_surface,
                "source_name": source_name,
                "file": rel_path,
                "citation_uri": citation_uri,
                "confidence": edge.confidence,
            })
        return rules


__all__ = (
    "DesignProcessCapability",
)
