"""GitHub capability — agents interact with issues, PRs, and project
boards via a GitHub App installation.

Design reference: ``colony_docs/markdown/plans/design_GitHubCapability.md``.

Auth flow: ``GitHubAppAuth`` mints a short-lived JWT from the App's
RSA private key; ``TokenCache`` trades the JWT for an installation
access token and keeps it fresh. ``GitHubClient`` wraps REST and
GraphQL with retries and distinct exception types. This capability is
a thin action surface over the client.

``claim_unassigned_issue`` is the coordination primitive the design
calls out: instead of trying to use GraphQL optimistic concurrency
(which GitHub does not reliably support), it applies a
``claimed-by:<agent_id>`` label as the atomic check — the second
concurrent caller sees the label and backs off.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Literal, TYPE_CHECKING
from overrides import override

import httpx

from ...base import AgentCapability
from ...metadata_parameters import ParameterScope, ParameterSpec
from ...blackboard.protocol import GitHubEventProtocol
from ...models import AgentSuspensionState
from ...scopes import BlackboardScope, get_scope_prefix
from ..actions import action_executor

from ._github import (
    GitHubAppAuth,
    GitHubClient,
    GitHubError,
    NotFoundError,
    RateLimitError,
    TokenCache,
)

if TYPE_CHECKING:
    from ...base import Agent


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper to uniform-shape every action return
# ---------------------------------------------------------------------------

def _ok(**fields: Any) -> dict[str, Any]:
    return {"ok": True, "message": "", **fields}


def _err(message: str, **fields: Any) -> dict[str, Any]:
    return {"ok": False, "message": message, **fields}


# ---------------------------------------------------------------------------
# Colony GitHub-identity conventions (top-level design plan §16.5)
# ---------------------------------------------------------------------------
#
# Every Colony-authored GitHub comment carries a one-line signature
# (visible) + a machine-readable HTML-comment footer (invisible to
# humans, parsed by the future GitHub-inbound poller / webhook in P8
# to join inbound comments back to their originating Colony session).
# The helpers below render both. They're pure (no I/O), module-level
# so the encoding is unit-testable + reusable from other capabilities
# that decide to adopt the same convention.

_SESSION_AGENT_SIGNATURE_TEMPLATE = (
    "<sub>🤖 **Colony · _{agent_role}_** {context_blurb}</sub>"
)
"""Markdown-rendered prefix. ``<sub>`` keeps the line visually small
on GitHub; ``_{agent_role}_`` is italic for the agent name, ``**``
bolds the Colony brand. Layout matches the top-level design plan §16.5."""


def _render_session_agent_signature(
    *,
    agent_role: str,
    trigger: Literal["chat", "scheduled", "mention", "automated"],
    session_id: str,
    replying_to: str | None = None,
    src_comment_id: int | None = None,
    scheduled_mission_name: str | None = None,
) -> str:
    """Render the visible-to-humans signature line for a Colony-
    authored comment.

    Trigger semantics:

    - ``chat`` — Colony posted on behalf of a user message; the
      signature reads "replying to @{login}".
    - ``mention`` — Colony was @-mentioned in a GitHub comment; the
      signature reads "replying to @{login} (mention #{comment_id})".
    - ``scheduled`` — Colony posted from a scheduled mission run; the
      signature reads "triggered by scheduled mission {name}".
    - ``automated`` — any other automated trigger; signature reads
      "(automated)".
    """

    if trigger == "scheduled":
        name = scheduled_mission_name or "<unnamed>"
        context_blurb = (
            f"triggered by scheduled mission `{name}` · "
            f"session `{session_id}`"
        )
    elif trigger == "mention" and replying_to and src_comment_id is not None:
        context_blurb = (
            f"replying to @{replying_to} (mention #{src_comment_id}) · "
            f"session `{session_id}`"
        )
    elif trigger == "mention" and replying_to:
        context_blurb = (
            f"replying to @{replying_to} (mention) · "
            f"session `{session_id}`"
        )
    elif replying_to:
        context_blurb = (
            f"replying to @{replying_to} · session `{session_id}`"
        )
    else:
        context_blurb = f"(automated) · session `{session_id}`"
    return _SESSION_AGENT_SIGNATURE_TEMPLATE.format(
        agent_role=agent_role, context_blurb=context_blurb,
    )


def _render_attribution_footer(
    *,
    author: str,
    session_id: str,
    run_id: str | None = None,
    user: str | None = None,
    trigger: Literal["chat", "scheduled", "mention", "automated"] = "chat",
    src_comment_id: int | None = None,
    scheduled_mission_name: str | None = None,
) -> str:
    """Render the machine-readable HTML-comment footer for a Colony-
    authored comment.

    The footer is invisible to humans rendering the comment on
    GitHub (HTML comments are stripped from the rendered view) but
    structured for the GitHub-inbound parser (P8): the parser pulls
    out ``session=...``, ``run=...``, ``user=...``, ``trigger=...``,
    ``src_comment=...``, ``scheduled_mission=...`` and writes the
    join into the InteractionLog so the inbound thread maps back to
    its originating Colony session.

    Whitespace inside the HTML comment is preserved by GitHub
    Markdown, so we keep one ``key=value`` per relevant slot
    separated by ``; `` for grep-friendliness.
    """

    parts = [f"author={author}", f"session={session_id}"]
    if run_id is not None:
        parts.append(f"run={run_id}")
    if user is not None:
        parts.append(f"user={user}")
    parts.append(f"trigger={trigger}")
    if src_comment_id is not None:
        parts.append(f"src_comment={src_comment_id}")
    if scheduled_mission_name is not None:
        parts.append(f"scheduled_mission={scheduled_mission_name}")
    return f"<!-- colony:{'; '.join(parts)} -->"


def _summarise_issue(raw: dict[str, Any]) -> dict[str, Any]:
    """Shrink GitHub's verbose issue/PR dict to what the LLM needs.

    The full payload is huge; the LLM only needs identifiers, text,
    state, and a couple of relationship links."""
    user = raw.get("user") or {}
    milestone = raw.get("milestone") or {}
    return {
        "number": raw.get("number"),
        "title": raw.get("title"),
        "body": raw.get("body"),
        "state": raw.get("state"),
        "state_reason": raw.get("state_reason"),
        "author": user.get("login"),
        "labels": [
            label["name"] if isinstance(label, dict) else str(label)
            for label in (raw.get("labels") or [])
        ],
        "assignees": [a.get("login") for a in (raw.get("assignees") or [])],
        "milestone": milestone.get("title") if milestone else None,
        "url": raw.get("html_url"),
        "created_at": raw.get("created_at"),
        "updated_at": raw.get("updated_at"),
        "closed_at": raw.get("closed_at"),
        "is_pull_request": raw.get("pull_request") is not None,
    }


def _summarise_pr(raw: dict[str, Any]) -> dict[str, Any]:
    base = _summarise_issue(raw)
    head = raw.get("head") or {}
    base_ref = raw.get("base") or {}
    base.update({
        "head": head.get("ref"),
        "head_sha": head.get("sha"),
        "base": base_ref.get("ref"),
        "draft": raw.get("draft"),
        "merged": raw.get("merged"),
        "mergeable": raw.get("mergeable"),
        "mergeable_state": raw.get("mergeable_state"),
    })
    return base


# ---------------------------------------------------------------------------
# Capability
# ---------------------------------------------------------------------------

class GitHubCapability(AgentCapability):
    """Agent-facing GitHub App integration.

    Args:
        agent: Owning agent.
        scope: Blackboard partition for event/audit writes.
        namespace: Capability sub-namespace.
        app_id: GitHub App ID. Falls back to ``GITHUB_APP_ID`` env var.
        private_key_pem: RSA private key in PEM form. Falls back to
            ``GITHUB_PRIVATE_KEY_PEM``, then to the file at
            ``private_key_path``.
        private_key_path: Filesystem path to the PEM file. Mutually
            exclusive with ``private_key_pem``.
        installation_id: GitHub App installation id (per-tenant).
            In production, read from
            ``agent.metadata.parameters["github_identity"]
            ["tenant_installation_id"]`` — this kwarg is for test
            injection only.
        default_repo: Optional ``owner/repo`` default — used when an
            action accepts ``repo=None``.
        default_project_id: Optional Projects v2 GraphQL node id used
            by project actions when the caller omits ``project_id``.
        max_requests_per_minute: Soft cap at the action level (the
            primary GitHub limit is 5 000/h — this is per-capability
            hygiene, not the authoritative limit).
        client: Injectable ``GitHubClient`` for tests; defaults to a
            live client with ``TokenCache``.
        httpx_client: Injectable ``httpx.AsyncClient`` used for the
            token exchange and for the live client when ``client`` is
            ``None``.
        audit_enabled: If True, every mutation writes an audit record
            to the colony-scoped audit log.
        capability_key: Dispatcher key.
        app_name: ``serving`` application name override.
    """

    # Key constant for the COLONY-scoped per-tenant + per-user identity
    # block. Single source of truth — the ``AGENT_METADATA_PARAMS``
    # spec below references this name, and cross-file consumers
    # (``design_monorepo/capabilities.py::_resolve_attribution`` /
    # ``design_monorepo/process.py::propose_task_assignments`` /
    # ``distributed/git_credentials.py``) import + use this constant
    # rather than re-typing the string.
    GITHUB_IDENTITY_KEY = "github_identity"

    # Colony-scoped metadata parameter the capability reads in
    # ``_build_live_client`` to mint a per-tenant installation token.
    # Optional with an empty-dict default — sessions for tenants
    # that haven't completed the GitHub-App-installation flow yet
    # surface the missing-installation-id error at the first live
    # action, not at init (matching the existing UX).
    AGENT_METADATA_PARAMS = (
        ParameterSpec(
            name=GITHUB_IDENTITY_KEY,
            scope=ParameterScope.COLONY,
            description=(
                "{tenant_installation_id, user_github_login, "
                "user_github_id, git_user_email, git_user_name}. "
                "``tenant_installation_id`` is read by "
                "``_build_live_client`` to mint REST tokens scoped "
                "to this tenant. The per-user fields are read by "
                "``propose_task_assignments`` and the "
                "design-monorepo trio's commit-attribution path."
            ),
            json_type="object",
            default_factory=dict,
        ),
    )

    _CLAIMED_BY_LABEL_PREFIX = "claimed-by:"

    def __init__(
        self,
        agent: Agent,
        scope: BlackboardScope = BlackboardScope.SESSION,
        namespace: str = "github",
        app_id: str | None = None,
        private_key_pem: str | None = None,
        private_key_path: str | None = None,
        installation_id: str | None = None,
        default_repo: str | None = None,
        default_project_id: str | None = None,
        max_requests_per_minute: int = 120,
        client: GitHubClient | None = None,
        httpx_client: httpx.AsyncClient | None = None,
        audit_enabled: bool = True,
        capability_key: str = "github",
        app_name: str | None = None,
    ):
        super().__init__(
            agent=agent,
            scope_id=get_scope_prefix(scope, agent, namespace=namespace),
            capability_key=capability_key,
            app_name=app_name,
        )
        self._default_repo = default_repo
        self._default_project_id = default_project_id
        self._audit_enabled = audit_enabled
        self._http_owned = httpx_client is None
        self._httpx_client = httpx_client
        self._app_id = app_id
        self._private_key_pem = private_key_pem
        self._private_key_path = private_key_path
        self._installation_id = installation_id
        self._client: GitHubClient | None = client
        self._init_error: str | None = None
        self._client_initialized = client is not None
        # whoami() caches the identity round-trip (the App-bot identity
        # doesn't change at runtime — see P7 of
        # ``colony/github_identity_fix_plan.md``).
        self._identity_cache: dict[str, Any] | None = None

    async def initialize(self) -> None:
        # Build a live client; any configuration error is captured so
        # the action surface can return it as a normal error dict
        # rather than raising at construction time.
        await super().initialize()
        if self._client_initialized:
            return
        try:
            self._client = await self._build_live_client()
            self._init_error = None
        except Exception as e:
            self._client = None
            self._init_error = str(e)
            logger.warning(
                "GitHubCapability: live client disabled: %s", e,
            )
        self._client_initialized = True

    # --- Internal construction --------------------------------------------

    async def _build_live_client(self) -> GitHubClient:
        # ``installation_id`` is per-tenant (P5 of
        # ``colony/github_identity_fix_plan.md``) and rides on agent
        # metadata as ``parameters["github_identity"]
        # ["tenant_installation_id"]``. Tests inject directly via
        # ``self._installation_id``; production reads metadata. The
        # App-creds + httpx + TokenCache + GitHubClient construction
        # is delegated to :func:`build_github_client_for_installation`,
        # the shared factory both this capability and
        # :class:`GitHubInboundCapability` use.
        from ._github.factory import build_github_client_for_installation

        # Read the PEM from the file path if neither inline nor env
        # path supplies it (rare; mostly test fixtures).
        private_key_pem = self._private_key_pem
        if not private_key_pem and self._private_key_path:
            with open(self._private_key_path, "r", encoding="utf-8") as fh:
                private_key_pem = fh.read()

        installation_id = self._installation_id
        if not installation_id and self._agent is not None:
            gh_identity = self._agent.metadata.parameters.get(
                self.GITHUB_IDENTITY_KEY,
            ) or {}
            installation_id = gh_identity.get("tenant_installation_id")

        if not installation_id:
            raise RuntimeError(
                "GitHubCapability: per-tenant GitHub App installation "
                "id is missing — the tenant admin must set it via "
                "the Tenant GitHub Installation panel before sessions "
                "in this tenant can use the REST API."
            )

        client, self._httpx_client = await build_github_client_for_installation(
            installation_id=installation_id,
            app_id_override=self._app_id,
            private_key_pem_override=private_key_pem,
            httpx_client=self._httpx_client,
            capability_name="GitHubCapability",
        )
        return client

    def get_action_group_description(self) -> str:
        return (
            "GitHub — read and write issues, pull requests, and "
            "project boards via a GitHub App installation. Actions "
            "cover the common reviewer/coordinator workflows: "
            "list/get/create/comment on issues and PRs, post PR "
            "reviews, move items on a project board, and atomically "
            "claim an unassigned issue for coordination with peer "
            "agents or human contributors."
        )

    def get_capability_tags(self) -> frozenset[str]:
        return frozenset({"github", "vcs", "external", "coordination"})

    @override
    async def serialize_suspension_state(
        self, state: AgentSuspensionState
    ) -> AgentSuspensionState:
        return state

    @override
    async def deserialize_suspension_state(
        self, state: AgentSuspensionState
    ) -> None:
        return None

    async def shutdown(self) -> None:
        """Close the underlying HTTP client if we own it."""
        if self._client is not None:
            await self._client.close()
        if self._http_owned and self._httpx_client is not None:
            try:
                await self._httpx_client.aclose()
            except Exception:  # pragma: no cover — defensive
                pass

    # --- Internal helpers -------------------------------------------------

    async def _ensure_client(self) -> tuple[GitHubClient | None, str | None]:
        if not self._client_initialized:
            await self.initialize()
        if self._client is None:
            return None, self._init_error or "GitHub client not configured"
        return self._client, None

    def _resolve_repo(self, repo: str | None) -> str | None:
        return repo or self._default_repo

    def _agent_id(self) -> str:
        return self.agent.agent_id if self._agent is not None else "unknown"

    @classmethod
    def _label_for_agent(cls, agent_id: str) -> str:
        return f"{cls._CLAIMED_BY_LABEL_PREFIX}{agent_id}"

    async def _write_audit(self, action: str, payload: dict[str, Any]) -> None:
        if not self._audit_enabled:
            return
        try:
            bb = await self.get_blackboard()
            key = f"audit:github:{int(time.time() * 1000)}:{uuid.uuid4().hex[:8]}"
            await bb.write(key, {
                "action": action,
                "agent_id": self._agent_id(),
                "ts": time.time(),
                **payload,
            })
        except Exception as e:  # pragma: no cover — defensive
            logger.debug("GitHubCapability: audit write failed: %s", e)

    @staticmethod
    def _shape_error(e: Exception) -> dict[str, Any]:
        if isinstance(e, NotFoundError):
            return _err(f"not found: {e}", status_code=404)
        if isinstance(e, RateLimitError):
            return _err(f"rate limited: {e}", status_code=e.status_code)
        if isinstance(e, GitHubError):
            return _err(
                f"GitHub error ({e.status_code}): {e}",
                status_code=e.status_code,
            )
        return _err(f"unexpected error: {e}")

    # =====================================================================
    # Repo + content
    # =====================================================================

    @action_executor()
    async def list_repos(self) -> dict[str, Any]:
        """List the repositories this App installation can access."""
        client, err = await self._ensure_client()
        if client is None:
            return _err(err or "", repos=[])
        try:
            data = await client.get("/installation/repositories")
        except Exception as e:
            return self._shape_error(e) | {"repos": []}
        repos = [
            {
                "full_name": r.get("full_name"),
                "private": r.get("private"),
                "default_branch": r.get("default_branch"),
                "description": r.get("description"),
                "url": r.get("html_url"),
            }
            for r in (data.get("repositories") or [])
        ]
        return _ok(repos=repos, count=len(repos))

    @action_executor()
    async def get_repo(self, repo: str | None = None) -> dict[str, Any]:
        """Fetch one repository's metadata."""
        repo = self._resolve_repo(repo)
        if not repo:
            return _err("no repo provided and no default_repo set")
        client, err = await self._ensure_client()
        if client is None:
            return _err(err or "")
        try:
            data = await client.get(f"/repos/{repo}")
        except Exception as e:
            return self._shape_error(e)
        return _ok(repo={
            "full_name": data.get("full_name"),
            "description": data.get("description"),
            "default_branch": data.get("default_branch"),
            "open_issues_count": data.get("open_issues_count"),
            "stargazers_count": data.get("stargazers_count"),
            "url": data.get("html_url"),
        })

    @action_executor()
    async def list_branches(
        self, repo: str | None = None, *, max_results: int = 100,
    ) -> dict[str, Any]:
        """List branches on a repo."""
        repo = self._resolve_repo(repo)
        if not repo:
            return _err("no repo provided and no default_repo set", branches=[])
        client, err = await self._ensure_client()
        if client is None:
            return _err(err or "", branches=[])
        branches: list[dict[str, Any]] = []
        try:
            async for b in client.iter_paginated(
                f"/repos/{repo}/branches", page_size=100,
            ):
                branches.append({
                    "name": b.get("name"),
                    "protected": b.get("protected"),
                    "sha": (b.get("commit") or {}).get("sha"),
                })
                if len(branches) >= max_results:
                    break
        except Exception as e:
            return self._shape_error(e) | {"branches": []}
        return _ok(branches=branches, count=len(branches))

    @action_executor()
    async def get_file_contents(
        self,
        path: str,
        *,
        repo: str | None = None,
        ref: str | None = None,
        max_bytes: int = 1_000_000,
    ) -> dict[str, Any]:
        """Fetch a file's content at a ref.

        Binary files are returned base64-encoded; text is decoded. The
        ``truncated`` flag reflects the ``max_bytes`` cap, not the
        file's committed ``truncated`` field (which refers to large-
        file storage).
        """
        repo = self._resolve_repo(repo)
        if not repo:
            return _err("no repo provided and no default_repo set")
        client, err = await self._ensure_client()
        if client is None:
            return _err(err or "")
        try:
            params = {"ref": ref} if ref else {}
            data = await client.get(
                f"/repos/{repo}/contents/{path}", **params,
            )
        except Exception as e:
            return self._shape_error(e)
        if isinstance(data, list):
            return _err(
                f"path {path!r} is a directory; use list_directory instead",
            )
        encoding = data.get("encoding", "")
        raw = data.get("content") or ""
        content: str
        binary = False
        if encoding == "base64":
            import base64
            try:
                decoded = base64.b64decode(raw.replace("\n", ""))
                content = decoded.decode("utf-8")
            except UnicodeDecodeError:
                content = raw  # keep base64 form
                binary = True
        else:
            content = raw
        truncated = False
        if len(content) > max_bytes:
            content = content[:max_bytes]
            truncated = True
        return _ok(
            path=path,
            size=data.get("size"),
            sha=data.get("sha"),
            content=content,
            binary=binary,
            truncated=truncated,
        )

    @action_executor()
    async def search_code(
        self,
        query: str,
        *,
        repo: str | None = None,
        max_results: int = 30,
    ) -> dict[str, Any]:
        """Search code across repos the App can see.

        When ``repo`` is provided, the query is automatically narrowed
        to that repo via the ``repo:`` qualifier.
        """
        client, err = await self._ensure_client()
        if client is None:
            return _err(err or "", hits=[])
        full_query = query
        target_repo = self._resolve_repo(repo)
        if target_repo and "repo:" not in query:
            full_query = f"{query} repo:{target_repo}"
        try:
            data = await client.get(
                "/search/code", q=full_query, per_page=min(max_results, 100),
            )
        except Exception as e:
            return self._shape_error(e) | {"hits": []}
        hits = [
            {
                "path": it.get("path"),
                "repo": (it.get("repository") or {}).get("full_name"),
                "url": it.get("html_url"),
                "score": it.get("score"),
            }
            for it in (data.get("items") or [])
        ]
        return _ok(
            hits=hits[:max_results], count=min(len(hits), max_results),
            total=data.get("total_count"),
        )

    # =====================================================================
    # Issues
    # =====================================================================

    @action_executor()
    async def list_issues(
        self,
        *,
        repo: str | None = None,
        state: Literal["open", "closed", "all"] = "open",
        labels: list[str] | None = None,
        assignee: str | None = None,
        max_results: int = 50,
    ) -> dict[str, Any]:
        """List issues, optionally filtered by state/labels/assignee.

        Pull requests are excluded from the result — they come from
        the PR-specific actions. This matches the distinction users
        expect when they say "issue".
        """
        repo = self._resolve_repo(repo)
        if not repo:
            return _err("no repo provided and no default_repo set", issues=[])
        client, err = await self._ensure_client()
        if client is None:
            return _err(err or "", issues=[])
        params: dict[str, Any] = {"state": state, "per_page": 100}
        if labels:
            params["labels"] = ",".join(labels)
        if assignee:
            params["assignee"] = assignee
        issues: list[dict[str, Any]] = []
        try:
            async for it in client.iter_paginated(
                f"/repos/{repo}/issues", page_size=100, **params,
            ):
                if it.get("pull_request"):
                    continue
                issues.append(_summarise_issue(it))
                if len(issues) >= max_results:
                    break
        except Exception as e:
            return self._shape_error(e) | {"issues": []}
        return _ok(issues=issues, count=len(issues))

    @action_executor()
    async def get_issue(
        self, issue_number: int, *, repo: str | None = None,
    ) -> dict[str, Any]:
        """Fetch one issue."""
        repo = self._resolve_repo(repo)
        if not repo:
            return _err("no repo provided and no default_repo set")
        client, err = await self._ensure_client()
        if client is None:
            return _err(err or "")
        try:
            data = await client.get(f"/repos/{repo}/issues/{issue_number}")
        except Exception as e:
            return self._shape_error(e)
        return _ok(issue=_summarise_issue(data))

    @action_executor()
    async def create_issue(
        self,
        *,
        title: str,
        body: str,
        repo: str | None = None,
        labels: list[str] | None = None,
        assignees: list[str] | None = None,
        auto_attach_to_default_project: bool = True,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Open a new issue.

        ``auto_attach_to_default_project`` (default ``True``) plus a
        resolved ``project_id`` (explicit arg wins; otherwise the
        capability's ``default_project_id``) makes the action also
        attach the new issue to the operator's Projects v2 board in
        the same call — one planner step instead of two. Attach
        failure (project not found, permissions, transient GraphQL
        hiccup) is **non-fatal**: the issue creation already
        succeeded, so we surface the attach error as
        ``project_attach_error`` in the response rather than rolling
        back. Set the flag to ``False`` for issues you intentionally
        want unsorted (e.g. spam/triage queue).
        """
        repo = self._resolve_repo(repo)
        if not repo:
            return _err("no repo provided and no default_repo set")
        client, err = await self._ensure_client()
        if client is None:
            return _err(err or "")
        payload: dict[str, Any] = {"title": title, "body": body}
        if labels:
            payload["labels"] = list(labels)
        if assignees:
            payload["assignees"] = list(assignees)
        try:
            data = await client.post(
                f"/repos/{repo}/issues", json=payload,
            )
        except Exception as e:
            return self._shape_error(e)

        out = _ok(issue=_summarise_issue(data))
        await self._write_audit("create_issue", {
            "repo": repo, "number": data.get("number"), "title": title,
        })

        # Optional Projects v2 attachment. Resolved at call time so an
        # operator can toggle ``default_project_id`` between sessions
        # without re-instantiating the capability.
        if auto_attach_to_default_project:
            pid = project_id or self._default_project_id
            content_node_id = data.get("node_id")
            if pid and content_node_id:
                attach = await self._add_to_project_v2(
                    client=client,
                    project_id=pid,
                    content_node_id=content_node_id,
                )
                if attach["ok"]:
                    out["project_item_id"] = attach["item_id"]
                    out["project_id"] = pid
                else:
                    # Mutation failed AFTER the issue was created — the
                    # issue exists, so we don't flip ``out["ok"]``.
                    # Surface the error verbatim so the planner can
                    # decide whether to retry the attach.
                    out["project_attach_error"] = attach["message"]
            elif pid and not content_node_id:
                out["project_attach_error"] = (
                    "issue created but its GraphQL node_id was missing "
                    "from the REST response; cannot run "
                    "addProjectV2ItemById without it."
                )
        return out

    @action_executor()
    async def comment_on_issue(
        self,
        issue_number: int,
        body: str,
        *,
        repo: str | None = None,
    ) -> dict[str, Any]:
        """Post a comment on an issue (or a pull request; PRs accept
        issue comments on the conversation timeline)."""
        repo = self._resolve_repo(repo)
        if not repo:
            return _err("no repo provided and no default_repo set")
        client, err = await self._ensure_client()
        if client is None:
            return _err(err or "")
        try:
            data = await client.post(
                f"/repos/{repo}/issues/{issue_number}/comments",
                json={"body": body},
            )
        except Exception as e:
            return self._shape_error(e)
        out = _ok(
            comment_id=data.get("id"), url=data.get("html_url"),
        )
        await self._write_audit("comment_on_issue", {
            "repo": repo, "number": issue_number,
            "comment_id": data.get("id"),
        })
        return out

    @action_executor()
    async def comment_as_session_agent(
        self,
        *,
        issue_number: int,
        body: str,
        session_id: str,
        replying_to: str | None = None,
        trigger: Literal[
            "chat", "scheduled", "mention", "automated",
        ] = "chat",
        run_id: str | None = None,
        src_comment_id: int | None = None,
        scheduled_mission_name: str | None = None,
        agent_role: str = "SessionAgent",
        repo: str | None = None,
    ) -> dict[str, Any]:
        """Post a comment with the Colony signing + attribution
        conventions baked in (top-level design plan §16.5).

        The planner should call THIS instead of bare
        :meth:`comment_on_issue` for any user-facing reply, so:

        - the comment carries the visible Colony signature line so
          humans reading the GitHub thread know what / who replied;
        - the comment carries the machine-readable HTML-comment
          footer the future GitHub-inbound poller / webhook (Phase
          P8) parses to join inbound replies back to their
          originating Colony session in the InteractionLog.

        ``trigger`` semantics:

        - ``chat`` (default) — replying to a user message in the
          chat UI. ``replying_to`` should be the user's GitHub login.
        - ``mention`` — replying because the bot was @-mentioned in
          a GitHub comment. Set ``src_comment_id`` to the comment
          that mentioned us.
        - ``scheduled`` — posted from a scheduled mission run. Set
          ``scheduled_mission_name`` so the signature reads
          "triggered by scheduled mission `<name>`".
        - ``automated`` — any other automated trigger; signature
          reads "(automated)".

        ``agent_role`` defaults to ``"SessionAgent"`` — override for
        other agent types (e.g. a coordinator agent posting status
        updates).
        """

        signature = _render_session_agent_signature(
            agent_role=agent_role,
            trigger=trigger,
            session_id=session_id,
            replying_to=replying_to,
            src_comment_id=src_comment_id,
            scheduled_mission_name=scheduled_mission_name,
        )
        footer = _render_attribution_footer(
            author="session-agent" if agent_role == "SessionAgent"
            else agent_role.lower().replace(" ", "-"),
            session_id=session_id,
            run_id=run_id,
            user=replying_to,
            trigger=trigger,
            src_comment_id=src_comment_id,
            scheduled_mission_name=scheduled_mission_name,
        )
        full_body = f"{signature}\n\n{body}\n\n{footer}"
        # Delegate to the bare action; it already handles repo
        # resolution, client init, error shaping, audit-write of the
        # ``comment_on_issue`` event. The audit row records the bare
        # call (the wrapper convention is captured in the on-wire
        # body via the footer, which the inbound parser will pull
        # out — no separate audit-row schema needed).
        return await self.comment_on_issue(
            issue_number, full_body, repo=repo,
        )

    @action_executor()
    async def close_issue(
        self,
        issue_number: int,
        *,
        repo: str | None = None,
        reason: Literal["completed", "not_planned"] = "completed",
    ) -> dict[str, Any]:
        """Close an issue with a state_reason."""
        repo = self._resolve_repo(repo)
        if not repo:
            return _err("no repo provided and no default_repo set")
        client, err = await self._ensure_client()
        if client is None:
            return _err(err or "")
        try:
            data = await client.patch(
                f"/repos/{repo}/issues/{issue_number}",
                json={"state": "closed", "state_reason": reason},
            )
        except Exception as e:
            return self._shape_error(e)
        await self._write_audit("close_issue", {
            "repo": repo, "number": issue_number, "reason": reason,
        })
        return _ok(issue=_summarise_issue(data))

    @action_executor()
    async def reopen_issue(
        self, issue_number: int, *, repo: str | None = None,
    ) -> dict[str, Any]:
        """Reopen a previously-closed issue."""
        repo = self._resolve_repo(repo)
        if not repo:
            return _err("no repo provided and no default_repo set")
        client, err = await self._ensure_client()
        if client is None:
            return _err(err or "")
        try:
            data = await client.patch(
                f"/repos/{repo}/issues/{issue_number}",
                json={"state": "open"},
            )
        except Exception as e:
            return self._shape_error(e)
        await self._write_audit("reopen_issue", {
            "repo": repo, "number": issue_number,
        })
        return _ok(issue=_summarise_issue(data))

    @action_executor()
    async def add_labels(
        self,
        issue_number: int,
        labels: list[str],
        *,
        repo: str | None = None,
    ) -> dict[str, Any]:
        """Add labels to an issue (additive; does not replace)."""
        repo = self._resolve_repo(repo)
        if not repo:
            return _err("no repo provided and no default_repo set")
        client, err = await self._ensure_client()
        if client is None:
            return _err(err or "")
        try:
            data = await client.post(
                f"/repos/{repo}/issues/{issue_number}/labels",
                json={"labels": list(labels)},
            )
        except Exception as e:
            return self._shape_error(e)
        return _ok(
            labels=[
                label.get("name") if isinstance(label, dict) else str(label)
                for label in (data or [])
            ],
        )

    @action_executor()
    async def list_milestones(
        self,
        *,
        repo: str | None = None,
        state: Literal["open", "closed", "all"] = "open",
        sort: Literal["due_on", "completeness"] = "due_on",
        direction: Literal["asc", "desc"] = "asc",
        max_results: int = 50,
    ) -> dict[str, Any]:
        """List repository milestones with their open/closed issue
        counts (provided by the GitHub API per-milestone — no
        per-milestone issue scan needed).

        Each returned entry has::

            {number, title, description, state, due_on,
             open_issues, closed_issues, html_url}

        Used by :meth:`DesignProcessCapability.summarise_progress`
        to roll milestone state up into a progress snapshot.
        """
        repo = self._resolve_repo(repo)
        if not repo:
            return _err(
                "no repo provided and no default_repo set",
                milestones=[],
            )
        client, err = await self._ensure_client()
        if client is None:
            return _err(err or "", milestones=[])
        params: dict[str, Any] = {
            "state": state,
            "sort": sort,
            "direction": direction,
            "per_page": min(100, max_results),
        }
        try:
            data = await client.get(
                f"/repos/{repo}/milestones", **params,
            )
        except Exception as e:
            return self._shape_error(e) | {"milestones": []}
        milestones = [
            {
                "number": m.get("number"),
                "title": m.get("title"),
                "description": m.get("description"),
                "state": m.get("state"),
                "due_on": m.get("due_on"),
                "open_issues": m.get("open_issues", 0),
                "closed_issues": m.get("closed_issues", 0),
                "html_url": m.get("html_url"),
            }
            for m in (data or [])
        ][:max_results]
        return _ok(milestones=milestones, count=len(milestones))

    @action_executor()
    async def whoami(self) -> dict[str, Any]:
        """Return the GitHub identity the capability authenticates as.

        Calls ``GET /user`` once against the per-tenant installation
        token (set up by P5) and caches the result on the capability
        — the App-bot identity is fixed for the lifetime of the
        installation, so we don't re-fetch per call. Returns
        ``{login, id, type, html_url, name, email}``; for
        installation-token auth, ``type`` is ``"Bot"`` and ``login``
        is ``<app-slug>[bot]``.
        """

        if self._identity_cache is not None:
            return self._identity_cache
        client, err = await self._ensure_client()
        if client is None:
            return _err(err or "")
        try:
            data = await client.get("/user")
        except Exception as e:
            return self._shape_error(e)
        result = _ok(
            login=data.get("login"),
            id=data.get("id"),
            type=data.get("type"),
            html_url=data.get("html_url"),
            name=data.get("name"),
            email=data.get("email"),
        )
        self._identity_cache = result
        return result

    @action_executor()
    async def assign_issue(
        self,
        issue_number: int,
        assignees: list[str],
        *,
        repo: str | None = None,
        replace: bool = True,
    ) -> dict[str, Any]:
        """Set the assignees on an issue.

        Args:
            issue_number: The issue number to assign.
            assignees: GitHub logins to assign. Bot logins take the
                form ``<app-slug>[bot]`` — use :meth:`whoami` to get
                Colony's bot login. Pass ``[]`` with ``replace=True``
                to unassign everyone.
            repo: ``owner/repo``; falls back to ``default_repo``.
            replace: If ``True`` (default), the issue's assignees are
                set to exactly ``assignees`` — anyone previously
                assigned and not in this list is removed. If
                ``False``, the list is additive: ``assignees`` is
                appended to whoever is already assigned (GitHub
                de-duplicates).

        Returns the updated issue summary.
        """
        repo = self._resolve_repo(repo)
        if not repo:
            return _err("no repo provided and no default_repo set")
        client, err = await self._ensure_client()
        if client is None:
            return _err(err or "")
        # Normalise; strip empties so callers can pass ``[None]`` etc.
        clean = [a for a in (assignees or []) if isinstance(a, str) and a.strip()]
        try:
            if replace:
                data = await client.patch(
                    f"/repos/{repo}/issues/{issue_number}",
                    json={"assignees": clean},
                )
            else:
                data = await client.post(
                    f"/repos/{repo}/issues/{issue_number}/assignees",
                    json={"assignees": clean},
                )
        except Exception as e:
            return self._shape_error(e)
        await self._write_audit("assign_issue", {
            "repo": repo, "issue_number": issue_number,
            "assignees": clean, "replace": replace,
        })
        return _ok(issue=_summarise_issue(data or {}))

    # =====================================================================
    # Pull requests
    # =====================================================================

    @action_executor()
    async def list_pull_requests(
        self,
        *,
        repo: str | None = None,
        state: Literal["open", "closed", "all"] = "open",
        max_results: int = 50,
    ) -> dict[str, Any]:
        """List pull requests."""
        repo = self._resolve_repo(repo)
        if not repo:
            return _err("no repo provided and no default_repo set", prs=[])
        client, err = await self._ensure_client()
        if client is None:
            return _err(err or "", prs=[])
        prs: list[dict[str, Any]] = []
        try:
            async for p in client.iter_paginated(
                f"/repos/{repo}/pulls", page_size=100, state=state,
            ):
                prs.append(_summarise_pr(p))
                if len(prs) >= max_results:
                    break
        except Exception as e:
            return self._shape_error(e) | {"prs": []}
        return _ok(prs=prs, count=len(prs))

    @action_executor()
    async def get_pull_request(
        self, number: int, *, repo: str | None = None,
    ) -> dict[str, Any]:
        """Fetch one pull request's metadata.

        Returns the same fields as ``list_pull_requests`` plus head /
        base refs, draft / merged / mergeable status — enough for the
        LLM to reason about whether to comment, review, or wait.
        """
        repo = self._resolve_repo(repo)
        if not repo:
            return _err("no repo provided and no default_repo set")
        client, err = await self._ensure_client()
        if client is None:
            return _err(err or "")
        try:
            data = await client.get(f"/repos/{repo}/pulls/{number}")
        except Exception as e:
            return self._shape_error(e)
        return _ok(pr=_summarise_pr(data))

    @action_executor()
    async def get_pr_diff(
        self,
        number: int,
        *,
        repo: str | None = None,
        max_bytes: int = 500_000,
    ) -> dict[str, Any]:
        """Fetch a PR's diff as plain text."""
        repo = self._resolve_repo(repo)
        if not repo:
            return _err("no repo provided and no default_repo set")
        client, err = await self._ensure_client()
        if client is None:
            return _err(err or "")
        try:
            # Temporarily override the Accept header via a custom call.
            # ``GitHubClient`` doesn't expose per-call header overrides,
            # so we go through ``_request`` + ``_raise_for_status``.
            resp = await client._request(
                "GET", f"/repos/{repo}/pulls/{number}",
            )
            # Re-issue with the diff media type; the client's token
            # plumbing handles 401-refresh and rate limits.
            diff_headers = await client._headers()
            diff_headers["Accept"] = "application/vnd.github.v3.diff"
            diff_resp = await client._client.get(
                f"{client._api_base}/repos/{repo}/pulls/{number}",
                headers=diff_headers,
            )
            if diff_resp.status_code >= 400:
                raise GitHubError(
                    f"diff fetch failed: {diff_resp.status_code}",
                    status_code=diff_resp.status_code,
                    body=diff_resp.text,
                )
        except Exception as e:
            return self._shape_error(e)
        diff = diff_resp.text
        truncated = False
        if len(diff) > max_bytes:
            diff = diff[:max_bytes]
            truncated = True
        return _ok(diff=diff, truncated=truncated, size=len(diff))

    @action_executor()
    async def create_pull_request(
        self,
        *,
        head: str,
        base: str,
        title: str,
        body: str,
        repo: str | None = None,
        draft: bool = False,
    ) -> dict[str, Any]:
        """Open a new PR."""
        repo = self._resolve_repo(repo)
        if not repo:
            return _err("no repo provided and no default_repo set")
        client, err = await self._ensure_client()
        if client is None:
            return _err(err or "")
        payload = {
            "head": head, "base": base, "title": title,
            "body": body, "draft": draft,
        }
        try:
            data = await client.post(f"/repos/{repo}/pulls", json=payload)
        except Exception as e:
            return self._shape_error(e)
        await self._write_audit("create_pull_request", {
            "repo": repo, "number": data.get("number"),
            "head": head, "base": base,
        })
        return _ok(pr=_summarise_pr(data))

    @action_executor()
    async def comment_on_pr(
        self,
        number: int,
        body: str,
        *,
        repo: str | None = None,
    ) -> dict[str, Any]:
        """Post a conversation-timeline comment on a PR."""
        return await self.comment_on_issue(
            issue_number=number, body=body, repo=repo,
        )

    @action_executor()
    async def review_pr(
        self,
        number: int,
        *,
        event: Literal["APPROVE", "REQUEST_CHANGES", "COMMENT"],
        body: str,
        repo: str | None = None,
        comments: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Submit a review.

        ``comments`` is forwarded verbatim to GitHub — each entry is a
        dict like ``{"path": "...", "position": 12, "body": "..."}``.
        """
        repo = self._resolve_repo(repo)
        if not repo:
            return _err("no repo provided and no default_repo set")
        client, err = await self._ensure_client()
        if client is None:
            return _err(err or "")
        payload: dict[str, Any] = {"event": event, "body": body}
        if comments:
            payload["comments"] = list(comments)
        try:
            data = await client.post(
                f"/repos/{repo}/pulls/{number}/reviews", json=payload,
            )
        except Exception as e:
            return self._shape_error(e)
        await self._write_audit("review_pr", {
            "repo": repo, "number": number, "event": event,
        })
        return _ok(
            review_id=data.get("id"), state=data.get("state"),
            url=data.get("html_url"),
        )

    @action_executor()
    async def get_pr_checks(
        self,
        number: int,
        *,
        repo: str | None = None,
    ) -> dict[str, Any]:
        """Summarise the check-runs on a PR's head commit."""
        repo = self._resolve_repo(repo)
        if not repo:
            return _err("no repo provided and no default_repo set")
        client, err = await self._ensure_client()
        if client is None:
            return _err(err or "")
        try:
            pr = await client.get(f"/repos/{repo}/pulls/{number}")
            sha = (pr.get("head") or {}).get("sha")
            if not sha:
                return _err("PR head SHA unavailable")
            data = await client.get(
                f"/repos/{repo}/commits/{sha}/check-runs",
            )
        except Exception as e:
            return self._shape_error(e)
        runs = [
            {
                "name": r.get("name"),
                "status": r.get("status"),
                "conclusion": r.get("conclusion"),
                "url": r.get("html_url"),
            }
            for r in (data.get("check_runs") or [])
        ]
        return _ok(runs=runs, count=len(runs))

    # =====================================================================
    # Projects v2 (GraphQL)
    # =====================================================================

    _PROJECTS_QUERY = """
    query($login: String!, $first: Int = 20) {
      viewer { organizations: login }
    }
    """.strip()

    # Idempotent server-side: re-adding the same content to a project
    # returns the existing item id rather than erroring. The mutation
    # accepts the project's GraphQL node id + the content's GraphQL
    # node id (issues + PRs both have a node_id REST returns as
    # ``node_id`` on the response payload).
    _ADD_PROJECT_V2_ITEM = """
    mutation($projectId: ID!, $contentId: ID!) {
      addProjectV2ItemById(input: {projectId: $projectId, contentId: $contentId}) {
        item { id }
      }
    }
    """.strip()

    _LIST_PROJECT_ITEMS = """
    query($projectId: ID!, $first: Int!) {
      node(id: $projectId) {
        ... on ProjectV2 {
          title
          items(first: $first) {
            nodes {
              id
              content {
                __typename
                ... on Issue { number title state url repository { nameWithOwner } }
                ... on PullRequest { number title state url repository { nameWithOwner } }
                ... on DraftIssue { title body }
              }
              fieldValues(first: 20) {
                nodes {
                  __typename
                  ... on ProjectV2ItemFieldSingleSelectValue {
                    field { ... on ProjectV2FieldCommon { name } }
                    name
                  }
                  ... on ProjectV2ItemFieldTextValue {
                    field { ... on ProjectV2FieldCommon { name } }
                    text
                  }
                }
              }
            }
          }
        }
      }
    }
    """

    @action_executor()
    async def list_project_items(
        self,
        project_id: str | None = None,
        *,
        max_items: int = 50,
    ) -> dict[str, Any]:
        """List items on a Projects v2 board.

        Each item reports its content (Issue / PR / Draft) plus every
        field value as ``{field_name: value}`` for the single-select
        and text fields the capability supports — enough to drive
        ``claim_unassigned_issue`` and status queries. More exotic
        field types can be added in a follow-up.
        """
        pid = project_id or self._default_project_id
        if not pid:
            return _err(
                "no project_id provided and no default_project_id set",
                items=[],
            )
        client, err = await self._ensure_client()
        if client is None:
            return _err(err or "", items=[])
        try:
            data = await client.graphql(
                self._LIST_PROJECT_ITEMS,
                variables={"projectId": pid, "first": max_items},
            )
        except Exception as e:
            return self._shape_error(e) | {"items": []}
        node = (data or {}).get("node") or {}
        items_raw = ((node.get("items") or {}).get("nodes") or [])
        items: list[dict[str, Any]] = []
        for it in items_raw:
            content = it.get("content") or {}
            field_values: dict[str, Any] = {}
            for fv in ((it.get("fieldValues") or {}).get("nodes") or []):
                name = ((fv or {}).get("field") or {}).get("name")
                if not name:
                    continue
                if "name" in fv:  # single-select
                    field_values[name] = fv["name"]
                elif "text" in fv:
                    field_values[name] = fv["text"]
            items.append({
                "item_id": it.get("id"),
                "content_type": content.get("__typename"),
                "number": content.get("number"),
                "title": content.get("title"),
                "url": content.get("url"),
                "repository": (content.get("repository") or {}).get(
                    "nameWithOwner",
                ),
                "field_values": field_values,
            })
        return _ok(
            project_title=node.get("title"),
            items=items, count=len(items),
        )

    async def _add_to_project_v2(
        self,
        *,
        client: "GitHubClient",
        project_id: str,
        content_node_id: str,
    ) -> dict[str, Any]:
        """Run the ``addProjectV2ItemById`` mutation. Returns the
        standard ``{ok, message, item_id?}`` envelope. Server-side
        idempotent — re-adding the same content returns the existing
        item id.

        Internal helper; callers should be public actions
        (``create_issue`` auto-attach, ``add_issue_to_project``,
        future ``add_pr_to_project``) so the audit log carries an
        action label.
        """

        try:
            data = await client.graphql(
                self._ADD_PROJECT_V2_ITEM,
                variables={
                    "projectId": project_id,
                    "contentId": content_node_id,
                },
            )
        except Exception as e:  # noqa: BLE001
            return self._shape_error(e)
        item = ((data or {}).get("addProjectV2ItemById") or {}).get("item") or {}
        item_id = item.get("id")
        if not item_id:
            return _err(
                "addProjectV2ItemById returned no item id (unexpected "
                "GraphQL shape); the issue may not have been attached.",
            )
        return _ok(item_id=item_id)

    @action_executor()
    async def add_issue_to_project(
        self,
        issue_number: int,
        *,
        project_id: str | None = None,
        repo: str | None = None,
    ) -> dict[str, Any]:
        """Attach an existing issue to a Projects v2 board.

        ``project_id`` falls back to the capability's
        ``default_project_id`` when omitted. Resolves the issue's
        GraphQL node id via REST first (one round-trip; cheaper than
        a GraphQL ``repository.issue.id`` lookup) and then runs the
        ``addProjectV2ItemById`` mutation. Idempotent server-side
        (re-adding returns the existing item id).
        """

        pid = project_id or self._default_project_id
        if not pid:
            return _err(
                "no project_id provided and no default_project_id set",
            )
        repo = self._resolve_repo(repo)
        if not repo:
            return _err("no repo provided and no default_repo set")
        client, err = await self._ensure_client()
        if client is None:
            return _err(err or "")
        try:
            issue_data = await client.get(
                f"/repos/{repo}/issues/{issue_number}",
            )
        except Exception as e:
            return self._shape_error(e)
        content_node_id = issue_data.get("node_id")
        if not content_node_id:
            return _err(
                f"issue {repo}#{issue_number} REST response had no "
                f"node_id; cannot run addProjectV2ItemById without it.",
            )
        result = await self._add_to_project_v2(
            client=client, project_id=pid,
            content_node_id=content_node_id,
        )
        if result["ok"]:
            await self._write_audit("add_issue_to_project", {
                "repo": repo, "number": issue_number,
                "project_id": pid, "item_id": result.get("item_id"),
            })
        return result | {"repo": repo, "number": issue_number, "project_id": pid}

    @action_executor(interruptible=True)
    async def claim_unassigned_issue(
        self,
        *,
        repo: str | None = None,
        label: str | None = None,
        max_candidates: int = 25,
    ) -> dict[str, Any]:
        """Atomically claim one unassigned issue.

        Race semantics: the capability lists candidate open issues
        that match the filters, then iterates them in order and tries
        to apply a ``claimed-by:<agent_id>`` label. A concurrent caller
        that already applied its own ``claimed-by:*`` label is
        detected on re-fetch and skipped. We stop at the first issue
        we successfully claim; ``None`` means nothing was available.

        Returns:
            On success: ``{"ok": True, "claimed": True, "issue": {...},
            "label_applied": "claimed-by:<agent_id>"}``.
            When nothing is available: ``{"ok": True, "claimed": False,
            "issue": None}``.
        """
        repo = self._resolve_repo(repo)
        if not repo:
            return _err(
                "no repo provided and no default_repo set", claimed=False,
            )
        client, err = await self._ensure_client()
        if client is None:
            return _err(err or "", claimed=False)
        params: dict[str, Any] = {
            "state": "open", "per_page": max_candidates,
            "assignee": "none",
        }
        if label:
            params["labels"] = label
        agent_label = self._label_for_agent(self._agent_id())
        try:
            candidates = await client.get(
                f"/repos/{repo}/issues", **params,
            )
        except Exception as e:
            return self._shape_error(e) | {"claimed": False}
        for it in (candidates or []):
            if it.get("pull_request"):
                continue
            number = it.get("number")
            existing_labels = {
                (label_.get("name") if isinstance(label_, dict) else str(label_))
                for label_ in (it.get("labels") or [])
            }
            if any(
                lbl.startswith(self._CLAIMED_BY_LABEL_PREFIX)
                for lbl in existing_labels
            ):
                continue
            # Apply our label. Conflicts surface as errors from add_labels.
            label_resp = await self.add_labels(
                issue_number=number, labels=[agent_label], repo=repo,
            )
            if not label_resp.get("ok"):
                continue
            # Re-fetch and confirm we are the sole claimant.
            try:
                refetched = await client.get(
                    f"/repos/{repo}/issues/{number}",
                )
            except Exception as e:
                return self._shape_error(e) | {"claimed": False}
            claims = [
                (label_.get("name") if isinstance(label_, dict) else str(label_))
                for label_ in (refetched.get("labels") or [])
                if (
                    label_.get("name") if isinstance(label_, dict)
                    else str(label_)
                ).startswith(self._CLAIMED_BY_LABEL_PREFIX)
            ]
            if claims == [agent_label]:
                await self._write_audit("claim_issue", {
                    "repo": repo, "number": number, "label": agent_label,
                })
                return _ok(
                    claimed=True,
                    issue=_summarise_issue(refetched),
                    label_applied=agent_label,
                )
            # Someone beat us to it — roll back our label before
            # moving on.
            try:
                await client.delete(
                    f"/repos/{repo}/issues/{number}/labels/{agent_label}",
                )
            except Exception:  # pragma: no cover — best-effort
                pass
        return _ok(claimed=False, issue=None)

    @action_executor()
    async def release_claim(
        self,
        issue_number: int,
        *,
        repo: str | None = None,
    ) -> dict[str, Any]:
        """Release this agent's ``claimed-by:`` label on an issue."""
        repo = self._resolve_repo(repo)
        if not repo:
            return _err("no repo provided and no default_repo set")
        client, err = await self._ensure_client()
        if client is None:
            return _err(err or "")
        agent_label = self._label_for_agent(self._agent_id())
        try:
            await client.delete(
                f"/repos/{repo}/issues/{issue_number}/labels/{agent_label}",
            )
        except Exception as e:
            return self._shape_error(e)
        await self._write_audit("release_claim", {
            "repo": repo, "number": issue_number, "label": agent_label,
        })
        return _ok()
