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
import os
import time
import uuid
from typing import Any, Literal, TYPE_CHECKING
from overrides import override

import httpx

from ...base import AgentCapability
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
        installation_id: GitHub App installation id. Falls back to
            ``GITHUB_INSTALLATION_ID``.
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
        if client is not None:
            self._client: GitHubClient | None = client
            self._init_error: str | None = None
            return
        # Build a live client; any configuration error is captured so
        # the action surface can return it as a normal error dict
        # rather than raising at construction time.
        try:
            self._client = self._build_live_client(
                app_id=app_id,
                private_key_pem=private_key_pem,
                private_key_path=private_key_path,
                installation_id=installation_id,
            )
            self._init_error = None
        except Exception as e:
            self._client = None
            self._init_error = str(e)
            logger.warning(
                "GitHubCapability: live client disabled: %s", e,
            )

    # --- Internal construction --------------------------------------------

    def _build_live_client(
        self,
        *,
        app_id: str | None,
        private_key_pem: str | None,
        private_key_path: str | None,
        installation_id: str | None,
    ) -> GitHubClient:
        app_id = app_id or os.environ.get("GITHUB_APP_ID")
        installation_id = (
            installation_id or os.environ.get("GITHUB_INSTALLATION_ID")
        )
        private_key_pem = private_key_pem or os.environ.get(
            "GITHUB_PRIVATE_KEY_PEM",
        )
        if not private_key_pem and private_key_path:
            with open(private_key_path, "r", encoding="utf-8") as fh:
                private_key_pem = fh.read()
        if not app_id or not installation_id or not private_key_pem:
            raise RuntimeError(
                "GitHubCapability: app_id, installation_id, and a "
                "private key are all required (explicit kwargs, "
                "environment variables, or private_key_path)."
            )
        if self._httpx_client is None:
            self._httpx_client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=10.0, read=30.0, write=10.0, pool=10.0,
                ),
            )
        auth = GitHubAppAuth(
            app_id=app_id, private_key_pem=private_key_pem,
        )
        tokens = TokenCache(
            app_auth=auth,
            installation_id=installation_id,
            client=self._httpx_client,
        )
        return GitHubClient(tokens=tokens, client=self._httpx_client)

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

    def _ensure_client(self) -> tuple[GitHubClient | None, str | None]:
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
        client, err = self._ensure_client()
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
        client, err = self._ensure_client()
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
        client, err = self._ensure_client()
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
        client, err = self._ensure_client()
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
        client, err = self._ensure_client()
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
        client, err = self._ensure_client()
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
        client, err = self._ensure_client()
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
    ) -> dict[str, Any]:
        """Open a new issue."""
        repo = self._resolve_repo(repo)
        if not repo:
            return _err("no repo provided and no default_repo set")
        client, err = self._ensure_client()
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
        client, err = self._ensure_client()
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
        client, err = self._ensure_client()
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
        client, err = self._ensure_client()
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
        client, err = self._ensure_client()
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
        client, err = self._ensure_client()
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
        client, err = self._ensure_client()
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
        client, err = self._ensure_client()
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
        client, err = self._ensure_client()
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
        client, err = self._ensure_client()
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
        client, err = self._ensure_client()
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
        client, err = self._ensure_client()
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
        client, err = self._ensure_client()
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
        client, err = self._ensure_client()
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
