"""Unit tests for ``GitHubCapability``.

No network. HTTP is stubbed via ``httpx.MockTransport`` and the
capability is constructed with a pre-built ``GitHubClient`` pointing
at the stub — this short-circuits real auth, so the tests exercise
the action surface end-to-end without any GitHub App credentials.

A separate set of tests exercises the ``GitHubAppAuth`` JWT payload
and the ``TokenCache`` refresh semantics against the stubbed
``/access_tokens`` endpoint.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from unittest.mock import MagicMock

import httpx
import jwt
import pytest
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from polymathera.colony.agents.blackboard.protocol import GitHubEventProtocol
from polymathera.colony.agents.patterns.capabilities._github import (
    GitHubAppAuth,
    GitHubClient,
    NotFoundError,
    RateLimitError,
    TokenCache,
)
from polymathera.colony.agents.patterns.capabilities.github import (
    GitHubCapability,
)
from polymathera.colony.agents.scopes import BlackboardScope
from polymathera.colony.distributed.ray_utils.serving.context import (
    execution_context, Ring,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_rsa_key_pem() -> tuple[str, str]:
    key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048, backend=default_backend(),
    )
    priv = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    pub = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormatSPKI if False else serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return priv, pub


class _StubClient:
    """Tiny capture helper for tests that don't need real HTTP."""

    def __init__(self):
        self.calls: list[tuple[str, str, dict, Any]] = []
        self.responses: dict[tuple[str, str], Any] = {}
        self.raises: dict[tuple[str, str], Exception] = {}
        self._closed = False
        self._api_base = "https://api.github.com"
        self._graphql_url = "https://api.github.com/graphql"

    def _handle(self, method, path, **kw):
        self.calls.append((method, path, dict(kw), kw.get("json")))
        key = (method, path)
        if key in self.raises:
            raise self.raises[key]
        return self.responses.get(key)

    async def get(self, path, **params):
        return self._handle("GET", path, params=params)

    async def post(self, path, *, json=None):
        return self._handle("POST", path, json=json)

    async def patch(self, path, *, json=None):
        return self._handle("PATCH", path, json=json)

    async def put(self, path, *, json=None):
        return self._handle("PUT", path, json=json)

    async def delete(self, path):
        return self._handle("DELETE", path)

    async def iter_paginated(self, path, *, page_size=100, **params):
        items = self.responses.get(("ITER", path), [])
        if isinstance(items, Exception):
            raise items
        for it in items:
            yield it

    async def graphql(self, query, *, variables=None):
        return self._handle("GRAPHQL", query[:40], json=variables)

    async def close(self):
        self._closed = True

    async def _request(self, method, path, *, params=None, json=None):
        # Used by ``get_pr_diff``. Return a stub response object so the
        # capability's second call (through self._client._client.get)
        # hits the MockTransport path — tests that exercise diff use
        # the real GitHubClient backed by MockTransport instead.
        raise NotImplementedError("use GitHubClient+MockTransport for diff")


def _mock_transport(
    handler: "callable[[httpx.Request], httpx.Response]",
) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def _make_cap(
    *, client: Any = None,
) -> GitHubCapability:
    agent = MagicMock()
    agent.agent_id = "agent-A"
    cap = GitHubCapability(
        agent=agent,
        scope=BlackboardScope.SESSION,
        client=client,
        default_repo="acme/proj",
    )
    return cap


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _with_context():
    return execution_context(
        ring=Ring.USER, tenant_id="t1", colony_id="c1", session_id="s1",
    )


# ---------------------------------------------------------------------------
# GitHubAppAuth + TokenCache
# ---------------------------------------------------------------------------

def test_app_auth_mints_jwt_with_expected_claims():
    priv, _ = _make_rsa_key_pem()
    auth = GitHubAppAuth(app_id="123", private_key_pem=priv)
    token = auth.mint_jwt(ttl_s=600)
    decoded = jwt.decode(
        token, options={"verify_signature": False},
    )
    assert decoded["iss"] == "123"
    assert decoded["exp"] - decoded["iat"] == 660  # 600 + 60s backdating


def test_app_auth_rejects_missing_inputs():
    with pytest.raises(ValueError):
        GitHubAppAuth(app_id="", private_key_pem="x")
    with pytest.raises(ValueError):
        GitHubAppAuth(app_id="x", private_key_pem="")


def test_token_cache_refreshes_when_expiring():
    priv, _ = _make_rsa_key_pem()

    call_counter = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_counter["n"] += 1
        # Expires 6 minutes from now so the pad (5 min) means the
        # second call crosses the threshold.
        return httpx.Response(201, json={
            "token": f"tok_{call_counter['n']}",
            "expires_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(time.time() + 6 * 60),
            ),
        })

    async def run():
        client = httpx.AsyncClient(transport=_mock_transport(handler))
        auth = GitHubAppAuth(app_id="1", private_key_pem=priv)
        cache = TokenCache(
            app_auth=auth, installation_id="42", client=client,
        )
        t1 = await cache.get()
        t1b = await cache.get()  # cached
        # Force-refresh crosses the pad and mints a new one.
        t2 = await cache.get(force_refresh=True)
        await client.aclose()
        return t1, t1b, t2

    t1, t1b, t2 = asyncio.get_event_loop().run_until_complete(run())
    assert t1 == t1b
    assert t2 != t1
    assert call_counter["n"] == 2


def test_token_cache_surfaces_exchange_errors():
    priv, _ = _make_rsa_key_pem()

    def handler(request):
        return httpx.Response(401, json={"message": "bad credentials"})

    async def run():
        client = httpx.AsyncClient(transport=_mock_transport(handler))
        auth = GitHubAppAuth(app_id="1", private_key_pem=priv)
        cache = TokenCache(
            app_auth=auth, installation_id="42", client=client,
        )
        try:
            await cache.get()
        finally:
            await client.aclose()

    with pytest.raises(RuntimeError, match="installation-token exchange"):
        asyncio.get_event_loop().run_until_complete(run())


# ---------------------------------------------------------------------------
# GitHubClient retry and rate-limit behaviour (against MockTransport)
# ---------------------------------------------------------------------------

def test_client_refreshes_token_on_401_once():
    priv, _ = _make_rsa_key_pem()
    calls = {"install_tokens": 0, "api": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "access_tokens" in url:
            calls["install_tokens"] += 1
            return httpx.Response(201, json={
                "token": f"tok_{calls['install_tokens']}",
                "expires_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ",
                    time.gmtime(time.time() + 3600),
                ),
            })
        calls["api"] += 1
        if calls["api"] == 1:
            return httpx.Response(401, json={"message": "bad creds"})
        return httpx.Response(200, json={"hello": "world"})

    async def run():
        http_client = httpx.AsyncClient(transport=_mock_transport(handler))
        auth = GitHubAppAuth(app_id="1", private_key_pem=priv)
        tokens = TokenCache(
            app_auth=auth, installation_id="42", client=http_client,
        )
        client = GitHubClient(tokens=tokens, client=http_client)
        data = await client.get("/repos/acme/proj")
        await client.close()
        return data

    data = asyncio.get_event_loop().run_until_complete(run())
    assert data == {"hello": "world"}
    assert calls["install_tokens"] == 2  # original + force-refresh
    assert calls["api"] == 2


def test_client_raises_rate_limit_error_on_primary_exhaustion():
    priv, _ = _make_rsa_key_pem()

    def handler(request: httpx.Request) -> httpx.Response:
        if "access_tokens" in str(request.url):
            return httpx.Response(201, json={
                "token": "t",
                "expires_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ",
                    time.gmtime(time.time() + 3600),
                ),
            })
        return httpx.Response(
            403,
            headers={"x-ratelimit-remaining": "0"},
            json={"message": "API rate limit exceeded"},
        )

    async def run():
        http_client = httpx.AsyncClient(transport=_mock_transport(handler))
        auth = GitHubAppAuth(app_id="1", private_key_pem=priv)
        tokens = TokenCache(
            app_auth=auth, installation_id="42", client=http_client,
        )
        client = GitHubClient(tokens=tokens, client=http_client)
        try:
            await client.get("/repos/x/y")
        finally:
            await client.close()

    with pytest.raises(RateLimitError):
        asyncio.get_event_loop().run_until_complete(run())


def test_client_iter_paginated_stops_when_page_is_short():
    priv, _ = _make_rsa_key_pem()

    def handler(request: httpx.Request) -> httpx.Response:
        if "access_tokens" in str(request.url):
            return httpx.Response(201, json={
                "token": "t",
                "expires_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ",
                    time.gmtime(time.time() + 3600),
                ),
            })
        page = int(request.url.params.get("page", "1"))
        if page == 1:
            # Full page
            return httpx.Response(200, json=[{"i": n} for n in range(100)])
        if page == 2:
            return httpx.Response(200, json=[{"i": 100}, {"i": 101}])
        return httpx.Response(200, json=[])

    async def run():
        http_client = httpx.AsyncClient(transport=_mock_transport(handler))
        auth = GitHubAppAuth(app_id="1", private_key_pem=priv)
        tokens = TokenCache(
            app_auth=auth, installation_id="42", client=http_client,
        )
        client = GitHubClient(tokens=tokens, client=http_client)
        out = []
        async for item in client.iter_paginated("/items"):
            out.append(item)
        await client.close()
        return out

    items = asyncio.get_event_loop().run_until_complete(run())
    assert len(items) == 102


# ---------------------------------------------------------------------------
# Capability — action surface
# ---------------------------------------------------------------------------

def test_list_repos_returns_installation_repositories():
    stub = _StubClient()
    stub.responses[("GET", "/installation/repositories")] = {
        "repositories": [
            {"full_name": "acme/a", "private": False,
             "default_branch": "main", "description": "",
             "html_url": "u"},
            {"full_name": "acme/b", "private": True,
             "default_branch": "main", "description": "",
             "html_url": "u"},
        ],
    }
    with _with_context():
        cap = _make_cap(client=stub)
        result = _run(cap.list_repos())
    assert result["ok"] is True
    assert result["count"] == 2
    assert [r["full_name"] for r in result["repos"]] == ["acme/a", "acme/b"]


def test_get_repo_uses_default_repo_when_none_given():
    stub = _StubClient()
    stub.responses[("GET", "/repos/acme/proj")] = {
        "full_name": "acme/proj", "default_branch": "main",
        "description": "d", "open_issues_count": 3,
        "stargazers_count": 10, "html_url": "u",
    }
    with _with_context():
        cap = _make_cap(client=stub)
        result = _run(cap.get_repo())
    assert result["ok"] is True
    assert result["repo"]["full_name"] == "acme/proj"


def test_get_repo_reports_missing_default():
    stub = _StubClient()
    agent = MagicMock(); agent.agent_id = "a"
    with _with_context():
        cap = GitHubCapability(
            agent=agent, scope=BlackboardScope.SESSION,
            client=stub, default_repo=None,
        )
        result = _run(cap.get_repo())
    assert result["ok"] is False
    assert "no repo" in result["message"]


def test_list_issues_excludes_pull_requests():
    stub = _StubClient()
    stub.responses[("ITER", "/repos/acme/proj/issues")] = [
        {"number": 1, "title": "issue",
         "state": "open", "user": {"login": "alice"},
         "labels": [], "assignees": []},
        {"number": 2, "title": "pr", "state": "open",
         "user": {"login": "bob"}, "labels": [], "assignees": [],
         "pull_request": {"url": "..."}},
    ]
    with _with_context():
        cap = _make_cap(client=stub)
        result = _run(cap.list_issues())
    assert result["count"] == 1
    assert result["issues"][0]["number"] == 1


def test_create_issue_writes_audit_record():
    stub = _StubClient()
    stub.responses[("POST", "/repos/acme/proj/issues")] = {
        "number": 7, "title": "t", "state": "open",
        "user": {"login": "a"}, "labels": [], "assignees": [],
    }
    bb_writes: list[tuple[str, Any]] = []

    class _Bb:
        async def write(self, k, v):
            bb_writes.append((k, v))

    with _with_context():
        cap = _make_cap(client=stub)
        cap._blackboard = _Bb()
        result = _run(cap.create_issue(title="t", body="b"))
    assert result["ok"] is True
    assert result["issue"]["number"] == 7
    audit = [w for w in bb_writes if w[0].startswith("audit:github:")]
    assert len(audit) == 1
    assert audit[0][1]["action"] == "create_issue"


def test_close_issue_uses_state_reason():
    stub = _StubClient()
    stub.responses[("PATCH", "/repos/acme/proj/issues/5")] = {
        "number": 5, "state": "closed", "state_reason": "completed",
        "user": {"login": "a"}, "labels": [], "assignees": [],
    }
    with _with_context():
        cap = _make_cap(client=stub)
        result = _run(cap.close_issue(issue_number=5, reason="not_planned"))
    method, path, kw, body = stub.calls[0]
    assert method == "PATCH"
    assert body == {"state": "closed", "state_reason": "not_planned"}
    assert result["ok"] is True


def test_comment_on_pr_delegates_to_issue_comment_endpoint():
    stub = _StubClient()
    stub.responses[("POST", "/repos/acme/proj/issues/9/comments")] = {
        "id": 123, "html_url": "u",
    }
    with _with_context():
        cap = _make_cap(client=stub)
        result = _run(cap.comment_on_pr(number=9, body="lgtm"))
    assert result["ok"] is True
    assert stub.calls[0][1] == "/repos/acme/proj/issues/9/comments"


def test_review_pr_forwards_event_and_comments():
    stub = _StubClient()
    stub.responses[("POST", "/repos/acme/proj/pulls/3/reviews")] = {
        "id": 55, "state": "APPROVED", "html_url": "u",
    }
    with _with_context():
        cap = _make_cap(client=stub)
        result = _run(cap.review_pr(
            number=3, event="APPROVE", body="lgtm",
            comments=[{"path": "x.py", "position": 1, "body": "nit"}],
        ))
    method, path, kw, body = stub.calls[0]
    assert body["event"] == "APPROVE"
    assert body["comments"][0]["path"] == "x.py"
    assert result["ok"] is True


def test_list_project_items_flattens_field_values():
    stub = _StubClient()
    # Key the stub on the actual 40-char prefix of the capability's query.
    key = ("GRAPHQL", GitHubCapability._LIST_PROJECT_ITEMS[:40])
    stub.responses[key] = {
        "node": {
            "title": "Board",
            "items": {
                "nodes": [
                    {
                        "id": "item_1",
                        "content": {
                            "__typename": "Issue",
                            "number": 10, "title": "X",
                            "state": "OPEN", "url": "u",
                            "repository": {"nameWithOwner": "acme/proj"},
                        },
                        "fieldValues": {
                            "nodes": [
                                {
                                    "__typename": "ProjectV2ItemFieldSingleSelectValue",
                                    "field": {"name": "Status"},
                                    "name": "In Progress",
                                },
                                {
                                    "__typename": "ProjectV2ItemFieldTextValue",
                                    "field": {"name": "Notes"},
                                    "text": "ready to merge",
                                },
                            ],
                        },
                    },
                ],
            },
        },
    }
    with _with_context():
        cap = _make_cap(client=stub)
        result = _run(cap.list_project_items(project_id="PVT_1"))
    assert result["count"] == 1
    item = result["items"][0]
    assert item["field_values"]["Status"] == "In Progress"
    assert item["field_values"]["Notes"] == "ready to merge"


# ---------------------------------------------------------------------------
# claim_unassigned_issue
# ---------------------------------------------------------------------------

def test_claim_unassigned_issue_applies_label_and_confirms():
    stub = _StubClient()
    stub.responses[("GET", "/repos/acme/proj/issues")] = [
        {"number": 11, "title": "t", "state": "open", "labels": [],
         "user": {"login": "a"}, "assignees": []},
    ]
    # add_labels goes through stub.post.
    stub.responses[("POST", "/repos/acme/proj/issues/11/labels")] = [
        {"name": "claimed-by:agent-A"},
    ]
    # Re-fetch shows our label only.
    stub.responses[("GET", "/repos/acme/proj/issues/11")] = {
        "number": 11, "title": "t", "state": "open",
        "labels": [{"name": "claimed-by:agent-A"}],
        "user": {"login": "a"}, "assignees": [],
    }
    with _with_context():
        cap = _make_cap(client=stub)
        result = _run(cap.claim_unassigned_issue())
    assert result["claimed"] is True
    assert result["label_applied"] == "claimed-by:agent-A"
    assert result["issue"]["number"] == 11


def test_claim_unassigned_issue_skips_already_claimed():
    stub = _StubClient()
    stub.responses[("GET", "/repos/acme/proj/issues")] = [
        {"number": 11, "title": "t", "state": "open",
         "labels": [{"name": "claimed-by:someone-else"}],
         "user": {"login": "a"}, "assignees": []},
        {"number": 12, "title": "t2", "state": "open", "labels": [],
         "user": {"login": "a"}, "assignees": []},
    ]
    stub.responses[("POST", "/repos/acme/proj/issues/12/labels")] = [
        {"name": "claimed-by:agent-A"},
    ]
    stub.responses[("GET", "/repos/acme/proj/issues/12")] = {
        "number": 12, "title": "t2", "state": "open",
        "labels": [{"name": "claimed-by:agent-A"}],
        "user": {"login": "a"}, "assignees": [],
    }
    with _with_context():
        cap = _make_cap(client=stub)
        result = _run(cap.claim_unassigned_issue())
    assert result["issue"]["number"] == 12


def test_claim_unassigned_issue_rolls_back_on_race():
    """A concurrent agent slipped in between our label-POST and the
    re-fetch; the capability must release its own label and move on."""
    stub = _StubClient()
    stub.responses[("GET", "/repos/acme/proj/issues")] = [
        {"number": 13, "title": "race", "state": "open", "labels": [],
         "user": {"login": "a"}, "assignees": []},
    ]
    stub.responses[("POST", "/repos/acme/proj/issues/13/labels")] = [
        {"name": "claimed-by:agent-A"},
    ]
    # Re-fetch shows both our label and another — we lost the race.
    stub.responses[("GET", "/repos/acme/proj/issues/13")] = {
        "number": 13, "title": "race", "state": "open",
        "labels": [
            {"name": "claimed-by:agent-A"},
            {"name": "claimed-by:agent-B"},
        ],
        "user": {"login": "a"}, "assignees": [],
    }
    # The capability then deletes its own label.
    stub.responses[
        ("DELETE", "/repos/acme/proj/issues/13/labels/claimed-by:agent-A")
    ] = None
    with _with_context():
        cap = _make_cap(client=stub)
        result = _run(cap.claim_unassigned_issue())
    assert result["claimed"] is False
    # Verify the rollback happened.
    assert any(
        method == "DELETE" and "labels/claimed-by:agent-A" in path
        for method, path, _, _ in stub.calls
    )


def test_claim_unassigned_issue_returns_nothing_when_no_candidates():
    stub = _StubClient()
    stub.responses[("GET", "/repos/acme/proj/issues")] = []
    with _with_context():
        cap = _make_cap(client=stub)
        result = _run(cap.claim_unassigned_issue())
    assert result["claimed"] is False
    assert result["issue"] is None


# ---------------------------------------------------------------------------
# GitHubEventProtocol
# ---------------------------------------------------------------------------

def test_event_protocol_round_trips_repo_and_number():
    key = GitHubEventProtocol.issue_opened_key("owner/repo", 42)
    assert key == "github:issue_opened:owner/repo:42"
    repo, number = GitHubEventProtocol.parse_issue_opened_key(key)
    assert repo == "owner/repo"
    assert number == 42


def test_event_protocol_rejects_foreign_prefix():
    with pytest.raises(ValueError):
        GitHubEventProtocol.parse_issue_opened_key("github:pr_opened:x:1")


def test_event_protocol_project_item_round_trip():
    key = GitHubEventProtocol.project_item_key("PVT_1", "I_abc")
    pid, iid = GitHubEventProtocol.parse_project_item_key(key)
    assert (pid, iid) == ("PVT_1", "I_abc")


# ---------------------------------------------------------------------------
# Init failure + blueprint
# ---------------------------------------------------------------------------

def test_missing_credentials_captures_init_error():
    """When the live client cannot be built, actions must report a
    clean error dict rather than crashing the agent."""
    agent = MagicMock(); agent.agent_id = "a"
    with _with_context():
        # No app_id, no env vars, no private key — init fails.
        cap = GitHubCapability(
            agent=agent, scope=BlackboardScope.SESSION,
        )
        result = _run(cap.list_repos())
    assert result["ok"] is False
    assert "app_id" in result["message"] or "required" in result["message"]


def test_action_executors_are_registered():
    import inspect
    keys = {
        m._action_key for _, m in inspect.getmembers(
            GitHubCapability, predicate=inspect.isfunction,
        ) if getattr(m, "_action_key", None)
    }
    expected = {
        # repo + content
        "list_repos", "get_repo", "list_branches",
        "get_file_contents", "search_code",
        # issues
        "list_issues", "get_issue", "create_issue",
        "comment_on_issue", "comment_as_session_agent",
        "close_issue", "reopen_issue", "add_labels",
        "list_milestones",
        # PRs
        "list_pull_requests", "get_pull_request",
        "get_pr_diff", "create_pull_request",
        "comment_on_pr", "review_pr", "get_pr_checks",
        # projects + coordination
        "list_project_items", "list_projects_for_repo",
        "add_issue_to_project",
        "claim_unassigned_issue", "release_claim",
        # P5d: identity + assignment
        "whoami", "assign_issue",
    }
    assert keys == expected


def test_bind_round_trips_through_cloudpickle():
    # Use Ray's vendored cloudpickle — that's what Ray's IPC actually
    # serializes through, so it is the right library to verify
    # bind-record compatibility against. Standalone PyPI ``cloudpickle``
    # is not a Ray dep (Ray vendors its own copy) and is therefore not
    # guaranteed to be installed.
    from ray import cloudpickle
    bp = GitHubCapability.bind(scope=BlackboardScope.SESSION)
    bp2 = cloudpickle.loads(cloudpickle.dumps(bp))
    assert bp2.cls is GitHubCapability


# ---------------------------------------------------------------------------
# P4: Projects v2 attachment (create_issue auto-attach + add_issue_to_project)
# ---------------------------------------------------------------------------


def _make_cap_with_default_project(
    *, client: Any = None, default_project_id: str | None = "PVT_123",
) -> GitHubCapability:
    agent = MagicMock()
    agent.agent_id = "agent-A"
    return GitHubCapability(
        agent=agent,
        scope=BlackboardScope.SESSION,
        client=client,
        default_repo="acme/proj",
        default_project_id=default_project_id,
    )


def test_create_issue_auto_attaches_to_default_project():
    """When a default_project_id is set + the flag is True (default),
    create_issue runs the addProjectV2ItemById mutation right after
    the REST POST and surfaces the new project item id in the
    response."""

    stub = _StubClient()
    stub.responses[("POST", "/repos/acme/proj/issues")] = {
        "number": 7, "node_id": "I_kwDO123",
        "title": "t", "state": "open",
        "user": {"login": "a"}, "labels": [], "assignees": [],
    }
    # The mutation key in _StubClient is ("GRAPHQL", query[:40]) — the
    # _ADD_PROJECT_V2_ITEM mutation starts with "mutation($projectId:
    # ID!, $contentId: ID!) {...".
    mutation_key = (
        "GRAPHQL", "mutation($projectId: ID!, $contentId: ID!"[:40],
    )
    stub.responses[mutation_key] = {
        "addProjectV2ItemById": {"item": {"id": "PVTI_ABC"}},
    }
    with _with_context():
        cap = _make_cap_with_default_project(client=stub)
        result = _run(cap.create_issue(title="t", body="b"))
    assert result["ok"] is True
    assert result["issue"]["number"] == 7
    assert result["project_item_id"] == "PVTI_ABC"
    assert result["project_id"] == "PVT_123"
    assert "project_attach_error" not in result

    # GraphQL mutation was actually called with the right variables.
    graphql_calls = [c for c in stub.calls if c[0] == "GRAPHQL"]
    assert len(graphql_calls) == 1
    _, _, _kw, variables = graphql_calls[0]
    assert variables == {"projectId": "PVT_123", "contentId": "I_kwDO123"}


def test_create_issue_skips_attach_when_flag_off():
    """``auto_attach_to_default_project=False`` skips the GraphQL
    call entirely even when a default_project_id is set."""

    stub = _StubClient()
    stub.responses[("POST", "/repos/acme/proj/issues")] = {
        "number": 7, "node_id": "I_kwDO123",
        "title": "t", "state": "open",
        "user": {"login": "a"}, "labels": [], "assignees": [],
    }
    with _with_context():
        cap = _make_cap_with_default_project(client=stub)
        result = _run(cap.create_issue(
            title="t", body="b", auto_attach_to_default_project=False,
        ))
    assert result["ok"] is True
    assert "project_item_id" not in result
    graphql_calls = [c for c in stub.calls if c[0] == "GRAPHQL"]
    assert graphql_calls == []


def test_create_issue_skips_attach_when_no_default_project_set():
    """No default_project_id → no auto-attach (silently). Operator
    explicitly passing ``project_id=`` would override."""

    stub = _StubClient()
    stub.responses[("POST", "/repos/acme/proj/issues")] = {
        "number": 7, "node_id": "I_kwDO123",
        "title": "t", "state": "open",
        "user": {"login": "a"}, "labels": [], "assignees": [],
    }
    with _with_context():
        cap = _make_cap_with_default_project(
            client=stub, default_project_id=None,
        )
        result = _run(cap.create_issue(title="t", body="b"))
    assert result["ok"] is True
    assert "project_item_id" not in result
    graphql_calls = [c for c in stub.calls if c[0] == "GRAPHQL"]
    assert graphql_calls == []


def test_create_issue_attach_failure_is_non_fatal():
    """The issue is already created when the GraphQL mutation runs;
    a mutation failure must NOT roll back the issue. Surface the
    error as ``project_attach_error`` in the otherwise-ok response."""

    stub = _StubClient()
    stub.responses[("POST", "/repos/acme/proj/issues")] = {
        "number": 7, "node_id": "I_kwDO123",
        "title": "t", "state": "open",
        "user": {"login": "a"}, "labels": [], "assignees": [],
    }
    mutation_key = (
        "GRAPHQL", "mutation($projectId: ID!, $contentId: ID!"[:40],
    )
    stub.raises[mutation_key] = RuntimeError("project not found")
    with _with_context():
        cap = _make_cap_with_default_project(client=stub)
        result = _run(cap.create_issue(title="t", body="b"))
    assert result["ok"] is True  # issue creation succeeded
    assert result["issue"]["number"] == 7
    assert "project not found" in result["project_attach_error"]


def test_create_issue_explicit_project_id_overrides_default():
    """``project_id=`` passed to create_issue overrides
    ``default_project_id`` for that call."""

    stub = _StubClient()
    stub.responses[("POST", "/repos/acme/proj/issues")] = {
        "number": 7, "node_id": "I_kwDO123",
        "title": "t", "state": "open",
        "user": {"login": "a"}, "labels": [], "assignees": [],
    }
    mutation_key = (
        "GRAPHQL", "mutation($projectId: ID!, $contentId: ID!"[:40],
    )
    stub.responses[mutation_key] = {
        "addProjectV2ItemById": {"item": {"id": "PVTI_OVERRIDE"}},
    }
    with _with_context():
        cap = _make_cap_with_default_project(client=stub)
        result = _run(cap.create_issue(
            title="t", body="b", project_id="PVT_OTHER",
        ))
    graphql_calls = [c for c in stub.calls if c[0] == "GRAPHQL"]
    _, _, _kw, variables = graphql_calls[0]
    assert variables["projectId"] == "PVT_OTHER"
    assert result["project_id"] == "PVT_OTHER"


def test_create_issue_no_node_id_records_attach_error():
    """If GitHub's REST response unexpectedly lacks ``node_id``,
    surface a clear error rather than silently skipping the attach."""

    stub = _StubClient()
    stub.responses[("POST", "/repos/acme/proj/issues")] = {
        "number": 7,  # no node_id!
        "title": "t", "state": "open",
        "user": {"login": "a"}, "labels": [], "assignees": [],
    }
    with _with_context():
        cap = _make_cap_with_default_project(client=stub)
        result = _run(cap.create_issue(title="t", body="b"))
    assert result["ok"] is True
    assert "node_id" in result["project_attach_error"]


def test_add_issue_to_project_resolves_node_id_then_calls_mutation():
    """Two-step: GET issue → mutation. Returns the new item id +
    echoes the (repo, number, project_id) for the planner."""

    stub = _StubClient()
    stub.responses[("GET", "/repos/acme/proj/issues/42")] = {
        "number": 42, "node_id": "I_kwDO_42",
        "title": "x", "state": "open",
        "user": {"login": "a"}, "labels": [], "assignees": [],
    }
    mutation_key = (
        "GRAPHQL", "mutation($projectId: ID!, $contentId: ID!"[:40],
    )
    stub.responses[mutation_key] = {
        "addProjectV2ItemById": {"item": {"id": "PVTI_42"}},
    }
    with _with_context():
        cap = _make_cap_with_default_project(client=stub)
        result = _run(cap.add_issue_to_project(issue_number=42))
    assert result["ok"] is True
    assert result["item_id"] == "PVTI_42"
    assert result["number"] == 42
    assert result["repo"] == "acme/proj"
    assert result["project_id"] == "PVT_123"

    # GraphQL got the right node_id from the REST GET.
    graphql_calls = [c for c in stub.calls if c[0] == "GRAPHQL"]
    _, _, _kw, variables = graphql_calls[0]
    assert variables == {"projectId": "PVT_123", "contentId": "I_kwDO_42"}


def test_add_issue_to_project_errors_when_no_project_id():
    """No project_id arg + no default → clear error, no API calls."""

    stub = _StubClient()
    with _with_context():
        cap = _make_cap_with_default_project(
            client=stub, default_project_id=None,
        )
        result = _run(cap.add_issue_to_project(issue_number=42))
    assert result["ok"] is False
    assert "project_id" in result["message"]
    # No REST GET, no GraphQL.
    assert stub.calls == []


def test_add_issue_to_project_errors_when_node_id_missing_on_issue():
    """REST returns issue without node_id → clear error before the
    mutation fires."""

    stub = _StubClient()
    stub.responses[("GET", "/repos/acme/proj/issues/42")] = {
        "number": 42,  # no node_id
        "title": "x", "state": "open",
        "user": {"login": "a"}, "labels": [], "assignees": [],
    }
    with _with_context():
        cap = _make_cap_with_default_project(client=stub)
        result = _run(cap.add_issue_to_project(issue_number=42))
    assert result["ok"] is False
    assert "node_id" in result["message"]
    graphql_calls = [c for c in stub.calls if c[0] == "GRAPHQL"]
    assert graphql_calls == []


# ---------------------------------------------------------------------------
# P4: comment_as_session_agent + signature/footer helpers
# ---------------------------------------------------------------------------


def test_render_session_agent_signature_chat_with_user():
    from polymathera.colony.agents.patterns.capabilities.github import (
        _render_session_agent_signature,
    )

    sig = _render_session_agent_signature(
        agent_role="SessionAgent",
        trigger="chat",
        session_id="sess_4f2a91",
        replying_to="amam-nassar",
    )
    assert "Colony" in sig
    assert "SessionAgent" in sig
    assert "@amam-nassar" in sig
    assert "sess_4f2a91" in sig
    assert sig.startswith("<sub>")
    assert sig.endswith("</sub>")


def test_render_session_agent_signature_mention_includes_src_comment():
    from polymathera.colony.agents.patterns.capabilities.github import (
        _render_session_agent_signature,
    )

    sig = _render_session_agent_signature(
        agent_role="SessionAgent",
        trigger="mention",
        session_id="sess_X",
        replying_to="alice",
        src_comment_id=12345,
    )
    assert "@alice" in sig
    assert "mention" in sig
    assert "12345" in sig


def test_render_session_agent_signature_scheduled_mission():
    from polymathera.colony.agents.patterns.capabilities.github import (
        _render_session_agent_signature,
    )

    sig = _render_session_agent_signature(
        agent_role="SessionAgent",
        trigger="scheduled",
        session_id="sess_X",
        scheduled_mission_name="bottleneck-sweep",
    )
    assert "scheduled mission" in sig
    assert "bottleneck-sweep" in sig
    assert "@" not in sig  # no replying_to in scheduled trigger


def test_render_session_agent_signature_automated_fallback():
    from polymathera.colony.agents.patterns.capabilities.github import (
        _render_session_agent_signature,
    )

    sig = _render_session_agent_signature(
        agent_role="SessionAgent", trigger="automated",
        session_id="sess_X",
    )
    assert "automated" in sig
    assert "sess_X" in sig


def test_render_attribution_footer_is_html_comment_with_parseable_kv():
    from polymathera.colony.agents.patterns.capabilities.github import (
        _render_attribution_footer,
    )

    footer = _render_attribution_footer(
        author="session-agent",
        session_id="sess_4f2a91",
        run_id="run_8e1c33",
        user="amam-nassar",
        trigger="mention",
        src_comment_id=12345,
    )
    assert footer.startswith("<!-- colony:")
    assert footer.endswith(" -->")
    # Each slot grep-able.
    for needle in (
        "author=session-agent",
        "session=sess_4f2a91",
        "run=run_8e1c33",
        "user=amam-nassar",
        "trigger=mention",
        "src_comment=12345",
    ):
        assert needle in footer


def test_render_attribution_footer_omits_unset_slots():
    """Optional fields (run_id, user, src_comment, scheduled_mission)
    are omitted from the footer when None — keeps the payload tight
    and the parser oblivious to absence."""

    from polymathera.colony.agents.patterns.capabilities.github import (
        _render_attribution_footer,
    )

    footer = _render_attribution_footer(
        author="session-agent", session_id="sess_X",
    )
    assert "run=" not in footer
    assert "user=" not in footer
    assert "src_comment=" not in footer
    assert "scheduled_mission=" not in footer
    # trigger defaults to chat and IS present.
    assert "trigger=chat" in footer


def test_comment_as_session_agent_wraps_body_with_signature_and_footer():
    """The wrapper builds full_body = signature + body + footer and
    delegates to comment_on_issue."""

    stub = _StubClient()
    stub.responses[("POST", "/repos/acme/proj/issues/9/comments")] = {
        "id": 123, "html_url": "u",
    }
    with _with_context():
        cap = _make_cap(client=stub)
        result = _run(cap.comment_as_session_agent(
            issue_number=9, body="here is the answer",
            session_id="sess_X", replying_to="alice",
        ))
    assert result["ok"] is True
    assert result["comment_id"] == 123

    # Inspect the body that was actually posted.
    posts = [c for c in stub.calls if c[0] == "POST"]
    assert len(posts) == 1
    body = posts[0][3]["body"]
    # Three blocks separated by blank lines.
    assert body.startswith("<sub>")
    assert "@alice" in body
    assert "sess_X" in body
    assert "here is the answer" in body
    assert "<!-- colony:" in body
    assert "session=sess_X" in body


def test_comment_as_session_agent_scheduled_trigger_uses_mission_name():
    stub = _StubClient()
    stub.responses[("POST", "/repos/acme/proj/issues/9/comments")] = {
        "id": 999, "html_url": "u",
    }
    with _with_context():
        cap = _make_cap(client=stub)
        _run(cap.comment_as_session_agent(
            issue_number=9, body="cron fired",
            session_id="sess_X", trigger="scheduled",
            scheduled_mission_name="weekly-suggestion-pass",
        ))
    body = [c for c in stub.calls if c[0] == "POST"][0][3]["body"]
    assert "scheduled mission" in body
    assert "weekly-suggestion-pass" in body
    assert "trigger=scheduled" in body
    assert "scheduled_mission=weekly-suggestion-pass" in body


def test_comment_as_session_agent_custom_agent_role():
    """``agent_role`` overrides the signature label + footer author
    so non-SessionAgent agents can adopt the same convention."""

    stub = _StubClient()
    stub.responses[("POST", "/repos/acme/proj/issues/9/comments")] = {
        "id": 1, "html_url": "u",
    }
    with _with_context():
        cap = _make_cap(client=stub)
        _run(cap.comment_as_session_agent(
            issue_number=9, body="x",
            session_id="sess_X",
            agent_role="DesignProcessCoordinator",
        ))
    body = [c for c in stub.calls if c[0] == "POST"][0][3]["body"]
    assert "DesignProcessCoordinator" in body
    assert "author=designprocesscoordinator" in body


# ---------------------------------------------------------------------------
# P5a: list_milestones
# ---------------------------------------------------------------------------


def test_list_milestones_flattens_to_planner_friendly_shape():
    """The action returns the API-side open/closed counts directly
    (no per-milestone issue scan) plus html_url + due_on for the
    Colony Status panel."""

    stub = _StubClient()
    stub.responses[("GET", "/repos/acme/proj/milestones")] = [
        {
            "number": 1, "title": "M1", "description": "first",
            "state": "open", "due_on": "2026-06-01T00:00:00Z",
            "open_issues": 7, "closed_issues": 3,
            "html_url": "https://github.com/acme/proj/milestone/1",
        },
        {
            "number": 2, "title": "M2", "description": None,
            "state": "open", "due_on": None,
            "open_issues": 0, "closed_issues": 0,
            "html_url": "u2",
        },
    ]
    with _with_context():
        cap = _make_cap(client=stub)
        result = _run(cap.list_milestones())
    assert result["ok"] is True
    assert result["count"] == 2
    assert result["milestones"][0] == {
        "number": 1, "title": "M1", "description": "first",
        "state": "open", "due_on": "2026-06-01T00:00:00Z",
        "open_issues": 7, "closed_issues": 3,
        "html_url": "https://github.com/acme/proj/milestone/1",
    }
    # The GET hit /milestones with state+sort+direction params.
    method, path, kw, _ = stub.calls[0]
    assert method == "GET"
    assert path == "/repos/acme/proj/milestones"
    assert kw["params"]["state"] == "open"
    assert kw["params"]["sort"] == "due_on"


def test_list_milestones_state_filter_passes_through():
    stub = _StubClient()
    stub.responses[("GET", "/repos/acme/proj/milestones")] = []
    with _with_context():
        cap = _make_cap(client=stub)
        _run(cap.list_milestones(state="closed"))
    assert stub.calls[0][2]["params"]["state"] == "closed"


def test_list_milestones_returns_empty_when_no_default_repo():
    stub = _StubClient()
    agent = MagicMock()
    agent.agent_id = "agent-A"
    with _with_context():
        cap = GitHubCapability(
            agent=agent, scope=BlackboardScope.SESSION, client=stub,
            # No default_repo set, no explicit repo arg.
        )
        result = _run(cap.list_milestones())
    assert result["ok"] is False
    assert result["milestones"] == []
    assert "default_repo" in result["message"]
    assert stub.calls == []  # no API call


# ---------------------------------------------------------------------------
# P5d: whoami + assign_issue
# ---------------------------------------------------------------------------


def test_whoami_returns_app_bot_identity_from_get_user():
    """``whoami`` calls ``GET /user`` once against the per-tenant
    installation token. For App-installation auth GitHub returns the
    App's bot user record — login = ``<slug>[bot]``, type = ``Bot``.
    """

    stub = _StubClient()
    stub.responses[("GET", "/user")] = {
        "login": "colony-bot[bot]",
        "id": 1_000_001,
        "type": "Bot",
        "html_url": "https://github.com/apps/colony-bot",
        "name": "Colony",
        "email": None,
    }
    with _with_context():
        cap = _make_cap(client=stub)
        result = _run(cap.whoami())
    assert result["ok"] is True
    assert result["login"] == "colony-bot[bot]"
    assert result["id"] == 1_000_001
    assert result["type"] == "Bot"
    assert result["html_url"] == "https://github.com/apps/colony-bot"


def test_whoami_caches_after_first_call():
    """The App-bot identity is fixed for the installation's lifetime,
    so ``whoami`` caches the round-trip and doesn't re-hit GitHub on
    repeat calls."""

    stub = _StubClient()
    stub.responses[("GET", "/user")] = {
        "login": "colony-bot[bot]", "id": 1, "type": "Bot",
    }
    with _with_context():
        cap = _make_cap(client=stub)
        first = _run(cap.whoami())
        second = _run(cap.whoami())
    assert first["login"] == "colony-bot[bot]"
    assert second is first  # exact same cached dict instance
    # Only one round-trip even though we called twice.
    get_calls = [c for c in stub.calls if c[0] == "GET" and c[1] == "/user"]
    assert len(get_calls) == 1


def test_whoami_surfaces_http_errors_via_shape_error():
    """When ``GET /user`` returns a 404 (e.g. the installation got
    revoked), the action returns a structured error instead of
    raising — mirrors every other action's error handling."""

    stub = _StubClient()
    stub.raises[("GET", "/user")] = NotFoundError(
        "/user not found", status_code=404,
    )
    with _with_context():
        cap = _make_cap(client=stub)
        result = _run(cap.whoami())
    assert result["ok"] is False
    assert result["status_code"] == 404


def test_whoami_surfaces_no_client_error():
    """No client configured (init error) → clean error result, no
    network attempt."""

    agent = MagicMock()
    agent.agent_id = "agent-A"
    with _with_context():
        cap = GitHubCapability(
            agent=agent, scope=BlackboardScope.SESSION,
            default_repo="acme/proj",
        )
        # Simulate the init-error path without actually wiring envs.
        cap._client = None
        cap._init_error = "credentials missing"
        cap._client_initialized = True
        result = _run(cap.whoami())
    assert result["ok"] is False
    assert "credentials missing" in result["message"]


def test_assign_issue_replace_mode_patches_with_full_list():
    """``replace=True`` (default) sets the issue's assignees to
    exactly the passed list via PATCH /issues/{n}."""

    stub = _StubClient()
    stub.responses[("PATCH", "/repos/acme/proj/issues/7")] = {
        "number": 7, "title": "t", "state": "open",
        "user": {"login": "a"}, "labels": [],
        "assignees": [{"login": "colony-bot[bot]"}],
    }
    with _with_context():
        cap = _make_cap(client=stub)
        result = _run(cap.assign_issue(7, ["colony-bot[bot]"]))
    assert result["ok"] is True
    assert result["issue"]["assignees"] == ["colony-bot[bot]"]
    method, path, _, body = stub.calls[0]
    assert method == "PATCH"
    assert path == "/repos/acme/proj/issues/7"
    assert body == {"assignees": ["colony-bot[bot]"]}


def test_assign_issue_replace_with_empty_list_unassigns_everyone():
    """``replace=True`` + ``assignees=[]`` clears the issue."""

    stub = _StubClient()
    stub.responses[("PATCH", "/repos/acme/proj/issues/7")] = {
        "number": 7, "title": "t", "state": "open",
        "user": {"login": "a"}, "labels": [], "assignees": [],
    }
    with _with_context():
        cap = _make_cap(client=stub)
        result = _run(cap.assign_issue(7, []))
    assert result["ok"] is True
    assert result["issue"]["assignees"] == []
    _, _, _, body = stub.calls[0]
    assert body == {"assignees": []}


def test_assign_issue_additive_mode_uses_post_endpoint():
    """``replace=False`` uses POST /issues/{n}/assignees which is
    additive — GitHub keeps existing assignees + adds the new ones."""

    stub = _StubClient()
    stub.responses[("POST", "/repos/acme/proj/issues/7/assignees")] = {
        "number": 7, "title": "t", "state": "open",
        "user": {"login": "a"}, "labels": [],
        "assignees": [
            {"login": "human"}, {"login": "colony-bot[bot]"},
        ],
    }
    with _with_context():
        cap = _make_cap(client=stub)
        result = _run(cap.assign_issue(
            7, ["colony-bot[bot]"], replace=False,
        ))
    assert result["ok"] is True
    method, path, _, body = stub.calls[0]
    assert method == "POST"
    assert path == "/repos/acme/proj/issues/7/assignees"
    assert body == {"assignees": ["colony-bot[bot]"]}


def test_assign_issue_strips_empty_assignees():
    """Empty / non-string entries are filtered before the API call so
    callers can pass ``[None, ""]`` without breaking GitHub."""

    stub = _StubClient()
    stub.responses[("PATCH", "/repos/acme/proj/issues/7")] = {
        "number": 7, "title": "t", "state": "open",
        "user": {"login": "a"}, "labels": [],
        "assignees": [{"login": "real"}],
    }
    with _with_context():
        cap = _make_cap(client=stub)
        result = _run(cap.assign_issue(
            7, ["real", "", None, "  "],  # type: ignore[list-item]
        ))
    assert result["ok"] is True
    _, _, _, body = stub.calls[0]
    assert body == {"assignees": ["real"]}


def test_assign_issue_returns_error_when_no_default_repo():
    stub = _StubClient()
    agent = MagicMock()
    agent.agent_id = "agent-A"
    with _with_context():
        cap = GitHubCapability(
            agent=agent, scope=BlackboardScope.SESSION, client=stub,
        )
        result = _run(cap.assign_issue(7, ["x"]))
    assert result["ok"] is False
    assert "default_repo" in result["message"]
    assert stub.calls == []


def test_assign_issue_surfaces_client_errors():
    from polymathera.colony.agents.patterns.capabilities._github.client import (
        NotFoundError,
    )

    stub = _StubClient()
    stub.raises[("PATCH", "/repos/acme/proj/issues/9")] = NotFoundError(
        "issue 9 missing", status_code=404,
    )
    with _with_context():
        cap = _make_cap(client=stub)
        result = _run(cap.assign_issue(9, ["x"]))
    assert result["ok"] is False
    assert result["status_code"] == 404


def test_assign_issue_added_to_registered_executors():
    """Catch regression where the action is silently dropped from
    the dispatcher surface."""

    import inspect
    keys = {
        m._action_key for _, m in inspect.getmembers(
            GitHubCapability, predicate=inspect.isfunction,
        ) if getattr(m, "_action_key", None)
    }
    assert "assign_issue" in keys
    assert "whoami" in keys


# ---------------------------------------------------------------------------
# P5 (github_identity_fix_plan): installation_id resolves from
# agent.metadata.parameters["github_identity"]["tenant_installation_id"]
# ---------------------------------------------------------------------------


def _patched_github_auth_config(
    monkeypatch: pytest.MonkeyPatch,
    *,
    app_id: str = "123",
    private_key_pem: str = "",
) -> None:
    """Patch ``get_github_auth_config`` so ``_build_live_client`` sees
    the requested deploy-wide App credentials without touching the
    real ConfigurationManager. Use the RSA-key fixture for
    ``private_key_pem`` when you want a valid PEM."""

    from polymathera.colony.agents.patterns.capabilities import (
        github as gh_mod,
    )

    class _Stub:
        def __init__(self, app_id: str, private_key_pem: str):
            self.app_id = app_id
            self.private_key_pem = private_key_pem

    async def _fake_get():
        return _Stub(app_id=app_id, private_key_pem=private_key_pem)

    # The capability module imports get_github_auth_config lazily inside
    # _build_live_client (``from ...configs import get_github_auth_config``)
    # so we patch the source module that re-exports it.
    from polymathera.colony.agents import configs as configs_mod
    monkeypatch.setattr(
        configs_mod, "get_github_auth_config", _fake_get,
    )
    # And the symbol on the capability module if it was bound at
    # import time (defensive — current code imports per-call).
    if hasattr(gh_mod, "get_github_auth_config"):
        monkeypatch.setattr(
            gh_mod, "get_github_auth_config", _fake_get,
        )


def test_build_live_client_reads_installation_id_from_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The per-tenant installation id rides on agent metadata, not env."""

    priv, _ = _make_rsa_key_pem()
    _patched_github_auth_config(monkeypatch, private_key_pem=priv)

    agent = MagicMock()
    agent.agent_id = "agent-A"
    agent.metadata.parameters = {
        "github_identity": {"tenant_installation_id": "777"},
    }

    with _with_context():
        cap = GitHubCapability(
            agent=agent, scope=BlackboardScope.SESSION,
            default_repo="acme/proj",
        )
        client = _run(cap._build_live_client())
    # TokenCache stores the installation_id as a string on a private
    # field; assert via that since _build_live_client returns the
    # client whose tokens carry it.
    assert client._tokens._installation_id == "777"


def test_build_live_client_kwarg_overrides_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit kwarg wins over metadata — preserves the test-injection
    path."""

    priv, _ = _make_rsa_key_pem()
    _patched_github_auth_config(monkeypatch, private_key_pem=priv)

    agent = MagicMock()
    agent.agent_id = "agent-A"
    agent.metadata.parameters = {
        "github_identity": {"tenant_installation_id": "from-metadata"},
    }

    with _with_context():
        cap = GitHubCapability(
            agent=agent, scope=BlackboardScope.SESSION,
            default_repo="acme/proj",
            installation_id="from-kwarg",
        )
        client = _run(cap._build_live_client())
    assert client._tokens._installation_id == "from-kwarg"


def test_build_live_client_errors_when_no_installation_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing per-tenant installation id → clean RuntimeError naming
    the Tenant GitHub Installation panel."""

    priv, _ = _make_rsa_key_pem()
    _patched_github_auth_config(monkeypatch, private_key_pem=priv)

    agent = MagicMock()
    agent.agent_id = "agent-A"
    agent.metadata.parameters = {}  # no github_identity at all

    with _with_context():
        cap = GitHubCapability(
            agent=agent, scope=BlackboardScope.SESSION,
            default_repo="acme/proj",
        )
        with pytest.raises(RuntimeError, match="Tenant GitHub Installation"):
            _run(cap._build_live_client())


def test_build_live_client_errors_when_no_app_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing deploy-level App ID / private key → clean RuntimeError
    naming the two env vars (so the operator knows where to look)."""

    _patched_github_auth_config(monkeypatch, app_id="", private_key_pem="")

    agent = MagicMock()
    agent.agent_id = "agent-A"
    agent.metadata.parameters = {
        "github_identity": {"tenant_installation_id": "777"},
    }

    with _with_context():
        cap = GitHubCapability(
            agent=agent, scope=BlackboardScope.SESSION,
            default_repo="acme/proj",
        )
        with pytest.raises(
            RuntimeError, match="GITHUB_APP_ID.*GITHUB_PRIVATE_KEY_PEM",
        ):
            _run(cap._build_live_client())


def test_build_live_client_tolerates_missing_github_identity_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metadata without ``github_identity`` shouldn't crash — the
    capability errors with the missing-installation_id message, not an
    AttributeError on ``None.get``."""

    priv, _ = _make_rsa_key_pem()
    _patched_github_auth_config(monkeypatch, private_key_pem=priv)

    agent = MagicMock()
    agent.agent_id = "agent-A"
    # ``parameters`` exists but is missing the github_identity key.
    agent.metadata.parameters = {"design_monorepo_url": "https://x"}

    with _with_context():
        cap = GitHubCapability(
            agent=agent, scope=BlackboardScope.SESSION,
            default_repo="acme/proj",
        )
        with pytest.raises(RuntimeError, match="installation"):
            _run(cap._build_live_client())
