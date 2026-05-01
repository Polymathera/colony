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
        "comment_on_issue", "close_issue", "reopen_issue",
        "add_labels",
        # PRs
        "list_pull_requests", "get_pull_request",
        "get_pr_diff", "create_pull_request",
        "comment_on_pr", "review_pr", "get_pr_checks",
        # projects + coordination
        "list_project_items",
        "claim_unassigned_issue", "release_claim",
    }
    assert keys == expected


def test_bind_round_trips_through_cloudpickle():
    import cloudpickle
    bp = GitHubCapability.bind(scope=BlackboardScope.SESSION)
    bp2 = cloudpickle.loads(cloudpickle.dumps(bp))
    assert bp2.cls is GitHubCapability
