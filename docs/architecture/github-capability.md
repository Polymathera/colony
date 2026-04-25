# `GitHubCapability`

Lets agents read and write GitHub issues, pull requests, and project boards via a GitHub App installation. The coordination primitive `claim_unassigned_issue` turns a Project board into an external work-queue that survives Colony restarts and interleaves cleanly with human contributors.

Code: `polymathera.colony.agents.patterns.capabilities.GitHubCapability`. Subpackage: `_github/{auth,client}.py`. Event protocol: `polymathera.colony.agents.blackboard.protocol.GitHubEventProtocol`.

## When to use

Many of the work products Colony agents produce — bug analyses, contract specifications, slicing reports, change-impact summaries — naturally land as GitHub issues, comments, or PR reviews. A coordinator that picks up unassigned issues from a Project board, runs an analysis, posts a comment with the result, and moves the issue to "Done" needs exactly this surface.

## Action surface (22 actions)

| Group | Actions |
|-------|---------|
| Repo + content | `list_repos`, `get_repo`, `list_branches`, `get_file_contents`, `search_code` |
| Issues | `list_issues`, `get_issue`, `create_issue`, `comment_on_issue`, `close_issue`, `reopen_issue`, `add_labels` |
| Pull requests | `list_pull_requests`, `get_pull_request`, `get_pr_diff`, `create_pull_request`, `comment_on_pr`, `review_pr`, `get_pr_checks` |
| Projects v2 | `list_project_items` |
| Coordination | `claim_unassigned_issue`, `release_claim` |

Every action returns a uniform `{"ok": bool, "message": str, …}` shape. Errors degrade rather than raise: `NotFoundError` → `{ok: false, status_code: 404}`, `RateLimitError` → `{ok: false, status_code: 403}`, etc. Mutations write an audit record to `audit:github:{ts}:{uuid}`.

## Authentication

GitHub Apps, not personal access tokens. The capability needs three pieces of configuration:

| What | Where it comes from |
|------|---------------------|
| App ID | Constructor `app_id=` or `GITHUB_APP_ID` env var |
| Private key (PEM) | Constructor `private_key_pem=`, file at `private_key_path=`, or `GITHUB_PRIVATE_KEY_PEM` env var |
| Installation ID | Constructor `installation_id=` or `GITHUB_INSTALLATION_ID` env var |

The flow:

1. `GitHubAppAuth.mint_jwt` signs a 9-minute JWT with the RSA key, backdated 60 s to tolerate clock skew.
2. `TokenCache.get` exchanges the JWT for an installation-scoped access token (1 h TTL) at `/app/installations/{id}/access_tokens`.
3. The token is cached and refreshed 5 min before expiry under a lock.

Missing credentials are *non-fatal*: the capability captures an `_init_error` and every action returns a clean error dict. Operators turn it on by setting env vars (or — eventually — through the Settings UI).

## Client behaviour

`GitHubClient` wraps `httpx.AsyncClient` with:

- **Auto token injection** on every request.
- **One 401-retry** with forced token refresh (handles a stale-cache window).
- **Primary rate-limit detection** (`x-ratelimit-remaining: 0`) → `RateLimitError` raised immediately.
- **Secondary rate-limit detection** (403/429 + `Retry-After` or "secondary"/"abuse" in body) → exponential backoff with jitter, honouring `Retry-After`, up to `max_retries`.
- **Pagination iterator** (`iter_paginated`) that stops when a page comes back short.
- **GraphQL helper** for Projects v2 (the only API GitHub no longer offers via REST).

## `claim_unassigned_issue` — the coordination primitive

GitHub Projects v2 doesn't support optimistic concurrency on field updates, so we lock with **labels** instead:

1. List candidate open issues, filtering out anything already carrying a `claimed-by:*` label.
2. For each candidate (in order):
   1. POST `claimed-by:<agent_id>` to the issue's labels.
   2. Re-fetch the issue.
   3. If our label is the *only* `claimed-by:*` label → claim succeeded; return the issue.
   4. Otherwise we lost a race — DELETE our label and try the next candidate.
3. If no candidate succeeds, return `{"claimed": false, "issue": None}`.

`release_claim(issue_number)` removes the label when work completes.

This pattern races safely against any number of concurrent agents and against humans who manually label issues. The audit log records every claim.

## Event protocol

`GitHubEventProtocol` defines the blackboard key shapes a future `POST /api/v1/github/webhook` endpoint will write:

```
github:issue_opened:{owner}/{repo}:{number}
github:issue_commented:{owner}/{repo}:{number}
github:issue_closed:{owner}/{repo}:{number}
github:pr_opened:{owner}/{repo}:{number}
github:pr_review_requested:{owner}/{repo}:{number}
github:pr_merged:{owner}/{repo}:{number}
github:project_item_changed:{project_id}:{item_id}
```

Other capabilities can subscribe today via `input_patterns=[GitHubEventProtocol.issue_opened_pattern(), …]`. The webhook endpoint itself is deferred — see the design doc for the FastAPI route plan.

## Configuration

```python
GitHubCapability.bind(
    scope=BlackboardScope.SESSION,
    default_repo="acme/proj",                  # used when actions accept repo=None
    default_project_id="PVT_kwDO…",            # GraphQL node id
    max_requests_per_minute=120,
    audit_enabled=True,
)
```

Wired into the session agent. With no credentials configured the capability stays disabled; the action surface returns clean error dicts so the LLM can suggest the user configure it.

## Test surface

`tests/test_github_capability.py` (26 tests). Every HTTP call goes through `httpx.MockTransport`. Covers:

- JWT claim shape, 60 s backdating.
- Token-cache refresh-before-expiry, exchange-error surfacing.
- Client 401-retry (one shot), primary rate-limit raising, paginated iterator short-page exit.
- Action surface via a `_StubClient`: list-with-PR-exclusion, ownership defaults, audit emission, project field flattening.
- `claim_unassigned_issue`: happy path, skip-already-claimed, **race-rollback** (concurrent agent applied its label first → we delete ours and move on), no-candidates.
- `GitHubEventProtocol` round-trips for repo/number and project/item keys.
- Missing-credentials init-error reporting.

## Open follow-ups

- **Webhook endpoint** at `POST /api/v1/github/webhook` (signature verified, normalised to `GitHubEventProtocol` keys).
- **Settings UI** for App credentials + project field mapping.
- **Per-tenant multi-installation** support.
- **`GitHubReadOnlyCapability`** subclass that omits mutations via `bind(exclude_actions=[...])`.
- **`GitHubActionsCapability`** sibling for `trigger_workflow` / `wait_for_workflow_run`.
