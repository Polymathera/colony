"""Auth + tenancy + licensing schema.

Clean shape — wipe-and-redeploy is the dev norm so there are no
back-compat migrations. Tables and indices are all
``IF NOT EXISTS`` so re-running on the same db is a no-op, but the
schema below is the canonical truth, not a sequence of patches over
some older shape.

Tables:

- ``tenants`` — a VCS organisation/group/workspace that has installed
  the Colony App. The unit of billing + ACL boundary.
- ``users`` — a human, identified by their VCS account. No password
  auth — sign-in is OAuth-only (see ``vcs/`` package + the OAuth
  routes added in PR 3).
- ``user_tenants`` — many-to-many: a user may belong to multiple
  tenants. Replaces the v1 ``users.tenant_id`` 1:1 shortcut.
- ``colonies`` — a workspace pointed at a VCS repo containing
  ``.colony/``. Multiple colonies may point at the same repo
  (per-tenant or cross-tenant) — git is the source of truth for
  design content; no SQL-level uniqueness on ``(tenant_id,
  vcs_repo_id)``.
- ``licenses`` — per-tenant plan + entitlements. Plan defaults +
  source-precedence are in ``auth/license_plans.py`` +
  ``auth/license_service.py``; this module only owns the table.

See ``colony/vcs_native_tenancy_plan.md §3`` + ``§9`` for the design.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tenants — VCS org/group/workspace
# ---------------------------------------------------------------------------

TENANTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS tenants (
    id                       TEXT PRIMARY KEY,
    name                     TEXT NOT NULL,
    -- Which VCS this tenant lives in.
    vcs_provider             TEXT NOT NULL DEFAULT 'github',
    -- Provider's stable identifier for the org/group/workspace.
    -- NULL during a transient window (a tenant row created from a
    -- non-VCS source); populated on first OAuth sign-in.
    vcs_org_id               TEXT,
    -- Display login/handle for the org (e.g. GitHub "polymathera-inc").
    vcs_org_login            TEXT,
    -- GitHub-specific: the App installation id minted when the tenant
    -- admin installs Colony's App. Other providers leave this NULL
    -- and use bot_token_encrypted instead.
    github_installation_id   TEXT,
    -- Provider-agnostic bot credential. GitLab GAT / Bitbucket
    -- workspace token. Encrypted at rest (see PR 6). Today's GitHub
    -- code path doesn't use this — installation_id is enough.
    bot_token_encrypted      TEXT,
    bot_token_expires_at     TIMESTAMPTZ,
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

TENANTS_VCS_ORG_UQ = """
CREATE UNIQUE INDEX IF NOT EXISTS tenants_vcs_org_uq
    ON tenants(vcs_provider, vcs_org_id)
    WHERE vcs_org_id IS NOT NULL;
"""


# ---------------------------------------------------------------------------
# Users — VCS-OAuth identity, no password
# ---------------------------------------------------------------------------

USERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id                   TEXT PRIMARY KEY,
    -- Which provider OAuth'd this user. A human with both GitHub and
    -- GitLab identities is TWO rows (account-linking is a future
    -- feature, out of scope here).
    vcs_provider         TEXT NOT NULL DEFAULT 'github',
    -- Provider's stable numeric id, stored as TEXT so providers with
    -- opaque/UUID identifiers don't need a column-type change.
    vcs_user_id          TEXT,
    -- The user's handle on the provider (e.g. GitHub "anassar").
    vcs_login            TEXT,
    -- Verified primary email (provider-verified). NULL when no
    -- verified email — caller decides whether to refuse sign-in.
    vcs_email            TEXT,
    -- Free-form display name from the provider's profile. Used in
    -- the Co-Authored-By trailer + UI. May be NULL.
    git_user_name        TEXT,
    -- When the OAuth identity was first persisted.
    vcs_connected_at     TIMESTAMPTZ,
    -- Refreshed on every sign-in so the UI can show "last verified".
    vcs_last_verified_at TIMESTAMPTZ,
    -- Per-user "default" colony — replaces the v1 ``colonies.is_default``
    -- per-tenant flag. Set on first colony discovery, updated on each
    -- explicit switch in the UI. SET NULL on colony delete so a
    -- dangling pointer can't crash the session-create handler.
    active_colony_id     TEXT REFERENCES colonies(id) ON DELETE SET NULL,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

USERS_VCS_IDENTITY_UQ = """
CREATE UNIQUE INDEX IF NOT EXISTS users_vcs_identity_uq
    ON users(vcs_provider, vcs_user_id)
    WHERE vcs_user_id IS NOT NULL;
"""


# ---------------------------------------------------------------------------
# user_tenants — many-to-many membership
# ---------------------------------------------------------------------------

USER_TENANTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS user_tenants (
    user_id          TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    tenant_id        TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    -- Coarse role derived from the user's VCS-side org permission.
    -- Refined per-colony by repo permission tier in PR 4+.
    role             TEXT NOT NULL CHECK (role IN ('member', 'admin')),
    joined_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- Refreshed on every sign-in walker pass. Stale rows (user no
    -- longer in the VCS org) get pruned by a follow-up GC PR.
    last_verified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, tenant_id)
);
"""


# ---------------------------------------------------------------------------
# Colonies — pointed at a VCS repo with .colony/
# ---------------------------------------------------------------------------

COLONIES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS colonies (
    id                      TEXT PRIMARY KEY,
    name                    TEXT NOT NULL,
    tenant_id               TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    description             TEXT NOT NULL DEFAULT '',
    -- VCS-side repo identity. NULL during a transient window when a
    -- colony is created before its repo is selected.
    vcs_repo_id             TEXT,
    vcs_repo_full_name      TEXT,
    default_branch          TEXT,
    -- Design-monorepo URL/branch/commit — derivable from the repo
    -- fields above once PR 4 wires the URL templates per provider,
    -- but kept as explicit columns for now so existing
    -- ``clone_or_retrieve_repository`` callers don't have to know
    -- about ``VcsProvider.url_for_repo`` yet.
    design_monorepo_url     TEXT,
    design_monorepo_branch  TEXT NOT NULL DEFAULT 'main',
    design_monorepo_commit  TEXT NOT NULL DEFAULT 'HEAD',
    -- Per-commit attribution preferences (P1 of the GitHub identity
    -- fix). Per-user identity (``git_user_name`` / ``vcs_email``)
    -- lives on ``users``; this row only carries the policy knobs.
    commit_principal        TEXT NOT NULL DEFAULT 'colony',
    commit_co_author        TEXT DEFAULT 'user',
    -- GitHub Projects v2 attachment. ``github_project_node_id`` is
    -- the GraphQL node id Colony stamps on every newly-created issue
    -- (via ``GitHubCapability.create_issue(project_id=...)``) and
    -- threads as ``default_project_id`` into ``GitHubCapability.bind``
    -- at session-create time. ``github_project_title`` is a UI cache
    -- so the picker can show the human-readable name without an
    -- extra GraphQL round-trip; the operator can re-sync via the
    -- "discover projects" route if the title drifts. Both NULL when
    -- the operator hasn't picked a project yet — session-create
    -- refuses to spawn until one is set, so the colony is unusable
    -- in that state by design.
    github_project_node_id  TEXT,
    github_project_title    TEXT,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

# Non-unique: multiple colonies CAN point at the same repo.
COLONIES_TENANT_REPO_IDX = """
CREATE INDEX IF NOT EXISTS colonies_tenant_repo_idx
    ON colonies(tenant_id, vcs_repo_id)
    WHERE vcs_repo_id IS NOT NULL;
"""

COLONIES_TENANT_IDX = """
CREATE INDEX IF NOT EXISTS colonies_tenant_idx
    ON colonies(tenant_id);
"""


# ---------------------------------------------------------------------------
# tenant_repos — cache of repos discovered during VCS sign-in
# ---------------------------------------------------------------------------
#
# Walker (services/colony_discovery.py) upserts one row per repo it
# sees inside each tenant. ``has_colony_marker`` flips to TRUE for
# repos with ``.colony/``. The dashboard's
# ``/api/v1/tenants/me/discoverable-repos`` route reads from here so
# the "New Colony" form + the per-colony "Design monorepo" field can
# render a dropdown without needing the user's OAuth token (which we
# discard right after the callback finishes).

TENANT_REPOS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS tenant_repos (
    tenant_id           TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    vcs_repo_id         TEXT NOT NULL,
    vcs_repo_full_name  TEXT NOT NULL,
    default_branch      TEXT NOT NULL,
    -- read | write | admin — caller-side enforcement, not CHECK
    -- constrained so a future provider tier doesn't require a
    -- migration.
    user_permission     TEXT NOT NULL,
    has_colony_marker   BOOLEAN NOT NULL DEFAULT FALSE,
    last_seen_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (tenant_id, vcs_repo_id)
);
"""


# ---------------------------------------------------------------------------
# Licenses — per-tenant plan + entitlements
# ---------------------------------------------------------------------------

LICENSES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS licenses (
    tenant_id     TEXT PRIMARY KEY REFERENCES tenants(id) ON DELETE CASCADE,
    plan          TEXT NOT NULL CHECK (plan IN
                      ('free','team','business','enterprise','dev')),
    -- Per-tenant overrides merged on top of PLAN_DEFAULTS[plan] at
    -- request time. Plan keys are documented in license_plans.py.
    entitlements  JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- How this row was created. Determines overwrite precedence — see
    -- ``license_service.upsert_license`` for the ladder.
    source        TEXT NOT NULL CHECK (source IN
                      ('default','env_bootstrap','marketplace',
                       'stripe','admin','license_jwt')),
    valid_until   TIMESTAMPTZ,    -- NULL = no expiry
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


# ---------------------------------------------------------------------------
# Schema bootstrap
# ---------------------------------------------------------------------------

# Order matters: ``users`` FK-references ``colonies(id)`` (via
# active_colony_id) and ``colonies`` FK-references ``tenants(id)``.
# ``users`` also doesn't depend on ``colonies`` at table-create time
# because the FK is SET NULL (the column can be NULL while colonies
# doesn't exist yet). But to keep the FK constraint creation simple
# we create ``colonies`` before ``users``. ``user_tenants`` FK-refs
# both; ``licenses`` FK-refs ``tenants``.
_TABLE_ORDER: tuple[str, ...] = (
    TENANTS_TABLE_SQL,
    COLONIES_TABLE_SQL,
    USERS_TABLE_SQL,
    USER_TENANTS_TABLE_SQL,
    TENANT_REPOS_TABLE_SQL,
    LICENSES_TABLE_SQL,
)

_INDEX_ORDER: tuple[str, ...] = (
    TENANTS_VCS_ORG_UQ,
    USERS_VCS_IDENTITY_UQ,
    COLONIES_TENANT_REPO_IDX,
    COLONIES_TENANT_IDX,
)


async def ensure_auth_schema(db_pool) -> None:
    """Create the auth/tenancy/license tables + indices if absent."""
    try:
        async with db_pool.acquire() as conn:
            for stmt in _TABLE_ORDER:
                await conn.execute(stmt)
            for stmt in _INDEX_ORDER:
                await conn.execute(stmt)
        logger.info(
            "Auth schema ensured (tenants + users + user_tenants + "
            "colonies + licenses).",
        )
    except Exception:
        logger.error("Failed to create auth schema", exc_info=True)
        raise
