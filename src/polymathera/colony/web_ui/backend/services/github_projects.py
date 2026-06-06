"""Dashboard-side GitHub Projects v2 discovery.

Shared between the two routes that need to enumerate Projects on a
given repo:

- ``GET /tenants/me/discoverable-projects?repo=owner/name`` — used by
  the "+ New Colony" form to gate the create button on at least one
  project existing on the picked repo. Pre-colony, so no colony_id.
- ``GET /colonies/{id}/discoverable-projects`` — used by the per-
  colony settings UI to populate the "change project" picker for an
  already-bound monorepo. Reads the monorepo URL off the colony row.

Both share the same hot path: resolve installation token → run
GraphQL → shape the response. Centralising it here keeps the route
modules thin and the credential plumbing single-sourced.
"""

from __future__ import annotations

import logging
from typing import Any

from ..auth import service as auth_service

logger = logging.getLogger(__name__)


class _ProjectsLookupFailed(Exception):
    """Provider-side or config failure surfaced verbatim to the UI."""


async def list_open_projects_for_repo(
    db_pool: Any,
    *,
    tenant_id: str,
    owner: str,
    name: str,
    max_results: int = 25,
) -> list[dict[str, Any]]:
    """Return the open Projects v2 boards attached to ``owner/name``.

    Each entry: ``{node_id, title, number, url}``. Closed projects
    are filtered server-side via the GraphQL ``query: "is:open"``
    argument and client-side as a belt-and-braces against indexer
    lag.

    Raises :class:`_ProjectsLookupFailed` for any config / network /
    GraphQL error — callers surface ``str(exc)`` verbatim in the
    ``error`` field of the response model.
    """

    tenant_install = await auth_service.get_tenant_github_installation(
        db_pool, tenant_id=tenant_id,
    )
    if (
        tenant_install is None
        or not tenant_install.get("installation_id")
    ):
        raise _ProjectsLookupFailed(
            "Tenant has no GitHub App installation configured. "
            "Set the installation id on the Tenant GitHub "
            "Installation panel, then retry.",
        )
    installation_id = tenant_install["installation_id"]

    # Lazy import — avoids loading the GitHub client stack on every
    # dashboard request that doesn't touch GitHub.
    from polymathera.colony.agents.patterns.capabilities._github.factory import (
        build_github_client_for_installation,
    )
    from polymathera.colony.agents.patterns.capabilities.github import (
        GitHubCapability,
    )

    try:
        client, httpx_client = await build_github_client_for_installation(
            installation_id=installation_id,
            capability_name="ColonyDashboard",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "list_open_projects_for_repo: failed to mint installation "
            "token for tenant=%s installation=%s",
            tenant_id, installation_id,
        )
        raise _ProjectsLookupFailed(str(exc)) from exc

    try:
        try:
            data = await client.graphql(
                GitHubCapability._LIST_REPO_PROJECTS,
                variables={
                    "owner": owner, "name": name,
                    "first": max_results,
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "list_open_projects_for_repo: GraphQL call failed "
                "for repo=%s/%s", owner, name,
            )
            raise _ProjectsLookupFailed(str(exc)) from exc
    finally:
        await client.close()
        await httpx_client.aclose()

    repo_node = (data or {}).get("repository") or {}
    nodes = ((repo_node.get("projectsV2") or {}).get("nodes") or [])
    return [
        {
            "node_id": n["id"],
            "title": n.get("title") or "",
            "number": n.get("number"),
            "url": n.get("url"),
        }
        for n in nodes
        if n and n.get("id") and not n.get("closed")
    ]


__all__ = (
    "list_open_projects_for_repo",
    "_ProjectsLookupFailed",
)
