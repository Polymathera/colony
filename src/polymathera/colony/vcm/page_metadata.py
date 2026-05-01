"""Typed metadata for ``VirtualContextPage``.

Per master §5.6 item 6, the convergence runtime needs page metadata to
be a first-class subscription surface — promoted out of free-form
``metadata: dict[str, Any]`` into typed columns that the subscription
index can match on without parsing JSON.

This module defines the typed shape ``PageMetadata`` with the four
fields master §5.1 calls out:

- ``data_type`` — open-set string (``"code"`` / ``"requirements"`` /
  ``"design_decision"`` / ``"standard_clause"`` / ``"paper_section"`` /
  ``"telemetry_window"`` / ``"cad_artifact"`` / ``"simulation_result"`` /
  ``"run_log"`` / etc.).
- ``source`` — origin URI (``"git:<remote>:<branch>:<commit>"`` /
  ``"arxiv:<id>:<version>"`` / ``"pubmed:<pmid>"`` /
  ``"book:<isbn>:<chapter>"`` / ``"jama:<doi>"`` /
  ``"semi:<no>:<rev>"`` / ``"blackboard:<scope_id>:<key>"`` / etc.).
- ``effective_at`` — design-time the content represents (distinct from
  ``created_at`` which is when the page was ingested).
- ``access_rights`` — read/write ACL summary.

The fields piggyback on ``VirtualContextPage.metadata: dict[str, Any]``
under reserved keys (``"data_type"``, ``"source"``, ``"effective_at"``,
``"access_rights"``) — see ``read_typed`` / ``write_typed`` below.
The SQLModel migration in ``models.py`` adds typed columns that mirror
these fields for indexed lookups; this module is the in-memory
representation.

Two reserved-key constants make the metadata-key contract explicit:

- ``METADATA_TYPE_KEY = "data_type"``
- ``METADATA_SOURCE_KEY = "source"``
- ``METADATA_EFFECTIVE_AT_KEY = "effective_at"``
- ``METADATA_ACCESS_RIGHTS_KEY = "access_rights"``
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


METADATA_TYPE_KEY = "data_type"
METADATA_SOURCE_KEY = "source"
METADATA_EFFECTIVE_AT_KEY = "effective_at"
METADATA_ACCESS_RIGHTS_KEY = "access_rights"


class AccessRights(BaseModel):
    """Read/write ACL summary on a page.

    The ``VirtualContextPage`` already carries ``allowed_tenant_ids`` /
    ``sensitivity_level`` for tenant-level isolation; ``AccessRights``
    refines this with per-agent and per-capability ACLs.

    All three lists are *additive*: an empty list means "anyone with
    tenant access". A non-empty list restricts to those identifiers.
    """

    model_config = ConfigDict(frozen=True)

    can_read: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Agent / capability ids that can read this page.",
    )
    can_write: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Agent / capability ids that can mutate the page-graph "
        "edges incident to this page (or, for paged blackboard scopes, "
        "the underlying scope).",
    )
    can_subscribe: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Agent / capability ids permitted to subscribe to "
        "page-change events for this page.",
    )


class PageMetadata(BaseModel):
    """Typed view of a page's metadata, used by the subscription engine.

    Treat this as the *typed projection* of ``VirtualContextPage.metadata``;
    everything not represented here continues to live in the free-form
    ``metadata`` dict and is preserved verbatim. The runtime reads / writes
    the typed fields through ``read_typed`` and ``write_typed``.
    """

    model_config = ConfigDict(frozen=True)

    data_type: str | None = None
    source: str | None = None
    effective_at: datetime | None = None
    access_rights: AccessRights = Field(default_factory=AccessRights)


def read_typed(metadata: dict[str, Any] | None) -> PageMetadata:
    """Extract typed metadata from a page's free-form ``metadata`` dict.

    Tolerant of missing keys and of values stored as strings (the SQL
    JSON serialiser stores ``effective_at`` as an ISO-8601 string).
    """

    if not metadata:
        return PageMetadata()

    data_type = metadata.get(METADATA_TYPE_KEY)
    source = metadata.get(METADATA_SOURCE_KEY)

    raw_effective = metadata.get(METADATA_EFFECTIVE_AT_KEY)
    effective_at: datetime | None
    if raw_effective is None:
        effective_at = None
    elif isinstance(raw_effective, datetime):
        effective_at = raw_effective
    elif isinstance(raw_effective, (int, float)):
        effective_at = datetime.fromtimestamp(float(raw_effective), timezone.utc)
    else:
        try:
            effective_at = datetime.fromisoformat(str(raw_effective))
        except ValueError:
            effective_at = None

    raw_rights = metadata.get(METADATA_ACCESS_RIGHTS_KEY)
    access_rights: AccessRights
    if isinstance(raw_rights, AccessRights):
        access_rights = raw_rights
    elif isinstance(raw_rights, dict):
        try:
            access_rights = AccessRights.model_validate(raw_rights)
        except Exception:  # noqa: BLE001 - pydantic ValidationError + edge cases
            access_rights = AccessRights()
    else:
        access_rights = AccessRights()

    return PageMetadata(
        data_type=str(data_type) if data_type else None,
        source=str(source) if source else None,
        effective_at=effective_at,
        access_rights=access_rights,
    )


def write_typed(
    metadata: dict[str, Any] | None, typed: PageMetadata,
) -> dict[str, Any]:
    """Return a new ``metadata`` dict with typed fields written back.

    Non-typed entries are preserved. ``None`` values are stored as ``None``
    (not deleted) so a downstream reader can distinguish "explicitly
    unset" from "never set" if needed; ``read_typed`` treats both as None.
    """

    out: dict[str, Any] = dict(metadata or {})
    out[METADATA_TYPE_KEY] = typed.data_type
    out[METADATA_SOURCE_KEY] = typed.source
    out[METADATA_EFFECTIVE_AT_KEY] = (
        typed.effective_at.isoformat() if typed.effective_at else None
    )
    out[METADATA_ACCESS_RIGHTS_KEY] = typed.access_rights.model_dump()
    return out


__all__ = (
    "AccessRights",
    "PageMetadata",
    "read_typed",
    "write_typed",
    "METADATA_TYPE_KEY",
    "METADATA_SOURCE_KEY",
    "METADATA_EFFECTIVE_AT_KEY",
    "METADATA_ACCESS_RIGHTS_KEY",
)
