"""Per-colony persistence of the operator's source-selection state
from the Design Monorepo tab's two checkbox lists.

Two independent selections, both keyed by ``colony_id``:

- ``vcm_sources`` enabled set        — for ``vcm_sources:`` rows
- ``knowledge_sources`` enabled set  — for ``knowledge_sources:`` rows

Storage flows through :class:`PolymatheraApp.get_state_manager` —
the project's standard per-key persistent state, configured per
deployment via ``sys_config.distributed_state.storage`` (Redis in
the local stack, swappable to etcd / etc.). Each call to
:func:`set_enabled_*` runs inside a CAS-protected
``write_transaction``, so two browser tabs toggling the same
colony's checkboxes can't race.

Semantics: an unset key (the default for a fresh colony) yields a
default :class:`SourceSelection` with ``enabled=None`` — "all rows
enabled", same convention as
``materialize_*_sources(enabled_sources=...)``.
"""

from __future__ import annotations

import logging

from ..distributed import get_polymathera
from ..distributed.state_management import SharedState


logger = logging.getLogger(__name__)


class SourceSelection(SharedState):
    """Persistent state model for a single side (VCM or KB) of one
    colony's checkbox list. Subclasses :class:`SharedState` (the
    project's :class:`StateManager`-compatible base — every state
    model needs the ``writable`` flag the transaction loop sets and
    clears around each yield). Default-constructible
    (``enabled=None``) so :class:`StateManager` can synthesize it
    for unset keys."""

    enabled: list[str] | None = None


def _vcm_state_key(colony_id: str) -> str:
    return f"colony:{colony_id}:source_selection:vcm"


def _knowledge_state_key(colony_id: str) -> str:
    return f"colony:{colony_id}:source_selection:knowledge"


async def _read(state_key: str) -> list[str] | None:
    sm = await get_polymathera().get_state_manager(
        state_type=SourceSelection, state_key=state_key,
    )
    enabled: list[str] | None = None
    async for state in sm.read_transaction():
        enabled = state.enabled
    return enabled


async def _write(state_key: str, names: list[str] | None) -> None:
    sm = await get_polymathera().get_state_manager(
        state_type=SourceSelection, state_key=state_key,
    )
    new_value = list(names) if names is not None else None
    # NB: ``write_transaction`` is CAS — the post-yield save must run
    # so the loop body cannot ``return`` / ``break`` after mutation.
    # See MEMORY.md "CRITICAL BUG: write_transaction + return/break".
    async for state in sm.write_transaction():
        state.enabled = new_value


# ---------------------------------------------------------------------------
# Public API — symmetric VCM / knowledge accessors.
# ---------------------------------------------------------------------------


async def list_enabled_vcm_sources(colony_id: str) -> list[str] | None:
    """Return the operator's currently-enabled ``vcm_sources`` row names.

    ``None`` means no filter — every ``vcm_sources:`` row is enabled.
    Same convention as ``materialize_vcm_sources(enabled_sources=...)``.
    """
    return await _read(_vcm_state_key(colony_id))


async def set_enabled_vcm_sources(
    colony_id: str, names: list[str] | None,
) -> None:
    """Persist the operator's ``vcm_sources`` selection. ``None``
    resets to "all enabled" defaults."""
    await _write(_vcm_state_key(colony_id), names)


async def list_enabled_knowledge_sources(colony_id: str) -> list[str] | None:
    """Return the operator's currently-enabled ``knowledge_sources``
    row names. ``None`` means no filter — every ``knowledge_sources:``
    row is enabled."""
    return await _read(_knowledge_state_key(colony_id))


async def set_enabled_knowledge_sources(
    colony_id: str, names: list[str] | None,
) -> None:
    """Persist the operator's ``knowledge_sources`` selection. ``None``
    resets to "all enabled" defaults."""
    await _write(_knowledge_state_key(colony_id), names)


__all__ = (
    "SourceSelection",
    "list_enabled_knowledge_sources",
    "list_enabled_vcm_sources",
    "set_enabled_knowledge_sources",
    "set_enabled_vcm_sources",
)
