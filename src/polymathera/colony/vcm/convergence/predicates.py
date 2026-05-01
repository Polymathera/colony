"""``PageMetadataPredicate`` — typed match expressions over page metadata.

Per master §5.2 mechanism 1, "a capability declares the *exact*
page-graph patterns it subscribes to — by ``data_type``, ``source`` URI
prefix, page-graph reachability, edge-type closure, or any conjunction".

This module defines the typed shape capabilities use to declare those
patterns, plus the matching logic the convergence runtime uses to test
whether a ``PageChangeEvent`` matches a subscription.

The predicate is intentionally narrow: a small set of typed matchers
combined by *implicit AND* (every set field must match). For richer
predicates, callers compose multiple subscriptions.

Field semantics:

- ``data_type`` — exact match against the page's ``data_type``
  (``read_typed(metadata).data_type``). ``None`` means "any".
- ``source_prefix`` — string prefix on the page's ``source`` URI.
  Useful for "any git source" (``"git:"``) or "this specific repo"
  (``"git:git@example/foo.git:"``).
- ``scope_id`` — exact match. ``None`` means "any scope".
- ``effective_at_after`` / ``effective_at_before`` — inclusive bounds
  on the page's ``effective_at`` (UTC). Pages without an
  ``effective_at`` are excluded when either bound is set.
- ``page_id_in`` — explicit allow-list of page ids. Useful for
  page-id-targeted subscriptions.
- ``edge_reach_root`` + ``edge_reach_max_hops`` + ``edge_reach_types`` —
  the predicate matches a page that is reachable in the page graph
  from ``edge_reach_root`` within ``edge_reach_max_hops`` edges, where
  every edge on the walk has a type in ``edge_reach_types`` (empty
  set means any edge type). Reachability is computed by the runtime
  using its snapshot of the page graph (``vcm.page_storage``).

The predicate's ``matches`` method takes a ``PageChangeEvent`` plus an
optional ``edge_reach_resolver`` callback. The callback is responsible
for the page-graph walk; passing it as a callback keeps the predicate
free of any direct VCM dependency, so the same predicate model can be
used against both a live runtime (with the real page graph) and a unit
test (with a fake reach resolver).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from ..page_events import PageChangeEvent


# Callback signature for page-graph reach queries. ``root, page_id,
# max_hops, edge_types`` -> True if ``page_id`` is reachable from
# ``root`` within ``max_hops`` edges where every edge type is in
# ``edge_types`` (or any edge type if ``edge_types`` is empty).
EdgeReachResolver = Callable[[str, str, int, frozenset[str]], bool]


class PageMetadataPredicate(BaseModel):
    """Typed match expression over a page-change event's metadata."""

    model_config = ConfigDict(frozen=True)

    data_type: str | None = None
    source_prefix: str | None = None
    scope_id: str | None = None
    effective_at_after: datetime | None = None
    effective_at_before: datetime | None = None
    page_id_in: tuple[str, ...] = Field(default_factory=tuple)

    # Edge-reachability matchers — all three must be set together (or
    # all unset). When set, ``matches`` requires an ``edge_reach_resolver``.
    edge_reach_root: str | None = None
    edge_reach_max_hops: int = Field(
        default=0,
        ge=0,
        description=(
            "Max page-graph hops from ``edge_reach_root``. 0 disables "
            "the matcher; positive values require ``edge_reach_root`` "
            "to be set."
        ),
    )
    edge_reach_types: frozenset[str] = Field(
        default_factory=frozenset,
        description=(
            "Allowed edge types on the reachability walk. Empty set "
            "means 'any edge type'."
        ),
    )

    @model_validator(mode="after")
    def _validate_edge_matcher(self) -> PageMetadataPredicate:
        if self.edge_reach_max_hops > 0 and not self.edge_reach_root:
            raise ValueError(
                "edge_reach_max_hops > 0 requires edge_reach_root to be set.",
            )
        if self.edge_reach_root and self.edge_reach_max_hops <= 0:
            raise ValueError(
                "edge_reach_root requires edge_reach_max_hops > 0.",
            )
        return self

    # ---- Matching ------------------------------------------------------

    def matches(
        self,
        event: "PageChangeEvent",
        *,
        edge_reach_resolver: EdgeReachResolver | None = None,
    ) -> bool:
        """Test whether ``event`` matches this predicate.

        Implicit AND across set fields. ``edge_reach_resolver`` must be
        supplied if the predicate sets the edge-reachability matcher;
        passing one when the matcher isn't set is a no-op.
        """

        if self.data_type is not None and event.data_type != self.data_type:
            return False
        if self.source_prefix is not None and not event.source.startswith(
            self.source_prefix
        ):
            return False
        if self.scope_id is not None and event.scope_id != self.scope_id:
            return False
        if self.page_id_in and event.page_id not in self.page_id_in:
            return False
        if self.effective_at_after is not None or self.effective_at_before is not None:
            effective = self._extract_effective_at(event)
            if effective is None:
                return False
            if (
                self.effective_at_after is not None
                and effective < self.effective_at_after
            ):
                return False
            if (
                self.effective_at_before is not None
                and effective > self.effective_at_before
            ):
                return False
        if self.edge_reach_max_hops > 0:
            if edge_reach_resolver is None:
                # Be conservative: a predicate that requires a resolver
                # cannot match without one.
                return False
            if not edge_reach_resolver(
                self.edge_reach_root or "",
                event.page_id,
                self.edge_reach_max_hops,
                self.edge_reach_types,
            ):
                return False
        return True

    # ---- Internals -----------------------------------------------------

    @staticmethod
    def _extract_effective_at(event: "PageChangeEvent") -> datetime | None:
        raw: Any = event.extra.get("effective_at")
        if raw is None:
            return None
        if isinstance(raw, datetime):
            return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
        if isinstance(raw, (int, float)):
            return datetime.fromtimestamp(float(raw), timezone.utc)
        try:
            parsed = datetime.fromisoformat(str(raw))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    # ---- Index hints ---------------------------------------------------

    @property
    def is_indexable(self) -> bool:
        """True if this predicate can be served from the
        ``(data_type, source_prefix)`` index without a linear scan.
        Edge-reach predicates and effective-at-window predicates are
        not indexable; the runtime falls back to linear scan for them.
        """

        if self.edge_reach_max_hops > 0:
            return False
        if self.effective_at_after is not None or self.effective_at_before is not None:
            return False
        return self.data_type is not None or self.source_prefix is not None


__all__ = ("PageMetadataPredicate", "EdgeReachResolver")
