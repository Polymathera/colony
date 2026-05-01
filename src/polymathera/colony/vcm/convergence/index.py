"""``SubscriptionIndex`` — fast lookup from page-change events to subscriptions.

Maintains two indices:

- ``by_data_type`` : ``data_type`` → list of subscriptions whose
  predicate sets that ``data_type``.
- ``by_source_prefix`` : sorted list of (source_prefix, subscription)
  pairs supporting prefix lookup via bisection.

Plus a fallback bucket for predicates that don't index (edge-reach,
effective-at-window, page-id-list-only). The fallback is scanned
linearly per event; in practice such subscriptions are rare.

For a given event the index returns the *union* of:

- Subscriptions whose predicate sets ``data_type == event.data_type``.
- Subscriptions whose predicate sets a ``source_prefix`` that prefixes
  ``event.source``.
- All non-indexable subscriptions.

The caller (``ConvergenceRuntime``) then re-checks each candidate's
``predicate.matches`` to apply the conjunction rules (so the index is
authorised to *over-approximate* but never *under-approximate*).
"""

from __future__ import annotations

import bisect
import threading
from collections import defaultdict
from collections.abc import Iterable

from .predicates import PageMetadataPredicate
from .subscriptions import PageSubscription


class SubscriptionRegistryFull(RuntimeError):
    """Raised when the registry hits its capacity (a safety cap)."""


class SubscriptionIndex:
    """Thread-safe registry + index over ``PageSubscription``s."""

    DEFAULT_MAX_SUBSCRIPTIONS = 100_000

    def __init__(self, max_subscriptions: int = DEFAULT_MAX_SUBSCRIPTIONS) -> None:
        self._lock = threading.RLock()
        self._max = max_subscriptions
        self._by_id: dict[str, PageSubscription] = {}
        self._by_data_type: dict[str, list[str]] = defaultdict(list)
        # (source_prefix, subscription_id) sorted by prefix.
        self._by_source_prefix: list[tuple[str, str]] = []
        self._unindexed: list[str] = []

    # ---- Mutation ------------------------------------------------------

    def add(self, sub: PageSubscription) -> None:
        with self._lock:
            if sub.subscription_id in self._by_id:
                raise ValueError(
                    f"Subscription id {sub.subscription_id!r} already registered.",
                )
            if len(self._by_id) >= self._max:
                raise SubscriptionRegistryFull(
                    f"Subscription registry hit capacity ({self._max}); "
                    "no new subscriptions allowed.",
                )
            self._by_id[sub.subscription_id] = sub
            pred = sub.predicate
            indexed = False
            if pred.data_type is not None:
                self._by_data_type[pred.data_type].append(sub.subscription_id)
                indexed = True
            if pred.source_prefix is not None:
                bisect.insort(
                    self._by_source_prefix,
                    (pred.source_prefix, sub.subscription_id),
                )
                indexed = True
            if not indexed:
                self._unindexed.append(sub.subscription_id)

    def remove(self, subscription_id: str) -> bool:
        with self._lock:
            sub = self._by_id.pop(subscription_id, None)
            if sub is None:
                return False
            pred = sub.predicate
            if pred.data_type is not None:
                bucket = self._by_data_type.get(pred.data_type)
                if bucket is not None:
                    try:
                        bucket.remove(subscription_id)
                    except ValueError:
                        pass
                    if not bucket:
                        self._by_data_type.pop(pred.data_type, None)
            if pred.source_prefix is not None:
                # Linear remove from the sorted list — N is small in
                # practice, and keeping the list sorted lets the prefix
                # lookup run in O(log N + k).
                self._by_source_prefix = [
                    item for item in self._by_source_prefix
                    if item[1] != subscription_id
                ]
            try:
                self._unindexed.remove(subscription_id)
            except ValueError:
                pass
            return True

    def clear(self) -> None:
        with self._lock:
            self._by_id.clear()
            self._by_data_type.clear()
            self._by_source_prefix.clear()
            self._unindexed.clear()

    # ---- Lookup --------------------------------------------------------

    def get(self, subscription_id: str) -> PageSubscription | None:
        with self._lock:
            return self._by_id.get(subscription_id)

    def all(self) -> list[PageSubscription]:
        with self._lock:
            return list(self._by_id.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._by_id)

    def candidates_for(
        self,
        *,
        data_type: str | None,
        source: str,
    ) -> list[PageSubscription]:
        """Return the over-approximate set of subscriptions that *might*
        match an event with the given ``data_type`` and ``source``.

        The caller re-checks each candidate's ``predicate.matches``
        with the full event (including ``scope_id``, ``effective_at``,
        edge reach).
        """

        seen: set[str] = set()
        out: list[PageSubscription] = []
        with self._lock:
            if data_type is not None:
                for sub_id in self._by_data_type.get(data_type, ()):
                    if sub_id in seen:
                        continue
                    seen.add(sub_id)
                    out.append(self._by_id[sub_id])
            for sub_id in self._prefix_candidates_unlocked(source):
                if sub_id in seen:
                    continue
                seen.add(sub_id)
                out.append(self._by_id[sub_id])
            for sub_id in self._unindexed:
                if sub_id in seen:
                    continue
                seen.add(sub_id)
                out.append(self._by_id[sub_id])
        return out

    def _prefix_candidates_unlocked(self, source: str) -> Iterable[str]:
        """Yield subscription ids whose source_prefix prefixes ``source``.

        Uses bisection on the sorted prefix list: only prefixes that are
        ``<=`` source can be candidates, so we can short-circuit.
        """

        idx = bisect.bisect_right(self._by_source_prefix, (source + "\x7f", ""))
        for i in range(idx):
            prefix, sub_id = self._by_source_prefix[i]
            if source.startswith(prefix):
                yield sub_id


__all__ = ("SubscriptionIndex", "SubscriptionRegistryFull")
