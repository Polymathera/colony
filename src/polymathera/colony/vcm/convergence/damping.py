"""``ConvergenceDamper`` — numeric tolerance for capability outputs.

Per master §5.2 mechanism 3: "A capability that produces a numeric
output (a budget propagation, an MDO step, a confidence interval) must
declare a tolerance. If two consecutive runs produce outputs within
tolerance, the capability is treated as converged and downstream
subscribers do not re-fire. This breaks the most common kind of
fixed-point cycle."

The damper caches the most-recent output value per
``(subscription_id, page_id)`` and, on a fresh dispatch attempt,
returns ``True`` (converged → skip) when the new value is within the
declared ``NumericTolerance``.

Output values are interpreted as scalars (a single float) or vectors
(a tuple/list of floats); the L∞ (max absolute difference) norm is
used. Non-numeric outputs are treated as "always changed" — the damper
returns ``False`` and the capability re-fires. This keeps the damping
opt-in: a capability that doesn't declare a tolerance never benefits
from damping but never gets *suppressed* either.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .subscriptions import NumericTolerance


_NumericValue = float | int | bool | Sequence[float] | Sequence[int]


@dataclass(frozen=True)
class _CachedOutput:
    value: tuple[float, ...]


class ConvergenceDamper:
    """Tracks per-(subscription, page) output values with tolerance check."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cache: dict[tuple[str, str], _CachedOutput] = {}

    def is_converged(
        self,
        *,
        subscription_id: str,
        page_id: str,
        new_output: Any,
        tolerance: NumericTolerance | None,
    ) -> bool:
        """Return True iff the new output is within tolerance of the
        cached one (and tolerance is set).

        ``False`` means "downstream should re-fire". ``True`` means
        "treat as converged; skip downstream dispatch".

        The cache is *unconditionally updated* on every call so the
        damper tracks what was actually produced. When tolerance is
        ``None`` the cache is still updated (to support a later
        decision to set tolerance), but the function always returns
        ``False``.
        """

        if tolerance is None:
            self._record(subscription_id, page_id, new_output)
            return False

        new_vec = self._coerce(new_output)
        if new_vec is None:
            # Non-numeric output: nothing to compare. Reset cache and
            # report not-converged so downstream still fires.
            self._record(subscription_id, page_id, new_output)
            return False

        key = (subscription_id, page_id)
        with self._lock:
            cached = self._cache.get(key)
            if cached is None:
                self._cache[key] = _CachedOutput(new_vec)
                return False
            if len(cached.value) != len(new_vec):
                # Shape change: not converged; refresh cache.
                self._cache[key] = _CachedOutput(new_vec)
                return False
            converged = self._within_tolerance(
                old=cached.value, new=new_vec, tol=tolerance,
            )
            self._cache[key] = _CachedOutput(new_vec)
            return converged

    def reset(
        self,
        *,
        subscription_id: str | None = None,
        page_id: str | None = None,
    ) -> None:
        """Drop cached outputs.

        Both filters optional: passing both clears the one matching key;
        passing one clears all entries with that subscription_id /
        page_id; passing neither clears everything.
        """

        with self._lock:
            if subscription_id is None and page_id is None:
                self._cache.clear()
                return
            keys = list(self._cache.keys())
            for sid, pid in keys:
                if subscription_id is not None and sid != subscription_id:
                    continue
                if page_id is not None and pid != page_id:
                    continue
                self._cache.pop((sid, pid), None)

    # ---- Internals -----------------------------------------------------

    def _record(
        self, subscription_id: str, page_id: str, new_output: Any,
    ) -> None:
        vec = self._coerce(new_output)
        if vec is None:
            with self._lock:
                self._cache.pop((subscription_id, page_id), None)
            return
        with self._lock:
            self._cache[(subscription_id, page_id)] = _CachedOutput(vec)

    @staticmethod
    def _coerce(value: Any) -> tuple[float, ...] | None:
        """Coerce ``value`` to a tuple of floats, or return None for
        non-numeric values.

        Booleans count as numeric (True=1.0, False=0.0); strings,
        dicts, None do not.
        """

        if isinstance(value, bool):
            return (float(value),)
        if isinstance(value, (int, float)):
            return (float(value),)
        if isinstance(value, (list, tuple)) and value:
            try:
                return tuple(float(v) for v in value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None
        return None

    @staticmethod
    def _within_tolerance(
        *,
        old: tuple[float, ...],
        new: tuple[float, ...],
        tol: NumericTolerance,
    ) -> bool:
        if tol.mode == "absolute":
            for a, b in zip(old, new):
                if abs(a - b) > tol.value:
                    return False
            return True
        # relative
        for a, b in zip(old, new):
            denom = max(abs(a), tol.epsilon)
            if abs(a - b) / denom > tol.value:
                return False
        return True


__all__ = ("ConvergenceDamper",)
