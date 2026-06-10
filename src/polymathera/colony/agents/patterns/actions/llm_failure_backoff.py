"""Exponential-backoff tracker for LLM-cluster failures in an action
policy's main loop.

Pairs with :class:`polymathera.colony.cluster.errors.LLMInferenceError`:
the policy catches that exception around ``plan_step`` and forwards
it here; this class owns the sleep, the backoff doubling, and the
``idle_wait_counter`` increment / decrement so the framework's
existing iteration-accounting machinery (master §) treats the wait
as polling, NOT as a consumed iteration toward ``max_iterations``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any


logger = logging.getLogger(__name__)


class LLMFailureBackoff:
    """Per-policy exponential backoff on LLM-cluster failure.

    Contract:

    - :meth:`handle_failure` is called by the policy's iteration loop
      when ``plan_step`` raised :class:`LLMInferenceError`. The first
      failure in a streak takes ``initial_delay_s`` and increments
      ``agent.metadata.idle_wait_counter`` (so the outer agent loop
      doesn't count this iteration toward ``max_iterations``).
      Subsequent failures double the delay up to ``cap_delay_s``.
    - :meth:`record_success` is called after any successful
      ``plan_step`` return; it decrements the idle-wait token if a
      streak was open and resets the backoff state.

    The idle-wait token is held for the duration of the streak (one
    paired increment / decrement per streak — matches the contract
    in :class:`AgentMetadata.idle_wait_counter`'s docstring). Wall-
    clock cap on a stuck cluster is the mission's
    ``max_runtime_seconds``.
    """

    DEFAULT_INITIAL_DELAY_S: float = 1.0
    DEFAULT_CAP_DELAY_S: float = 60.0

    def __init__(
        self,
        agent: Any,
        *,
        initial_delay_s: float = DEFAULT_INITIAL_DELAY_S,
        cap_delay_s: float = DEFAULT_CAP_DELAY_S,
    ) -> None:
        self._agent = agent
        self._initial_delay_s = initial_delay_s
        self._cap_delay_s = cap_delay_s
        self._next_delay_s: float = 0.0
        self._failure_count: int = 0
        self._is_in_streak: bool = False
        self._last_failure_at: float | None = None
        self._last_error_message: str | None = None

    async def handle_failure(self, exc: BaseException) -> None:
        """Record a failure + sleep the next backoff interval.

        Idempotent on the idle-wait token: only the FIRST failure of
        a streak increments ``idle_wait_counter``; subsequent failures
        only double the delay. The matching decrement happens in
        :meth:`record_success`.
        """

        if not self._is_in_streak:
            self._agent.metadata.idle_wait_counter += 1
            self._is_in_streak = True
        if self._next_delay_s <= 0.0:
            self._next_delay_s = self._initial_delay_s
        else:
            self._next_delay_s = min(
                self._cap_delay_s, self._next_delay_s * 2,
            )
        self._failure_count += 1
        self._last_failure_at = time.time()
        self._last_error_message = str(exc)
        logger.warning(
            "LLMFailureBackoff[%s]: failure #%d — sleeping %.1fs before "
            "next plan_step (cause: %s)",
            getattr(self._agent, "agent_id", "?"),
            self._failure_count, self._next_delay_s, exc,
        )
        await asyncio.sleep(self._next_delay_s)

    def record_success(self) -> None:
        """Close an open streak (decrement idle-wait token, reset
        backoff). No-op when no streak is open."""

        if not self._is_in_streak:
            return
        self._agent.metadata.idle_wait_counter = max(
            0, self._agent.metadata.idle_wait_counter - 1,
        )
        self._is_in_streak = False
        self._next_delay_s = 0.0
        self._failure_count = 0

    def snapshot(self) -> dict[str, Any]:
        """Read-only view for ``get_status_snapshot`` / ``/status``."""

        return {
            "in_backoff_streak": self._is_in_streak,
            "failure_count": self._failure_count,
            "next_delay_s": self._next_delay_s,
            "last_failure_at": self._last_failure_at,
            "last_error_message": self._last_error_message,
        }
