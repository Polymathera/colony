"""In-memory progress tracking for the OAuth sign-in walker.

The OAuth callback returns immediately with an HTML loading page +
auth cookies, then runs the (potentially slow — repo discovery
walks N repos through the provider's API) walker as a background
task. The loading page polls
``GET /api/v1/auth/sign-in-progress/{nonce}`` to surface walker
progress to the user.

State is in-memory, dashboard-process-local. That's fine because:

- Each sign-in is a transient flow that completes in seconds-to-minutes.
- The progress entry is keyed by the per-flow ``state`` nonce the
  callback already verified for CSRF.
- A dashboard restart drops in-flight progress, but the user can
  just re-click "Sign in with GitHub" and start over — no
  persistent state to recover.

Concurrent sign-ins (multiple browser tabs, two users) are isolated
by the nonce key. A held reference to the background task
(``_tasks``) prevents the asyncio garbage collector from cancelling
it while the polling page reads progress.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# Seconds after ``mark_done`` before the entry is purged. Long
# enough that a slow client poll picks up the final state, short
# enough that the dict doesn't grow unboundedly.
_TTL_AFTER_DONE_S = 60.0


@dataclass
class SigninProgress:
    """Per-nonce progress state. Mutated by the walker via
    :func:`emit` / :func:`mark_done`; read by the polling endpoint
    via :func:`get`."""

    nonce: str
    messages: list[str] = field(
        default_factory=lambda: ["Starting…"],
    )
    done: bool = False
    error: str | None = None
    # Where the JS should send the user when ``done=True``. Defaults
    # to ``/`` but the callback can stash a custom return URL if we
    # want post-signin landing tweaks later.
    redirect_to: str = "/"


_progress: dict[str, SigninProgress] = {}
_tasks: dict[str, asyncio.Task] = {}


def start(nonce: str) -> SigninProgress:
    """Begin tracking sign-in progress for ``nonce``. Returns the
    fresh state so the caller can inspect / extend before spawning
    the background task."""
    p = SigninProgress(nonce=nonce)
    _progress[nonce] = p
    return p


def emit(nonce: str, message: str) -> None:
    """Append a progress message. No-op when the nonce isn't tracked
    (walker fired against a state we already cleaned up) or when
    the flow is already marked ``done`` (don't append after the JS
    has redirected)."""
    p = _progress.get(nonce)
    if p is None or p.done:
        return
    p.messages.append(message)


def mark_done(nonce: str, *, error: str | None = None) -> None:
    """Flag the flow as complete. ``error`` non-None tells the JS to
    render an error state instead of redirecting. Schedules a delayed
    cleanup so the polling client has time to see the final state."""
    p = _progress.get(nonce)
    if p is None:
        return
    p.done = True
    p.error = error
    asyncio.create_task(_cleanup_after_delay(nonce))


def get(nonce: str) -> SigninProgress | None:
    """Read the current state, or None when the nonce isn't tracked."""
    return _progress.get(nonce)


def register_task(nonce: str, task: asyncio.Task) -> None:
    """Hold a reference to the background walker task so the asyncio
    garbage collector doesn't cancel it. Released by the delayed
    cleanup."""
    _tasks[nonce] = task


async def _cleanup_after_delay(nonce: str) -> None:
    """Drop the entry after ``_TTL_AFTER_DONE_S`` so the in-memory
    dict doesn't grow unboundedly across many sign-ins."""
    try:
        await asyncio.sleep(_TTL_AFTER_DONE_S)
    except asyncio.CancelledError:
        return
    _progress.pop(nonce, None)
    _tasks.pop(nonce, None)


def reset_for_testing() -> None:
    """Test-only — wipe both dicts so each test starts clean."""
    _progress.clear()
    for task in _tasks.values():
        task.cancel()
    _tasks.clear()


__all__ = (
    "SigninProgress",
    "emit",
    "get",
    "mark_done",
    "register_task",
    "reset_for_testing",
    "start",
)
