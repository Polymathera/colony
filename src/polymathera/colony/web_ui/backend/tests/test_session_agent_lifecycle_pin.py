"""Regression pin for the SessionAgent's lifecycle declaration in
``routers/sessions.py``.

R11 forensic: the SessionAgent was killed at iteration 20 in the
middle of a conversation because ``AgentMetadata.lifecycle_mode``
defaulted to ``ONE_SHOT`` and ``max_iterations`` defaulted to ``20``
— both inappropriate for a long-lived chat-service agent that
processes many user messages over a session.

The fix is two fields on the ``AgentMetadata(...)`` call site in
``routers/sessions.py``. This test pins both at the source level —
because the construction is inline inside the session-creation
handler (a long async function with router dependencies), a
behavioral test would require setting up the full router pipeline.
A source-level pin is enough to catch the regression cheaply: if a
future edit removes either line, this test fails immediately rather
than waiting for the next user-visible silent death.

If/when the ``AgentMetadata(...)`` construction is extracted into a
pure builder function, this test should be rewritten as a direct
behavioral assertion on that function's output."""

from __future__ import annotations

from pathlib import Path


# PR1-B extracted the user-session blueprint construction from
# ``routers/sessions.py:create_session`` into
# ``chat/user_session_factory.py:build_user_session_agent_blueprint``.
# These pins follow the code to the factory; the router's call site
# is pinned by ``test_pr1b_respawn.py`` (which asserts the router
# calls the factory's spawn helper instead of inlining the blueprint).
_USER_SESSION_FACTORY = (
    Path(__file__).resolve().parents[1]
    / "chat"
    / "user_session_factory.py"
)


def _user_session_factory_source() -> str:
    return _USER_SESSION_FACTORY.read_text(encoding="utf-8")


def test_session_agent_metadata_declares_continuous_lifecycle() -> None:
    """``LifecycleMode.CONTINUOUS`` is required so the agent loop
    bypasses the iteration cap via
    :func:`effective_loop_max_iterations`. Without it, the loop
    inherits the default ONE_SHOT cap and the SessionAgent dies in
    the middle of long conversations."""

    src = _user_session_factory_source()
    assert "lifecycle_mode=LifecycleMode.CONTINUOUS" in src, (
        "chat/user_session_factory.py must construct the SessionAgent's "
        "AgentMetadata with lifecycle_mode=LifecycleMode.CONTINUOUS. "
        "See R11 forensic + test_lifecycle_aware_iteration_cap.py for "
        "why."
    )


def test_session_agent_metadata_declares_max_iterations_none() -> None:
    """Explicit ``max_iterations=None`` at the construction site
    makes intent explicit even though
    :func:`effective_loop_max_iterations` would also bypass the
    configured cap for CONTINUOUS agents. The redundancy is
    intentional: a future reader doesn't have to chase the loop
    logic to learn that the cap is meant to be absent here."""

    src = _user_session_factory_source()
    assert "max_iterations=None" in src, (
        "chat/user_session_factory.py must construct the SessionAgent's "
        "AgentMetadata with max_iterations=None. See R11 forensic + "
        "test_lifecycle_aware_iteration_cap.py for why."
    )


def test_session_agent_metadata_imports_lifecycle_mode() -> None:
    """The lifecycle-mode import must travel with the construction;
    a missing import would surface here, not at deploy time."""

    src = _user_session_factory_source()
    assert "LifecycleMode" in src
    assert (
        "from polymathera.colony.agents.models import LifecycleMode" in src
    )


# ---------------------------------------------------------------------------
# Parallel pin for the SYSTEM session agent (system_session.py)
# ---------------------------------------------------------------------------
#
# The system session is an always-on per-colony service that hosts
# GitHubInbound polling, mention routing, and the interaction log.
# Pre-R11 it inherited the default ``max_iterations=20`` +
# ``lifecycle_mode=ONE_SHOT`` (the system-session metadata factory
# didn't override either), so it had the same silent-death pattern
# as the user SessionAgent: ran ~20 reasoning iterations and was
# killed mid-work. The fix declares CONTINUOUS lifecycle +
# ``max_iterations=None`` at the construction site. This source-level
# pin prevents a regression — if either field is removed, the test
# fails immediately rather than waiting for the next colony-wide
# silent death.


_SYSTEM_SESSION_MODULE = (
    Path(__file__).resolve().parents[1]
    / "chat"
    / "system_session.py"
)


def _system_session_source() -> str:
    return _SYSTEM_SESSION_MODULE.read_text(encoding="utf-8")


def test_system_session_metadata_declares_continuous_lifecycle() -> None:
    """The system session agent must declare CONTINUOUS lifecycle
    so :func:`effective_loop_max_iterations` bypasses the iteration
    cap. Without it, the per-colony singleton host dies after 20
    iterations of GitHub-poll + mention-router work."""

    src = _system_session_source()
    assert "lifecycle_mode=LifecycleMode.CONTINUOUS" in src, (
        "chat/system_session.py must construct the system "
        "SessionAgent's AgentMetadata with "
        "lifecycle_mode=LifecycleMode.CONTINUOUS — same reason as "
        "the user SessionAgent (R11 forensic)."
    )


def test_system_session_metadata_declares_max_iterations_none() -> None:
    """Explicit ``max_iterations=None`` at the construction site —
    same rationale as the user SessionAgent."""

    src = _system_session_source()
    assert "max_iterations=None" in src, (
        "chat/system_session.py must construct the system "
        "SessionAgent's AgentMetadata with max_iterations=None — "
        "same reason as the user SessionAgent (R11 forensic)."
    )
