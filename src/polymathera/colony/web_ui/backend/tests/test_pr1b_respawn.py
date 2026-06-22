"""Tests for PR1-B (R12-ROOT-CAUSE-C) — SessionAgent auto-respawn.

PR1-A added the user-visible death message. PR1-B closes the loop:
the dashboard's chat router detects a dead SessionAgent on the next
user activity and rebuilds it via the extracted blueprint factory
in ``chat/user_session_factory.py``. The CAS-guarded
``SessionManager.replace_session_agent_id`` endpoint atomically swaps
the session's ``session_agent_id`` pointer so concurrent respawns
don't race.

We pin three layers:

1. **Factory exports** — ``user_session_factory.py`` exposes the
   blueprint + spawn helper used by both create-session and respawn.
2. **Endpoint surface** — ``SessionManager.replace_session_agent_id``
   exists, is a ``@serving.endpoint``, and accepts the CAS shape.
3. **Wiring** — ``routers/sessions.py:create_session`` calls the
   factory's spawn helper (no longer inline); ``routers/chat.py``
   defines ``ensure_session_agent_alive`` and ``_post_user_message``
   calls it before writing to the chat blackboard.

We do NOT pin the end-to-end respawn loop here — that requires a
running cluster + AgentSystem. The above source-level pins catch
the regressions cheaply; live e2e is the next deploy.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Layer 1: Factory exports
# ---------------------------------------------------------------------------


def test_factory_exports_build_blueprint_and_spawn_helper() -> None:
    """The two public entry points the respawn path depends on must
    be present and callable. Pin by import — a typo in the function
    name would surface here, not at first respawn."""

    from polymathera.colony.web_ui.backend.chat.user_session_factory import (
        build_user_session_agent_blueprint,
        spawn_user_session_agent_for_session,
        _resolve_github_identity,
    )
    assert callable(build_user_session_agent_blueprint)
    assert callable(spawn_user_session_agent_for_session)
    assert callable(_resolve_github_identity)


def test_resolve_github_identity_returns_five_keys_with_none_inputs() -> None:
    """The shape downstream readers (GitHubCapability) rely on:
    five keys, all ``None`` when neither row is populated. Pin so a
    refactor that drops a key surfaces here, not as a KeyError in
    the agent at runtime."""

    from polymathera.colony.web_ui.backend.chat.user_session_factory import (
        _resolve_github_identity,
    )
    result = _resolve_github_identity(None, None)
    assert set(result.keys()) == {
        "tenant_installation_id",
        "user_github_login",
        "user_github_id",
        "git_user_email",
        "git_user_name",
    }
    assert all(v is None for v in result.values())


def test_resolve_github_identity_combines_tenant_and_user_rows() -> None:
    """Both rows populated: tenant_installation_id from the tenant row;
    the four user fields from the user row. Mirrors the chat-side
    create_session path's contract."""

    from polymathera.colony.web_ui.backend.chat.user_session_factory import (
        _resolve_github_identity,
    )
    result = _resolve_github_identity(
        {"installation_id": "inst-123"},
        {
            "github_login": "octocat",
            "github_user_id": "u-9",
            "github_email": "o@x.com",
            "git_user_name": "Octo Cat",
        },
    )
    assert result["tenant_installation_id"] == "inst-123"
    assert result["user_github_login"] == "octocat"
    assert result["user_github_id"] == "u-9"
    assert result["git_user_email"] == "o@x.com"
    assert result["git_user_name"] == "Octo Cat"


# ---------------------------------------------------------------------------
# Layer 2: SessionManager.replace_session_agent_id surface
# ---------------------------------------------------------------------------


def test_session_manager_has_replace_session_agent_id() -> None:
    """The CAS-guarded swap endpoint must exist on
    SessionManagerDeployment. Without it, the chat router has no
    way to atomically wire a replacement SessionAgent — the
    original ``set_session_agent_id`` refuses to overwrite a
    non-null value (intentional, for the initial-spawn race)."""

    from polymathera.colony.agents.sessions.manager import (
        SessionManagerDeployment,
    )
    assert hasattr(SessionManagerDeployment, "replace_session_agent_id")


def test_replace_session_agent_id_accepts_cas_signature() -> None:
    """The shape the chat router calls with: ``session_id``,
    ``new_agent_id``, and the keyword-only ``expected_old_agent_id``
    for the CAS guard."""

    import inspect
    from polymathera.colony.agents.sessions.manager import (
        SessionManagerDeployment,
    )
    sig = inspect.signature(
        SessionManagerDeployment.replace_session_agent_id,
    )
    params = sig.parameters
    assert "session_id" in params
    assert "new_agent_id" in params
    assert "expected_old_agent_id" in params
    # Keyword-only so the CAS arg can't be passed positionally.
    assert (
        params["expected_old_agent_id"].kind
        == inspect.Parameter.KEYWORD_ONLY
    )


# ---------------------------------------------------------------------------
# Layer 3: Wiring at create-session and chat-router sites
# ---------------------------------------------------------------------------


def _router_source(name: str) -> str:
    return (
        Path(__file__).resolve().parents[1]
        / "routers"
        / name
    ).read_text(encoding="utf-8")


def _chat_module_source(name: str) -> str:
    return (
        Path(__file__).resolve().parents[1]
        / "chat"
        / name
    ).read_text(encoding="utf-8")


def test_create_session_uses_factory_spawn_helper() -> None:
    """``routers/sessions.py:create_session`` must call the
    extracted factory's spawn helper instead of inlining the
    blueprint construction. Source-pin so a future PR that drifts
    back to the inline shape surfaces here, not as a divergence
    between create-session and respawn."""

    src = _router_source("sessions.py")
    assert "spawn_user_session_agent_for_session" in src
    assert "from ..chat.user_session_factory import" in src
    # The pre-extraction inline ``SessionAgent.bind(...)`` is gone
    # from the create-session path — pin the absence so the next
    # refactor doesn't accidentally restore a parallel shape.
    create_session_section = src.split("@router.post(\"/sessions/\"")[-1]
    create_session_section = create_session_section.split("@router")[0]
    assert "SessionAgent.bind(" not in create_session_section


def test_create_session_persists_user_sub_on_session_metadata() -> None:
    """The user's auth ``sub`` must travel onto
    ``SessionMetadata.created_by`` at create time so respawn can
    rebuild the GitHub identity for the same user."""

    src = _router_source("sessions.py")
    assert "created_by=user.get(\"sub\", \"\")" in src


def test_chat_router_defines_ensure_session_agent_alive() -> None:
    """The respawn entry point lives on the dashboard side
    (chat router has postgres + ColonyConnection); pin its
    presence."""

    src = _router_source("chat.py")
    assert "async def ensure_session_agent_alive" in src


def test_post_user_message_calls_ensure_alive_before_write() -> None:
    """Source-pin the call order: ``ensure_session_agent_alive``
    runs BEFORE ``_get_session_chat_blackboard`` and the
    ``bb.write`` for the user message. Without this, a dead
    SessionAgent silently swallows the user's message
    (R12-ROOT-CAUSE-C symptom)."""

    src = _router_source("chat.py")
    # Find the body of _post_user_message
    fn_start = src.find("async def _post_user_message(")
    assert fn_start > 0
    # Take a generous slice past the function header
    fn_body = src[fn_start: fn_start + 4000]
    ensure_idx = fn_body.find("ensure_session_agent_alive(")
    write_idx = fn_body.find("await bb.write(")
    assert ensure_idx > 0, "ensure_session_agent_alive not called in _post_user_message"
    assert write_idx > 0, "bb.write not present in _post_user_message"
    assert ensure_idx < write_idx, (
        "ensure_session_agent_alive must run BEFORE bb.write so a "
        "dead SessionAgent is respawned before the user's message "
        "is enqueued."
    )


def test_session_info_carries_created_by() -> None:
    """``_SessionInfo`` must surface ``created_by`` so respawn has
    the user_sub to resolve GitHub identity for."""

    src = _router_source("chat.py")
    # Source-level pin (the dataclass is private)
    assert "created_by: str" in src
    assert "created_by=session.metadata.created_by" in src


# ---------------------------------------------------------------------------
# PR1-C: orphan user-message replay on respawn
# ---------------------------------------------------------------------------


def test_ensure_session_agent_alive_replays_orphan_user_messages() -> None:
    """After a successful respawn, ``_replay_orphaned_user_messages``
    re-writes any chat:user:* keys that arrived after the latest
    chat:agent:* response. Without this, messages sent during the
    dead-agent gap rot — the new agent's subscription only sees
    FUTURE writes."""

    src = _router_source("chat.py")
    # Replay helper exists and is called from ensure_session_agent_alive.
    assert "async def _replay_orphaned_user_messages(" in src
    # Wiring: ensure_session_agent_alive awaits the replay after
    # the CAS-swap succeeds. Pin by ordering — replay must come
    # AFTER the "respawned SessionAgent" log so it runs on a live
    # new agent only.
    ensure_idx = src.find("async def ensure_session_agent_alive")
    ensure_body = src[ensure_idx: ensure_idx + 6000]
    log_idx = ensure_body.find("respawned ")
    replay_idx = ensure_body.find("_replay_orphaned_user_messages(")
    assert log_idx > 0 and replay_idx > 0
    assert replay_idx > log_idx


def test_replay_filters_to_orphans_after_latest_agent_response() -> None:
    """The replay filters chat:user:* by timestamp > latest
    chat:agent:* timestamp. Pin the comparison shape so a refactor
    doesn't re-introduce blind replay-of-everything (which would
    re-fire already-answered messages on every respawn)."""

    src = _router_source("chat.py")
    fn_idx = src.find("async def _replay_orphaned_user_messages")
    body = src[fn_idx: fn_idx + 4000]
    assert "latest_agent_ts" in body
    assert "agent_message_pattern" in body
    assert "user_message_pattern" in body
    # Replay re-writes with a fresh msg_replay_ key, not the original
    # key (idempotent across re-respawns).
    assert "msg_replay_" in body
    assert "replayed_from" in body
