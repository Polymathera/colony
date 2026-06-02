"""Tests for ``resolve_commit_identity`` and the
``Co-Authored-By:`` trailer helper.

Pin the well-known principal kinds (``user`` / ``colony`` /
``agent``) and the agent-type fallback so a future refactor can't
silently change what gets stamped onto commits.
"""

from __future__ import annotations

import pytest

from polymathera.colony.design_monorepo.identity import (
    CommitIdentity,
    PRINCIPAL_AGENT,
    PRINCIPAL_COLONY,
    PRINCIPAL_USER,
    append_co_author_trailer,
    resolve_commit_identity,
)


def test_user_principal_uses_configured_name_and_email() -> None:
    i = resolve_commit_identity(
        PRINCIPAL_USER,
        colony_id="c1",
        user_name="Ada Lovelace",
        user_email="ada@example.com",
    )
    assert isinstance(i, CommitIdentity)
    assert i.git_name == "Ada Lovelace"
    assert i.git_email == "ada@example.com"


def test_user_principal_without_name_and_email_raises() -> None:
    with pytest.raises(ValueError, match="Connect GitHub"):
        resolve_commit_identity(PRINCIPAL_USER, colony_id="c1")


def test_colony_principal_renders_collective_identity() -> None:
    i = resolve_commit_identity(PRINCIPAL_COLONY, colony_id="c1")
    assert i.git_name == "colony:c1"
    assert i.git_email == "c1@agent.colony.local"


def test_agent_principal_preserves_master_8_5_form() -> None:
    """Master spec §8.5: agent commits use
    ``agent:<agent_id>:<role>`` and
    ``<agent_id>@<colony_id>.<domain>``. Pin it so the audit-trail
    discipline stays intact when ``agent`` is selected.
    """
    i = resolve_commit_identity(
        PRINCIPAL_AGENT, colony_id="c1",
        agent_id="agent-7", role="session_orchestrator",
    )
    assert i.git_name == "agent:agent-7:session_orchestrator"
    assert i.git_email == "agent-7@c1.agent.colony.local"


def test_agent_principal_without_agent_id_raises() -> None:
    with pytest.raises(ValueError, match="agent_id"):
        resolve_commit_identity(PRINCIPAL_AGENT, colony_id="c1")


def test_agent_type_label_synthesises_identity() -> None:
    """Anything other than the three well-known principals is
    treated as an agent-type label — coarser than per-instance,
    finer than per-colony. Useful when agent types stabilise but
    instances stay ephemeral.
    """
    i = resolve_commit_identity("session_agent", colony_id="c1")
    assert i.git_name == "session_agent:c1"
    assert i.git_email == "session_agent@c1.agent.colony.local"


def test_trailer_appended_for_co_authored_commit() -> None:
    co = resolve_commit_identity(PRINCIPAL_COLONY, colony_id="c1")
    out = append_co_author_trailer("init: scaffold", co)
    assert out.endswith(
        "\n\nCo-Authored-By: colony:c1 <c1@agent.colony.local>\n"
    )
    assert out.startswith("init: scaffold")


def test_trailer_is_a_noop_when_co_author_is_none() -> None:
    assert append_co_author_trailer("just a commit", None) == "just a commit"
