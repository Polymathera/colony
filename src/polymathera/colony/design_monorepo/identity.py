"""Per-commit attribution for the design monorepo.

Every commit produced by the framework carries two pieces of
attribution:

- A **principal** — the actor whose identity appears as
  ``author`` / ``committer`` on the commit. By default this is the
  *colony* (a collective identity that survives the ephemeral
  agents); operators can change it per-colony to the user, the
  specific agent, or any agent-type label.
- An optional **co-author** — a second identity stamped into the
  commit message via the ``Co-Authored-By:`` trailer. Default is the
  user, so the human who started the session is searchable in
  ``git log --grep`` and visible in GitHub UI even when an
  autonomous agent did the actual work.

The well-known principal strings are ``"user"``, ``"colony"``,
``"agent"``. Anything else is treated as an *agent-type* label
(e.g. ``"session_agent"``, ``"simulation_agent"``) so the framework
can carry attribution that's coarser than the ephemeral
``agent_id`` but finer than ``colony``. Each principal resolves to a
``CommitIdentity`` (a ``(name, email)`` pair the GitPython ``Actor``
constructor accepts) via :func:`resolve_commit_identity`.

This is the audit trail master §8.5 specifies. The identity is set on
the GitPython ``Repo`` *per commit*, not in the global git config —
two simultaneous commits from different agents on the same working
tree must not clobber each other's authorship.

Optional GPG signing is supported through git's ``commit.gpgsign`` and
``user.signingkey`` config keys, set transactionally for the duration
of a single commit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from git import Actor


_ROLE_DEFAULT = "agent"

# Default agent-email domain when the manifest doesn't override it.
# Mirrors :attr:`DesignMonorepoManifest.agent_email_domain`'s default.
_DEFAULT_AGENT_EMAIL_DOMAIN = "agent.colony.local"

# Free-form principal strings carry these well-known values; anything
# else is interpreted as an agent-type label.
PRINCIPAL_USER = "user"
PRINCIPAL_COLONY = "colony"
PRINCIPAL_AGENT = "agent"


@dataclass(frozen=True)
class CommitIdentity:
    """A name + email pair used as ``author`` / ``committer`` and as
    the value of the ``Co-Authored-By:`` trailer.

    Built by :func:`resolve_commit_identity` from a free-form
    principal string. The resolver knows how to render every
    well-known principal kind (``user``, ``colony``, ``agent``) and
    falls back to a synthetic agent-type identity for anything else.
    """

    git_name: str
    git_email: str
    signing_key: str | None = None

    def actor(self) -> "Actor":
        """Build a GitPython ``Actor`` from this identity."""

        from git import Actor as _Actor

        return _Actor(self.git_name, self.git_email)

    def trailer(self) -> str:
        """Render the ``Co-Authored-By:`` trailer line. The trailing
        newline is the caller's concern."""

        return f"Co-Authored-By: {self.git_name} <{self.git_email}>"


def resolve_commit_identity(
    principal: str,
    *,
    colony_id: str,
    agent_id: str | None = None,
    role: str | None = None,
    user_name: str | None = None,
    user_email: str | None = None,
    agent_email_domain: str = _DEFAULT_AGENT_EMAIL_DOMAIN,
    signing_key: str | None = None,
) -> CommitIdentity:
    """Translate a free-form ``principal`` string into a concrete
    :class:`CommitIdentity`.

    Well-known principals:

    - ``"user"`` — uses ``user_name`` / ``user_email`` directly. These
      come from the colony row's ``git_user_name`` / ``git_user_email``
      fields. Raises ``ValueError`` when they're missing — operator
      must configure them on the landing page before selecting
      ``"user"`` as principal or co-author.
    - ``"colony"`` — synthetic collective identity for the whole
      colony. Renders as ``colony:<colony_id> <colony_id>@<domain>``.
      Persists across the ephemeral lifetime of individual agents,
      which is why it makes a sensible default principal.
    - ``"agent"`` — the specific ephemeral agent. Renders as the
      original ``agent:<agent_id>:<role> <agent_id>@<colony_id>.<domain>``
      form (master §8.5 audit-trail discipline). Requires
      ``agent_id``; raises ``ValueError`` otherwise.
    - any other string — treated as an *agent-type* label
      (``"session_agent"``, ``"simulation_agent"``, …). Renders as
      ``<type>:<colony_id> <type>@<colony_id>.<domain>``. Useful when
      the framework wants attribution coarser than per-instance but
      finer than per-colony.

    The resulting :class:`CommitIdentity` is fed to GitPython's
    ``Actor`` for author/committer fields, or rendered as the
    ``Co-Authored-By:`` trailer when used as the co-author.
    """

    if principal == PRINCIPAL_USER:
        if not user_name or not user_email:
            raise ValueError(
                "principal='user' requires git_user_name and "
                "git_user_email to be configured on the colony "
                "(landing page → Colonies → pencil → Save).",
            )
        return CommitIdentity(
            git_name=user_name,
            git_email=user_email,
            signing_key=signing_key,
        )
    if principal == PRINCIPAL_COLONY:
        return CommitIdentity(
            git_name=f"colony:{colony_id}",
            git_email=f"{colony_id}@{agent_email_domain}",
            signing_key=signing_key,
        )
    if principal == PRINCIPAL_AGENT:
        if not agent_id:
            raise ValueError(
                "principal='agent' requires an agent_id; this is "
                "missing when the capability is in detached mode.",
            )
        return CommitIdentity(
            git_name=f"agent:{agent_id}:{role or _ROLE_DEFAULT}",
            git_email=f"{agent_id}@{colony_id}.{agent_email_domain}",
            signing_key=signing_key,
        )
    # Fall-through: agent-type label. Coarser than per-instance,
    # finer than per-colony — useful for grouping commits by the
    # *kind* of agent that produced them across many ephemeral
    # instances.
    return CommitIdentity(
        git_name=f"{principal}:{colony_id}",
        git_email=f"{principal}@{colony_id}.{agent_email_domain}",
        signing_key=signing_key,
    )


@dataclass(frozen=True)
class AgentIdentity:
    """Identity used to author and commit changes to the design monorepo.

    Kept for backward-compatibility with callers (notably
    ``DesignMonorepoClient.commit_with_identity`` and
    ``bootstrap_design_monorepo``) that hand-construct an agent
    identity. New code should resolve a :class:`CommitIdentity` via
    :func:`resolve_commit_identity` instead.
    """

    agent_id: str
    role: str = _ROLE_DEFAULT
    colony_id: str = "default"
    agent_email_domain: str = _DEFAULT_AGENT_EMAIL_DOMAIN
    signing_key: str | None = None

    @property
    def git_name(self) -> str:
        """The ``user.name`` form: ``agent:<agent_id>:<role>``."""
        return f"agent:{self.agent_id}:{self.role}"

    @property
    def git_email(self) -> str:
        """The ``user.email`` form: ``<agent_id>@<colony_id>.<domain>``."""
        return f"{self.agent_id}@{self.colony_id}.{self.agent_email_domain}"

    def actor(self) -> Actor:
        """Build a GitPython ``Actor`` from this identity."""

        from git import Actor as _Actor

        return _Actor(self.git_name, self.git_email)

    def to_commit_identity(self) -> CommitIdentity:
        """Convert to the new :class:`CommitIdentity` shape so legacy
        ``AgentIdentity`` callers can flow through the unified commit
        helper without churn."""

        return CommitIdentity(
            git_name=self.git_name,
            git_email=self.git_email,
            signing_key=self.signing_key,
        )


def append_co_author_trailer(
    message: str, co_author: CommitIdentity | None,
) -> str:
    """Return ``message`` with a ``Co-Authored-By:`` trailer appended.

    When ``co_author`` is ``None`` the message is returned unchanged.
    The trailer follows the
    `git interpret-trailers`_ shape (one blank line before the trailer
    block, key-value form), which GitHub parses for the "co-authored"
    avatar in commit views.

    .. _git interpret-trailers: https://git-scm.com/docs/git-interpret-trailers
    """

    if co_author is None:
        return message
    body = message.rstrip("\n")
    return f"{body}\n\n{co_author.trailer()}\n"


__all__ = (
    "AgentIdentity",
    "CommitIdentity",
    "PRINCIPAL_AGENT",
    "PRINCIPAL_COLONY",
    "PRINCIPAL_USER",
    "append_co_author_trailer",
    "resolve_commit_identity",
)
