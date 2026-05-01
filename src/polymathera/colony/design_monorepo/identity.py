"""Per-commit agent identity for the design monorepo.

Every commit produced by an agent carries a transactional identity:

    user.name  = ``agent:<agent_id>:<role>``
    user.email = ``<agent_id>@<colony_id>.<agent_email_domain>``

This is the audit trail master §8.5 specifies. The identity is set on
the GitPython ``Repo`` *per commit*, not in the global git config —
two simultaneous commits from different agents on the same working
copy must not clobber each other's authorship.

GitPython supports per-commit identity via the ``Actor`` argument to
``Repo.index.commit(actor=..., committer=...)``. The wrapper layer in
``client.py`` uses this helper rather than mutating ``.git/config``.

Optional GPG signing is supported through git's ``commit.gpgsign`` and
``user.signingkey`` config keys, set transactionally for the duration
of a single commit.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from git import Actor, Repo


_ROLE_DEFAULT = "agent"


@dataclass(frozen=True)
class AgentIdentity:
    """Identity used to author and commit changes to the design monorepo.

    The framework constructs this once per agent boundary (typically at
    the start of an action) and passes it into the wrapper layer. The
    wrapper layer turns it into a GitPython ``Actor`` for each commit.
    """

    agent_id: str
    role: str = _ROLE_DEFAULT
    colony_id: str = "default"
    agent_email_domain: str = "agent.colony.local"
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


@contextmanager
def signing_enabled(repo: "Repo", identity: AgentIdentity) -> Iterator[None]:
    """Transactionally enable commit signing for the duration of a block.

    The git config keys ``commit.gpgsign`` and ``user.signingkey`` are
    set on the repo's local config at entry and reverted on exit, so
    the signing key is never written to a shared config that another
    agent could pick up.

    No-op when ``identity.signing_key`` is None.
    """

    if identity.signing_key is None:
        yield
        return

    cw = repo.config_writer(config_level="repository")
    prev_gpgsign = None
    prev_signingkey = None
    try:
        try:
            prev_gpgsign = cw.get_value("commit", "gpgsign")
        except Exception:  # noqa: BLE001 - GitPython raises subclasses
            prev_gpgsign = None
        try:
            prev_signingkey = cw.get_value("user", "signingkey")
        except Exception:  # noqa: BLE001
            prev_signingkey = None
        cw.set_value("commit", "gpgsign", "true")
        cw.set_value("user", "signingkey", identity.signing_key)
        cw.release()
        yield
    finally:
        cw = repo.config_writer(config_level="repository")
        try:
            if prev_gpgsign is None:
                cw.remove_option("commit", "gpgsign")
            else:
                cw.set_value("commit", "gpgsign", prev_gpgsign)
            if prev_signingkey is None:
                cw.remove_option("user", "signingkey")
            else:
                cw.set_value("user", "signingkey", prev_signingkey)
        finally:
            cw.release()


__all__ = ("AgentIdentity", "signing_enabled")
