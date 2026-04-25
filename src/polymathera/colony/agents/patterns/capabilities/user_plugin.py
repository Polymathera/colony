"""User-plugin capability — drop-in skills and plugins the LLM can run.

Agents gain domain-specific tools by *discovering* them from the
filesystem: a skill is a directory with ``SKILL.md`` (YAML
frontmatter + markdown body); a plugin is a directory with
``.claude-plugin/plugin.json`` that namespaces a set of skills.

The layout deliberately overlaps with Claude Code so the same
directory can be shared between Colony and Claude Code with minimal
translation. Divergences (``sandbox_image_role``, required ``script``
field, strict param validation) are documented in
``colony_docs/markdown/plans/design_UserPluginCapability.md``.

Execution is delegated to ``SandboxedShellCapability`` — a skill never
runs in the host process, only inside a curated Docker image. That
boundary is the load-bearing security property of this capability.
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING
from overrides import override

from ...base import AgentCapability
from ...models import AgentSuspensionState
from ...scopes import BlackboardScope, get_scope_prefix
from ..actions import action_executor

from ._plugin import (
    DiscoveryResult,
    PluginSpec,
    SkillParam,
    SkillSource,
    SkillSpec,
    discover_plugins,
    discover_skills,
)
from .sandboxed_shell import SandboxedShellCapability

if TYPE_CHECKING:
    from ...base import Agent


logger = logging.getLogger(__name__)


# Defaults mirroring design §5.2.
_DEFAULT_SYSTEM_ROOT = "/etc/colony"
_DEFAULT_USER_ROOT = os.path.expanduser("~/.colony")
_DEFAULT_WORKSPACE_ROOT = "/workspace/.colony"
_SKILL_CONTAINER_MOUNT = "/skill"


# ---------------------------------------------------------------------------
# Capability
# ---------------------------------------------------------------------------

class UserPluginCapability(AgentCapability):
    """Discover and run user-supplied skills inside sandboxes.

    Actions:

    - ``list_skills`` / ``get_skill`` / ``search_skills``
    - ``list_plugins``
    - ``run_skill(name, params=...)``
    - ``reload_skills``

    The capability requires ``SandboxedShellCapability`` on the same
    agent; ``run_skill`` raises a clean error dict if it is missing.
    The LLM discovers available skills either via ``list_skills`` or
    via the dynamically built ``get_action_group_description`` (the
    description enumerates every loaded skill with a one-liner).

    Args:
        agent: Owning agent.
        scope: Blackboard partition this capability writes under.
        namespace: Capability sub-namespace.
        skill_roots: Skills-root directories in priority order
            (highest-priority first). When ``None``, computed from
            ``workspace_root`` + ``~/.colony`` + ``/etc/colony``.
        plugin_roots: Plugin-root directories in priority order.
            Same fallbacks as ``skill_roots`` but under ``plugins/``.
        workspace_root: Host path treated as the session workspace
            (mounted at ``/workspace`` in sandboxes). Defaults to
            ``/workspace/.colony`` so it matches the sandbox-side
            conventions.
        default_sandbox_image_role: Role used when a skill does not
            declare ``sandbox_image_role``. Must exist in the sandbox
            image registry or ``run_skill`` returns an error.
        sandbox_capability_key: Capability key used to resolve the
            sandbox on the agent. Default ``"sandboxed_shell"``.
        allow_model_invocation_override: If False (default), the
            capability refuses to run skills whose frontmatter has
            ``disable-model-invocation: true``. Set True only for
            test harnesses that exercise those skills.
        capability_key: Dispatcher key.
        app_name: ``serving`` app override.
    """

    def __init__(
        self,
        agent: Agent,
        scope: BlackboardScope = BlackboardScope.SESSION,
        namespace: str = "skills",
        skill_roots: list[str] | None = None,
        plugin_roots: list[str] | None = None,
        extra_skill_roots: list[str] | None = None,
        extra_plugin_roots: list[str] | None = None,
        workspace_root: str = _DEFAULT_WORKSPACE_ROOT,
        default_sandbox_image_role: str = "default",
        sandbox_capability_key: str = "sandboxed_shell",
        allow_model_invocation_override: bool = False,
        capability_key: str = "user_plugin",
        app_name: str | None = None,
    ):
        super().__init__(
            agent=agent,
            scope_id=get_scope_prefix(scope, agent, namespace=namespace),
            capability_key=capability_key,
            app_name=app_name,
        )
        self._workspace_root = workspace_root
        self._skill_roots = self._resolve_roots(
            skill_roots, kind="skills", extra=extra_skill_roots,
        )
        self._plugin_roots = self._resolve_roots(
            plugin_roots, kind="plugins", extra=extra_plugin_roots,
        )
        self._default_sandbox_image_role = default_sandbox_image_role
        self._sandbox_capability_key = sandbox_capability_key
        self._allow_model_invocation_override = allow_model_invocation_override
        self._discovery: DiscoveryResult = DiscoveryResult()
        self._load()

    def get_action_group_description(self) -> str:
        """Describe the capability and enumerate loaded skills.

        The LLM sees this text in the planning prompt, so having it
        name the available skills removes a ``list_skills`` round-trip
        in the common case.
        """
        header = (
            "User Plugins — run user-supplied skills discovered from "
            f"~/.colony/skills, {self._workspace_root}/skills, and "
            f"{_DEFAULT_SYSTEM_ROOT}/skills. Each skill executes "
            "inside a SandboxedShellCapability container. Call "
            "run_skill(name, params) to invoke one; search_skills or "
            "list_skills for discovery."
        )
        if not self._discovery.skills:
            return header + " No skills currently loaded."
        lines: list[str] = [header, ""]
        for name in sorted(self._discovery.skills.keys()):
            sk = self._discovery.skills[name]
            lines.append(
                f"- {sk.qualified_name}: "
                f"{(sk.description or sk.when_to_use)[:140]}"
            )
        return "\n".join(lines)

    def get_capability_tags(self) -> frozenset[str]:
        return frozenset({"skills", "plugins", "extensibility"})

    @override
    async def serialize_suspension_state(
        self, state: AgentSuspensionState
    ) -> AgentSuspensionState:
        # Skills live on disk; nothing to persist through suspension.
        return state

    @override
    async def deserialize_suspension_state(
        self, state: AgentSuspensionState
    ) -> None:
        return None

    # --- Internal ---------------------------------------------------------

    def _resolve_roots(
        self,
        roots: list[str] | None,
        *,
        kind: str,
        extra: list[str] | None = None,
    ) -> list[tuple[Path, SkillSource]]:
        """Turn a list of string paths (or defaults) into a priority-
        ordered list of ``(Path, SkillSource)`` tuples.

        Priority, highest-first: SESSION (workspace) → USER (~/.colony)
        → SYSTEM (/etc/colony). The caller may override by passing
        ``roots`` explicitly; we assume the explicit order matches
        priority and default all tags to SESSION. ``extra`` paths are
        appended at SYSTEM priority (so they lose to user/session
        skills with the same name) — used to ship bundled samples
        without shadowing what the user has installed locally.
        """
        if roots is not None:
            base = [
                (Path(p).expanduser(), SkillSource.SESSION) for p in roots
            ]
        else:
            base = [
                (Path(self._workspace_root) / kind, SkillSource.SESSION),
                (Path(_DEFAULT_USER_ROOT) / kind, SkillSource.USER),
                (Path(_DEFAULT_SYSTEM_ROOT) / kind, SkillSource.SYSTEM),
            ]
        for p in (extra or []):
            base.append((Path(p).expanduser(), SkillSource.SYSTEM))
        return base

    def _load(self) -> None:
        """(Re)scan all roots and rebuild ``self._discovery``."""
        result = discover_skills(self._skill_roots)
        result = discover_plugins(self._plugin_roots, into=result)
        self._discovery = result
        logger.info(
            "UserPluginCapability: loaded %d skill(s) and %d plugin(s) "
            "(errors=%d, collisions=%d)",
            len(result.skills), len(result.plugins),
            len(result.errors), len(result.collisions),
        )

    def _get_sandbox(self) -> SandboxedShellCapability | None:
        """Look up the sandbox capability on the agent by type.

        The lookup is by *type* rather than by key so alternate
        sandboxes (``ShellCapability`` subclasses for K8s/remote
        Docker) also satisfy the dependency.
        """
        if self._agent is None:
            return None
        try:
            return self.agent.get_capability_by_type(SandboxedShellCapability)
        except Exception:
            return None

    # --- Param validation -------------------------------------------------

    _TYPE_CHECKS: dict[str, type | tuple[type, ...]] = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    def _validate_params(
        self, skill: SkillSpec, args: dict[str, Any],
    ) -> str | None:
        """Return ``None`` if ``args`` satisfies ``skill.params``,
        else a human-readable error reason."""
        declared = {p.name: p for p in skill.params}
        for name, param in declared.items():
            if param.required and name not in args:
                return f"missing required param {name!r}"
        for name, value in args.items():
            param = declared.get(name)
            if param is None:
                continue  # unknown params are passed through
            expected = self._TYPE_CHECKS.get(param.type)
            if expected is None:
                continue  # unknown type — trust the caller
            if isinstance(expected, tuple):
                ok = isinstance(value, expected) and not isinstance(
                    value, bool,
                ) if int in expected else isinstance(value, expected)
            else:
                if expected is int:
                    # bool is a subclass of int; exclude it here so
                    # ``type=integer`` does not silently accept True.
                    ok = isinstance(value, int) and not isinstance(value, bool)
                else:
                    ok = isinstance(value, expected)
            if not ok:
                return (
                    f"param {name!r} expected type {param.type}, got "
                    f"{type(value).__name__}"
                )
        return None

    # --- Command rendering -----------------------------------------------

    _PLACEHOLDER = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")

    def _render_script_command(
        self, skill: SkillSpec, args: dict[str, Any],
    ) -> list[str]:
        """Build the bash command line for a skill execution.

        The skill's ``script`` path is resolved relative to
        ``_SKILL_CONTAINER_MOUNT``. Placeholders of the form
        ``{param}`` in the script path itself are substituted first
        (for scripts that include an arg in the filename); the
        substituted values are shell-quoted individually when
        appended as positional arguments.
        """
        if not skill.script:
            raise ValueError(
                f"skill {skill.qualified_name!r} has no 'script' "
                f"field; cannot execute"
            )
        # Substitute placeholders in the script path first.
        script = skill.script.format(**args) if "{" in skill.script else skill.script
        consumed = set(self._PLACEHOLDER.findall(skill.script))
        # Remaining params become positional args, quoted.
        positional: list[str] = []
        for name, value in args.items():
            if name in consumed:
                continue
            positional.append(f"--{name}")
            positional.append(str(value))
        quoted = " ".join(_shell_quote(x) for x in positional)
        cmd_str = (
            f"cd {_SKILL_CONTAINER_MOUNT} && "
            f"bash {_shell_quote(script)} {quoted}".strip()
        )
        return ["bash", "-lc", cmd_str]

    # === Action executors =================================================

    @action_executor()
    async def list_skills(
        self, *, source: str | None = None,
    ) -> dict[str, Any]:
        """List every loaded skill.

        Args:
            source: Optional filter: ``"session"`` / ``"user"`` /
                ``"system"`` / ``"plugin"``. When ``None``, all
                sources are returned. Unknown values return an error.
        """
        valid = {"session", "user", "system", "plugin", None}
        if source not in valid:
            return {
                "skills": [], "count": 0,
                "message": (
                    f"unknown source {source!r}; valid: "
                    f"{sorted(s for s in valid if s)}"
                ),
            }
        skills: list[dict[str, Any]] = []
        for sk in self._discovery.skills.values():
            if source == "plugin":
                if sk.plugin_name is None:
                    continue
            elif source is not None:
                if sk.source.value != source:
                    continue
                if sk.plugin_name is not None:
                    continue
            skills.append(sk.to_summary())
        skills.sort(key=lambda s: s["qualified_name"])
        return {
            "skills": skills, "count": len(skills), "message": "",
        }

    @action_executor()
    async def get_skill(self, name: str) -> dict[str, Any]:
        """Return full metadata + body markdown for one skill.

        Args:
            name: Qualified name (``"<plugin>/<skill>"``) or bare name.
                Bare names match only when unambiguous.
        """
        sk = self._resolve_skill(name)
        if sk is None:
            return {
                "skill": None, "found": False,
                "message": f"skill {name!r} not found",
            }
        return {"skill": sk.to_detail(), "found": True, "message": ""}

    @action_executor()
    async def search_skills(
        self, query: str, *, max_results: int = 10,
    ) -> dict[str, Any]:
        """Rank skills by simple substring match against their name,
        description, and ``when_to_use``.

        This is intentionally naive (no embeddings, no fuzzy match):
        the LLM can form its own semantic intuition; the capability
        only needs to surface obvious candidates.
        """
        q = query.lower().strip()
        if not q:
            return {"skills": [], "count": 0, "message": "empty query"}
        scored: list[tuple[int, SkillSpec]] = []
        for sk in self._discovery.skills.values():
            if sk.disable_model_invocation and not self._allow_model_invocation_override:
                continue
            haystack = " ".join((
                sk.name, sk.qualified_name, sk.description,
                sk.when_to_use,
            )).lower()
            if q in haystack:
                # Higher score when the query matches the name field.
                score = 3 if q in sk.name.lower() else (
                    2 if q in sk.description.lower() else 1
                )
                scored.append((score, sk))
        scored.sort(key=lambda t: (-t[0], t[1].qualified_name))
        hits = [s.to_summary() for _, s in scored[:max_results]]
        return {"skills": hits, "count": len(hits), "message": ""}

    @action_executor()
    async def list_plugins(self) -> dict[str, Any]:
        """List every loaded plugin (not the individual skills)."""
        plugins = [p.to_summary() for p in self._discovery.plugins.values()]
        plugins.sort(key=lambda p: p["name"])
        return {"plugins": plugins, "count": len(plugins)}

    @action_executor()
    async def reload_skills(self) -> dict[str, Any]:
        """Rescan every discovery root, replacing the cached result.

        The LLM may call this after a user tells the agent "I just
        installed a skill" — avoids requiring an agent restart.
        """
        self._load()
        return {
            "skill_count": len(self._discovery.skills),
            "plugin_count": len(self._discovery.plugins),
            "errors": [
                {"path": p, "reason": r}
                for p, r in self._discovery.errors
            ],
            "collisions": [
                {"name": n, "kept": k, "skipped": s}
                for n, k, s in self._discovery.collisions
            ],
        }

    @action_executor()
    async def run_skill(
        self,
        name: str,
        *,
        params: dict[str, Any] | None = None,
        container_id: str | None = None,
        timeout_seconds: int | None = None,
        stream_to_blackboard: bool = False,
    ) -> dict[str, Any]:
        """Execute a skill in a sandboxed container.

        If ``container_id`` is None, the capability launches a fresh
        container using the skill's declared ``sandbox_image_role``
        (falling back to ``default_sandbox_image_role``) and stops it
        after the skill exits. If ``container_id`` is provided, the
        caller is responsible for the container's lifecycle; this
        action only execs.

        Args:
            name: Qualified name or bare name.
            params: Parameter values substituted into the skill's
                script template.
            container_id: Optional existing container to run in.
            timeout_seconds: Override the skill's declared timeout.
            stream_to_blackboard: Pass-through to
                ``SandboxedShellCapability.execute_command``.
        """
        args = dict(params or {})
        sk = self._resolve_skill(name)
        if sk is None:
            return self._run_error(name, f"skill {name!r} not found")
        if sk.disable_model_invocation and not self._allow_model_invocation_override:
            return self._run_error(
                name,
                "skill is marked disable-model-invocation and cannot be "
                "invoked by the LLM",
            )

        err = self._validate_params(sk, args)
        if err is not None:
            return self._run_error(name, err)
        try:
            cmd = self._render_script_command(sk, args)
        except ValueError as e:
            return self._run_error(name, str(e))

        sandbox = self._get_sandbox()
        if sandbox is None:
            return self._run_error(
                name,
                "UserPluginCapability requires SandboxedShellCapability "
                "on the same agent",
            )

        owned_container: bool = False
        cid: str | None = container_id
        launch_message = ""
        if cid is None:
            role = sk.sandbox_image_role or self._default_sandbox_image_role
            launched = await sandbox.launch_container(
                image_role=role,
                extra_volumes=[{
                    "src": str(sk.directory),
                    "dst": _SKILL_CONTAINER_MOUNT,
                    "mode": "ro",
                }],
                max_wall_time_seconds=max(
                    sk.timeout_seconds + 60, 600,
                ),
            )
            if not launched.get("started"):
                return self._run_error(
                    name,
                    f"failed to launch sandbox: "
                    f"{launched.get('message', 'unknown error')}",
                )
            cid = launched["container_id"]
            owned_container = True
            launch_message = launched.get("message", "") or ""

        effective_timeout = (
            timeout_seconds if timeout_seconds is not None
            else sk.timeout_seconds
        )
        try:
            exec_result = await sandbox.execute_command(
                container_id=cid, command=cmd,
                timeout_seconds=effective_timeout,
                stream_to_blackboard=stream_to_blackboard,
            )
        finally:
            if owned_container and cid is not None:
                try:
                    await sandbox.stop_container(cid)
                except Exception as e:  # pragma: no cover — defensive
                    logger.debug(
                        "UserPluginCapability: stop after run failed: %s",
                        e,
                    )
        out = dict(exec_result)
        out["skill"] = sk.qualified_name
        out["owned_container"] = owned_container
        if launch_message:
            out.setdefault("launch_message", launch_message)
        return out

    # --- Helpers ----------------------------------------------------------

    def _resolve_skill(self, name: str) -> SkillSpec | None:
        if name in self._discovery.skills:
            return self._discovery.skills[name]
        # Bare-name lookup — unambiguous only.
        matches = [
            sk for sk in self._discovery.skills.values()
            if sk.name == name
        ]
        if len(matches) == 1:
            return matches[0]
        return None

    @staticmethod
    def _run_error(name: str, message: str) -> dict[str, Any]:
        return {
            "skill": name,
            "exit_code": -1,
            "stdout": "",
            "stderr": "",
            "wall_time_ms": 0,
            "truncated": False,
            "stream_key": None,
            "container_id": None,
            "owned_container": False,
            "message": message,
        }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _shell_quote(s: str) -> str:
    """Minimal shell-safe quoting for our own argv builder.

    Use single-quotes, escaping embedded singles via the classic
    ``'...'"'"'...'`` trick. Good enough for arguments that only need
    protection from word splitting and parameter expansion.
    """
    if not s:
        return "''"
    if all(c.isalnum() or c in "@%+=:,./_-" for c in s):
        return s
    escaped = s.replace("'", "'\"'\"'")
    return f"'{escaped}'"
