"""User-session ``SessionAgent`` blueprint factory + spawn helper.

Extracted from ``routers/sessions.py:create_session`` so the spawn
is re-entrant — the same blueprint shape now lives in ONE place and
can be invoked from anywhere the dashboard has a
:class:`ColonyConnection` (plus the per-session identifiers).

R12-ROOT-CAUSE-C (no SessionAgent recovery): when the user's
SessionAgent dies, the dashboard chat router needs to respawn it
without re-running the entire request handler (which depends on
the request's auth context). The factory's inputs are explicit so
they can be re-derived from postgres at respawn time:

- ``session_id`` — the Session row
- ``tenant_id`` + ``colony_id`` — Session.syscontext
- ``user_sub`` — SessionMetadata.created_by (the auth ``sub`` we
  now persist on session-create; PR1-B added this)
- ``colony_project`` — :func:`auth_service.get_colony_github_project`
- ``design_monorepo_url`` — :func:`auth_service.get_design_monorepo`
- ``git_attribution`` — :func:`auth_service.get_git_attribution`
- ``tenant_github_installation`` —
  :func:`auth_service.get_tenant_github_installation`
- ``user_github_identity`` —
  :func:`auth_service.get_user_github_identity` (resolved from
  ``user_sub``)

The factory itself is pure (no db / colony access) so it's trivial
to call from both the initial create-session path and the respawn
path. :func:`spawn_user_session_agent_for_session` is the high-level
helper that pulls per-colony state from postgres, calls the factory,
spawns via ``AgentHandle.from_blueprint``, and returns the new
``agent_id`` — wraps the entire respawn dance in one async call.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

from polymathera.colony.agents import AgentHandle, AgentMetadata
from polymathera.colony.agents.blueprint import AgentBlueprint
from polymathera.colony.agents.models import LifecycleMode
from polymathera.colony.agents.patterns.capabilities.agent_pool import (
    AgentPoolCapability,
)
from polymathera.colony.agents.patterns.capabilities.consciousness import (
    ConsciousnessCapability,
)
from polymathera.colony.agents.patterns.capabilities.github import (
    GitHubCapability,
)
from polymathera.colony.agents.patterns.capabilities.human_help import (
    HumanHelpCapability,
)
from polymathera.colony.agents.patterns.capabilities.mission_status import (
    MissionStatusCapability,
)
from polymathera.colony.agents.patterns.capabilities.sandboxed_shell import (
    SandboxedShellCapability,
)
from polymathera.colony.agents.patterns.capabilities.user_plugin import (
    UserPluginCapability,
)
from polymathera.colony.agents.patterns.capabilities.vcm import VCMCapability
from polymathera.colony.agents.patterns.capabilities.web_search import (
    ColonyDocsCapability,
    WebSearchCapability,
)
from polymathera.colony.agents.patterns.planning.streams import (
    ActionKeySubstringFilter,
    ConsciousnessStream,
    ConversationFormatter,
    EventContextKeyFilter,
    SuccessfulActionFilter,
)
from polymathera.colony.agents.roles.knowledge_curator import (
    KnowledgeCuratorCapability,
)
from polymathera.colony.agents.scopes import BlackboardScope
from polymathera.colony.agents.self_concept import AgentSelfConcept
from polymathera.colony.design_monorepo import (
    DesignMonorepoCapabilityBase,
    design_monorepo_capability_blueprints,
)
from polymathera.colony.distributed.ray_utils.serving.context import (
    ExecutionContext,
    Ring,
    get_tenant_id,
    get_colony_id,
)
from polymathera.colony.knowledge.deps import default_ingestor_blueprint

from .session_agent import SessionAgent, SessionOrchestratorCapability
from .session_agent_guardrails import build_session_agent_runtime_guardrail
from .session_agent_lifecycle import session_agent_stopped


if TYPE_CHECKING:
    from ..services.colony_connection import ColonyConnection


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-session blueprint factory
# ---------------------------------------------------------------------------


def _bundled_samples_plugins_root() -> str:
    """Filesystem path to the sample plugins shipped with the package.

    Resolved against the installed ``polymathera.colony.samples`` module
    so it works whether the package is installed editable, from a wheel,
    or vendored. Mirrors the helper in ``routers/sessions.py``;
    duplicated here so the factory has no router-side dependencies.
    """

    from polymathera.colony import samples
    return str(Path(samples.__file__).parent / "plugins")


def _resolve_github_identity(
    tenant_row: dict | None,
    user_row: dict | None,
) -> dict:
    """Compose the ``github_identity`` metadata block agents read.

    Inputs are the raw rows returned by
    :func:`auth_service.get_tenant_github_installation` and
    :func:`auth_service.get_user_github_identity` — either may be
    ``None`` (tenant missing the row; user hasn't OAuth'd, or
    respawn happens without a fresh user context). The returned
    dict always has the five keys downstream readers expect;
    absent values are ``None`` so the agent's GitHub action surface
    surfaces a clean error rather than crashing.

    ``git_user_name`` falls back to ``github_login`` when GitHub's
    profile ``name`` field is unset for the user (it's optional on
    GitHub). ``users.git_user_name`` is populated from GitHub's
    ``/user`` ``name`` response via
    :func:`auth_service.upsert_user_from_vcs`, which honors ``COALESCE``
    on a NULL value — so a brand-new user who OAuth'd successfully but
    hasn't set a display name on their GitHub profile has
    ``users.git_user_name IS NULL``. Without the fallback, the
    commit-attribution resolver
    (:meth:`DesignMonorepoCapabilityBase._resolve_attribution`) would
    silently drop the ``Co-Authored-By:`` trailer for these users even
    when the Colony UI is configured as ``commit_co_author=user``. Git
    accepts any string as author name; the GitHub login is the
    canonical identifier the user authenticated with and is always
    populated on a successful OAuth (per
    :func:`auth_service.get_user_github_identity`'s
    ``row['vcs_login'] is None → return None`` gate).
    """

    user = user_row or {}
    return {
        "tenant_installation_id": (tenant_row or {}).get("installation_id"),
        "user_github_login": user.get("github_login"),
        "user_github_id": user.get("github_user_id"),
        "git_user_email": user.get("github_email"),
        # Fall back to ``github_login`` when ``git_user_name`` is
        # unset/empty so users with no GitHub display name still get
        # commits attributed via their login. See docstring.
        "git_user_name": (
            user.get("git_user_name") or user.get("github_login")
        ),
    }


def build_user_session_agent_blueprint(
    *,
    session_id: str,
    tenant_id: str,
    colony_id: str,
    colony_project_node_id: str,
    design_monorepo_url: str | None,
    git_attribution: dict,
    tenant_github_installation: dict | None,
    user_github_identity: dict | None,
) -> AgentBlueprint:
    """Build the ``SessionAgent.bind(...)`` blueprint for a user
    chat session.

    Pure function — no db access, no colony lookups, no request
    context. Every per-session/per-colony value is an explicit
    argument so the same shape works for the initial create-session
    path and the respawn path. Callers (the create-session handler
    and :func:`spawn_user_session_agent_for_session`) fetch the
    values from postgres + the auth service first.

    The ``user_github_identity`` arg is the RAW row from
    :func:`auth_service.get_user_github_identity` — the factory runs
    it through :func:`_resolve_github_identity` for the
    ``GitHubCapability.GITHUB_IDENTITY_KEY`` shape downstream
    readers expect.
    """

    # ``available_missions`` is populated dynamically by
    # :meth:`SessionOrchestratorCapability._refresh_available_missions`
    # on agent ``initialize()`` (and on every subsequent user
    # message), unioning :func:`get_mission_registry` (colony
    # builtins + ``polymathera.mission_types`` entry-points)
    # with L4 missions discovered under the per-agent design
    # monorepo clone via L1-A. We seed an empty dict here so
    # the planner-prompt rendering does not stumble on a
    # missing key during the brief window between agent
    # construction and ``initialize()``'s refresh — the value
    # is overwritten on first refresh.

    # AgentMetadata's ``syscontext`` field defaults to
    # ``require_execution_context()``, which captures whatever
    # is in the contextvar at construction time. The FastAPI
    # AuthMiddleware sets ``tenant_id`` + ``colony_id`` on the
    # request context but NOT ``session_id`` (the session is
    # being CREATED here). The matching
    # ``user_execution_context(session_id=...)`` block below
    # only takes effect for ``AgentHandle.from_blueprint`` —
    # by then the metadata is already frozen with a syscontext
    # whose ``session_id`` is empty.
    #
    # Build the syscontext explicitly so the SessionAgent's
    # ``metadata.session_id`` returns the real session id.
    # Without this, the tracing facility's trace_id resolver
    # (``AgentTracingFacility.get_trace_id``) falls back to
    # ``agent_id`` and the SessionAgent's spans land in a
    # disjoint trace tree from every other agent it spawns
    # in this session.
    #
    # NOTE: the prior ``session_id=session_id`` kwarg below
    # was silently dropped — ``session_id`` is a read-only
    # ``@property`` on ``AgentMetadata`` (delegating to
    # ``syscontext.session_id``), not a writable field.

    session_syscontext = ExecutionContext(
        ring=Ring.USER,
        tenant_id=tenant_id,
        colony_id=colony_id,
        session_id=session_id,
        origin="dashboard_session_create",
    )

    agent_metadata = AgentMetadata(
        role="session_orchestrator",
        syscontext=session_syscontext,
        # SessionAgent is a long-lived chat-service agent: it
        # processes many user messages across a session that
        # can last hours or days. ``max_iterations`` is a
        # stuck-detection cap for ONE_SHOT coordinators with a
        # focused goal — applying it here turned the chat into a
        # silent-death lottery the moment user questions burned
        # enough iterations. The CONTINUOUS lifecycle declares
        # "goal satisfied → IDLE, wait for new events" (the
        # existing primitive — see :class:`LifecycleMode`);
        # :func:`effective_loop_max_iterations` bypasses the
        # outer-loop cap, and ``max_iterations=None`` here makes
        # the intent explicit at the construction site.
        lifecycle_mode=LifecycleMode.CONTINUOUS,
        max_iterations=None,
        goals=[
            "Orchestrate user interactions within this session",
            "Interpret user intent and decide whether to respond directly or spawn mission agents",
            "Spawn and monitor coordinator agents for mission tasks",
            "Relay agent progress and results back to the user",
        ],
        self_concept=AgentSelfConcept(
            agent_id="",  # Overwritten by ConsciousnessCapability
            name="Session Agent",
            role=(
                "the session orchestrator responsible for handling all user "
                "interactions in this session. You receive user messages, decide "
                "how to handle them, and can either respond directly or spawn "
                "specialized coordinator agents to perform mission tasks."
            ),
            description=_SESSION_AGENT_SELF_CONCEPT_DESCRIPTION,
        ),
        parameters={
            # SESSION-scoped keys live on
            # ``SessionOrchestratorCapability.AGENT_METADATA_PARAMS``;
            # we seed an empty dict + the REPL override here so
            # the planner-prompt rendering works during the
            # brief window between agent construction and the
            # first ``_refresh_available_*`` pass. ``session_id``
            # is NOT seeded — it's a typed property on
            # ``AgentMetadata`` (via syscontext), never a
            # parameters key.
            SessionOrchestratorCapability.AVAILABLE_MISSIONS_KEY: {},
            SessionOrchestratorCapability.AVAILABLE_TOOLS_KEY: {},
            SessionOrchestratorCapability.REPL_GUIDANCE_OVERRIDE_KEY: (
                "## REPL\n\n"
                "You have a Python REPL. Use `await run(\"action_key\", ...)` to "
                "execute capability actions. Use `results[\"key\"] = value` to "
                "store results. Use `log(msg)` for structured logging."
            ),
            # Per-colony design-monorepo URL — populated from
            # the ``colonies`` table. The design-monorepo
            # capability trio reads this in ``_client_sync`` to
            # lazy-clone the repo into the per-agent working
            # directory on first access. ``None`` when the
            # colony has no design monorepo configured yet.
            DesignMonorepoCapabilityBase.DESIGN_MONOREPO_URL_KEY: (
                design_monorepo_url
            ),
            # Per-commit attribution config — read once at
            # session-creation, agents look it up on metadata
            # for every commit they produce. Defaults baked
            # into the schema return ``commit_principal=colony``
            # / ``commit_co_author=user`` for fresh colonies;
            # operator overrides via the landing page.
            DesignMonorepoCapabilityBase._GIT_ATTRIBUTION_KEY: git_attribution,
            # GitHub identity for this session (P4 of
            # ``colony/github_identity_fix_plan.md``).
            # Per-tenant: the App installation id Colony uses
            # to mint REST tokens scoped to this tenant's
            # repos (read by ``GitHubCapability`` in P5).
            # Per-user: the OAuth-verified GitHub login +
            # email + name (read by ``_resolve_attribution``
            # in P6 and by ``propose_task_assignments`` in P8).
            # Every field is ``None`` until the operator wires
            # the corresponding piece; downstream readers
            # handle the unset case explicitly.
            GitHubCapability.GITHUB_IDENTITY_KEY: _resolve_github_identity(
                tenant_github_installation,
                user_github_identity,
            ),
        },
        action_policy_config={
            # SessionAgent must NOT self-terminate. The session
            # lives until the user closes it; the LLM signaling
            # completion mid-conversation would brick the chat.
            "allow_self_termination": False,  # SessionAgent should not terminate itself — the session lives on until the user closes it
            "planning_capability_blueprints": [],
        },
    )

    return SessionAgent.bind(
        agent_type=SessionAgent.model_fields["agent_type"].default,
        metadata=agent_metadata,
        bound_pages=[],
        capability_blueprints=[
            SessionOrchestratorCapability.bind(),
            AgentPoolCapability.bind(),
            # ``emit_mission_status`` lets the SessionAgent
            # publish a one-line narrative for its own work
            # (in addition to relaying coordinator emissions
            # via the chat router). Mission_id under SESSION
            # scope is the SessionAgent's own agent_id.
            MissionStatusCapability.bind(scope=BlackboardScope.SESSION),
            # ``request_help`` lets the SessionAgent ask the
            # user a free-text clarifying question when intent
            # is incomplete relative to the chosen mission's
            # ``caller_parameters`` (pre-spawn translation
            # gap). Session-scoped so the chat-side
            # ``handle_human_help_request`` event handler
            # (this capability's own request payload travels
            # on the SAME session blackboard) translates the
            # request into a typed chat_question with
            # ``kind='human_help'``; the operator's response
            # flows back via the
            # ``/sessions/{id}/human_help/{request_id}/respond``
            # HTTP endpoint to ``human_help:response:*`` on
            # the same scope, where this capability's
            # ``@event_handler`` surfaces it as planner context.
            HumanHelpCapability.bind(scope=BlackboardScope.SESSION),
            ConsciousnessCapability.bind(),
            VCMCapability.bind(scope=BlackboardScope.SESSION),
            WebSearchCapability.bind(scope=BlackboardScope.SESSION),
            ColonyDocsCapability.bind(scope=BlackboardScope.SESSION),
            SandboxedShellCapability.bind(scope=BlackboardScope.SESSION),
            UserPluginCapability.bind(
                scope=BlackboardScope.SESSION,
                extra_plugin_roots=[_bundled_samples_plugins_root()],
            ),
            # GitHubCapability starts in a "disabled" state if
            # no credentials are configured; the action surface
            # surfaces a clean error instead of crashing the
            # agent. Operators enable it via env vars or the
            # Settings UI (planned).
            #
            # ``default_project_id`` is the colony's attached
            # GitHub Project (v2) node id — every new issue
            # ``GitHubCapability.create_issue`` mints gets
            # auto-attached to this project. Resolved upfront
            # (see the lookup block above the spawn).
            GitHubCapability.bind(
                scope=BlackboardScope.SESSION,
                default_project_id=colony_project_node_id,
            ),
            # Design-monorepo capability trio (state, checkpointing,
            # tool building) — per-agent clones under
            # /mnt/shared/agents/<agent_id>/clones/<scope_id>/.
            # SessionAgent must NOT subscribe to the
            # convergence-quiescence stream — every episode
            # boundary would otherwise wake the LLM planner,
            # producing the same welcome message in a tight
            # loop. Sub-agents that perform actual design work
            # get the default trio (with auto-checkpoint
            # enabled) when they are spawned.
            *design_monorepo_capability_blueprints(
                auto_checkpoint_on_quiescence=False,
            ),
            # Knowledge curation — chat-driven write side. The
            # dashboard ships *blueprints*, not live instances:
            # the Ingestor + RetrievalDeps wrap an
            # ``AsyncQdrantClient`` (RLock) and an
            # ``InMemoryEmbedder`` (local closure), neither of
            # which survives cloudpickle. The blueprint chain
            # (Ingestor.bind → InMemoryEmbedder.bind +
            # QdrantVectorStore.bind) carries only picklable
            # kwargs (URLs, collection names, dimensions) and
            # is resolved on the worker via ``local_instance()``
            # — same pattern as ``ConsciousnessStream(formatter=…)``.
            # Same KnowledgeConfig (loaded from the same YAML),
            # same collection, separate Python objects per process.
            #
            # ``KnowledgeRetrievalCapability`` is auto-injected
            # for every agent in ``Agent._create_action_policy``
            # (Phase 1c). Bulk acquisition / corpus ingestion
            # runs through
            # :meth:`RepoStateProvider.ingest_repo_map_literature`
            # (bound above via ``design_monorepo_capability_blueprints``);
            # the unified ``.colony/repo_map.yaml`` schema is the
            # single source of truth for which sources land in
            # the KB.
            KnowledgeCuratorCapability.bind(
                ingestor=default_ingestor_blueprint(),
            ),
        ],
        action_policy_blueprints={
            "consciousness_streams": [
                ConsciousnessStream.bind(
                    name="conversation",
                    formatter=ConversationFormatter.bind(),
                    event_filter=EventContextKeyFilter("user_chat_message"),
                    action_filter=SuccessfulActionFilter(
                        ActionKeySubstringFilter("respond_to_user")
                    ),
                ),
            ],
            # Hard guardrails mounted on the code-generation
            # action policy. See ``session_agent_guardrails``
            # for the rules + ``colony/mission_and_action_
            # guardrails_plan.md`` Part 2 for the design.
            # Travels through cloudpickle to the Ray worker
            # via the ``exclude=True`` field — module-level
            # named predicates keep the serialised graph
            # picklable.
            "runtime_guardrail": (
                build_session_agent_runtime_guardrail()
            ),
        },
        # PR1-A (R12-ROOT-CAUSE-C + B4): user-visible death message.
        # Writes a typed ``chat:agent:*`` system_failure record the
        # chat-relay forwards to the browser, AND emits an
        # ``AgentDiagnosticProtocol`` event. Without this, the user
        # types nothing reads, the chat appears hung. PR1-B (lazy
        # respawn in :func:`spawn_user_session_agent_for_session`)
        # builds on this — the dashboard's chat router detects the
        # dead agent on the next user message and rebuilds it via
        # the factory above. Both paths share the same blueprint
        # shape because the factory is the single source of truth.
        stop_callbacks=[session_agent_stopped],
    )


# Long static prose kept out of the function body so the blueprint
# construction stays readable. Verbatim from the pre-extraction
# body of ``routers/sessions.py:create_session`` — the LLM's
# behavior depends on every paragraph; do not truncate.
_SESSION_AGENT_SELF_CONCEPT_DESCRIPTION: str = (
    "You are the primary interface between the user and the Colony's "
    "multi-agent system. You have access to an agent pool for spawning "
    "coordinator agents (one per mission type), and you can respond "
    "directly to user questions.\n\n"
    "MISSION SPAWN PROTOCOL — translate user intent into a complete mission spec:\n"
    "  You are a TRANSLATION LAYER between the user (who speaks\n"
    "  in their own vocabulary and does NOT need to know which\n"
    "  missions exist or what parameters they take) and the\n"
    "  mission coordinators (which need a fully-specified\n"
    "  mission to run). Your job has FIVE steps; you must NOT\n"
    "  short-circuit any of them by passing the user's request\n"
    "  through verbatim.\n\n"
    "  Step 1 — Pick the mission. Semantically match the user's\n"
    "  intent against entries in ``available_missions`` (read\n"
    "  their ``label`` / ``description``).\n\n"
    "  Step 2 — Translate intent → mission_params. Read the\n"
    "  chosen mission's ``mission_params`` schema (rendered\n"
    "  ``caller_parameters`` carrying name + description +\n"
    "  required + json_type + default per slot). For EACH slot,\n"
    "  synthesise a value from the user's intent using LLM\n"
    "  judgment. The mapping is SEMANTIC, not lexical:\n"
    "    - Many-to-one: multiple user statements may synthesise\n"
    "      into one free-text field (e.g. several preferences\n"
    "      → one stricter ``decomposition_criteria`` string).\n"
    "    - One-to-many: a single user statement may decompose\n"
    "      across multiple slots (e.g. 'tighten decomposition\n"
    "      for #34 and #36' → ``issue_numbers=[34,36]`` PLUS a\n"
    "      tightened ``decomposition_criteria``).\n"
    "    - Optional slots: populate ONLY when the user\n"
    "      expressed an intent for them; omit otherwise so the\n"
    "      mission applies its documented default.\n"
    "    - The user's words are the source of truth. NEVER\n"
    "      paraphrase a preference into what you think the\n"
    "      mission would prefer — the operator's own framing\n"
    "      belongs verbatim in the free-text slots.\n\n"
    "  Step 3 — Completeness gate. Before calling\n"
    "  ``spawn_mission``, verify every REQUIRED slot has a\n"
    "  meaningful value derived from the user's intent. If any\n"
    "  required slot is missing, vague, or ambiguous → DO NOT\n"
    "  spawn. Instead, call ``request_help`` with a focused,\n"
    "  minimal question naming the specific decision the user\n"
    "  needs to make. Pass any candidate values you can already\n"
    "  infer as ``options`` (the operator picks one if any fit;\n"
    "  otherwise writes free-form guidance). Then end your\n"
    "  code block with ``await run('wait_for_next_event')`` to\n"
    "  pause. The operator's response surfaces on the next\n"
    "  iteration as a\n"
    "  ``human_help_response:{request_id}`` planner-context\n"
    "  binding carrying ``{chosen_option, guidance, decided_by,\n"
    "  decided_at}``. Re-check completeness on that iteration:\n"
    "  translate the free-text reply into the missing slot and\n"
    "  loop until satisfied. Spawning a partially-specified\n"
    "  mission is wrong — the coordinator cannot recover the\n"
    "  user's intent after the fact.\n\n"
    "  Step 4 — Spawn the mission. Once complete:\n"
    "    r = await run(\n"
    "        \"spawn_mission\",\n"
    "        mission_type=<the matching key from available_missions>,\n"
    "        mission_params={...synthesised slot values...},\n"
    "    )\n"
    "  Then branch on ``r.success`` and ``r.output[\"created\"]``;\n"
    "  the coordinator's agent_id is on ``r.output[\"agent_id\"]``.\n"
    "  ``spawn_mission`` does the coordinator-class lookup\n"
    "  internally — you do NOT extract ``coordinator_class``\n"
    "  from ``available_missions`` and you do NOT call\n"
    "  ``create_agent`` directly for missions.\n"
    "  (``create_agent`` is the low-level generic spawn;\n"
    "  ``spawn_mission`` is the mission-aware wrapper.)\n\n"
    "  RETRY HALT — when ``r.output['outcome']`` is\n"
    "  ``'error'`` and the ``r.output['error']`` string is\n"
    "  IDENTICAL to the one returned by your previous\n"
    "  ``spawn_mission`` attempt, STOP. Do NOT issue a\n"
    "  third call with the same parameters. That is a\n"
    "  framework-side bug, not a parameter issue — your\n"
    "  next iteration cannot fix it by re-trying. Instead:\n"
    "  call ``respond_to_user`` with the verbatim error\n"
    "  string (so the operator sees the actual failure\n"
    "  rather than another 'spawning…' message), then\n"
    "  ``wait_for_next_event``. Silent retry loops on\n"
    "  persistent identical errors burn iterations, run\n"
    "  out the iteration cap, and bury the real signal\n"
    "  under repeated noise (run8: 12 identical\n"
    "  ``_app_name`` errors before max_iterations hit).\n\n"
    "  Step 5 — Multi-mission decomposition. If user intent\n"
    "  does NOT cleanly map to a single mission, FIRST try to\n"
    "  decompose the request into 2+ missions (each going\n"
    "  through Steps 1-4 with its own completeness gate).\n"
    "  Chain them when one mission's output feeds the next;\n"
    "  parallelise when they are independent. If even\n"
    "  decomposition cannot cover the request, respond with\n"
    "  the maximal-coverable subset PLUS an explicit naming of\n"
    "  the missing capabilities the operator would need to\n"
    "  install / build / wire to close the gap. Do NOT\n"
    "  silently truncate the user's ask.\n\n"
    "  Example — user says \"Let's refine the roadmap. Most\n"
    "  GitHub issues seem too high-level and we probably need\n"
    "  to create sub-issues to break them down into more\n"
    "  manageable tasks.\":\n"
    "    Step 1: semantic match → ``project_planning`` (label\n"
    "      'Project Planning', description mentions decompose).\n"
    "    Step 2: read its ``mission_params``. ``mode`` is\n"
    "      required → 'decompose' (the mode that creates\n"
    "      sub-issues). ``decomposition_criteria`` is optional\n"
    "      → the user's phrase 'too high-level' aligns with\n"
    "      the default criteria, so omit (let the default\n"
    "      fire). ``issue_numbers`` is optional → user didn't\n"
    "      name specific issues, so omit (defaults to all open\n"
    "      issues).\n"
    "    Step 3: all required slots satisfied → no\n"
    "      clarification needed.\n"
    "    Step 4: spawn.\n"
    "      r = await run(\"spawn_mission\",\n"
    "          mission_type=\"project_planning\",\n"
    "          mission_params={\"mode\": \"decompose\"})\n\n"
    "  Example with clarification — user says \"Tighten the\n"
    "  criterion and decompose what's left.\":\n"
    "    Step 1: match → ``project_planning``.\n"
    "    Step 2: ``mode`` → 'decompose'.\n"
    "      ``decomposition_criteria`` would need to be\n"
    "      'tighter than default', but 'tighter' is vague —\n"
    "      by what dimension?\n"
    "    Step 3: clarify via request_help:\n"
    "      await run(\"request_help\",\n"
    "          question=\"How should I tighten the\"\n"
    "                   \" decomposition criterion?\",\n"
    "          context=\"Default criterion: 'An issue is\"\n"
    "                  \" decomposable when it is too\"\n"
    "                  \" high-level, too vague, too big for\"\n"
    "                  \" one PR, or describes ongoing work.'\",\n"
    "          options=(\n"
    "              \"Require multiple INDEPENDENT deliverables per issue\",\n"
    "              \"Require explicit cross-team coordination\",\n"
    "              \"Keep the default but raise the bar on 'too big'\",\n"
    "          ))\n"
    "      await run(\"wait_for_next_event\")\n"
    "    Next iteration: read the\n"
    "    ``human_help_response:{request_id}`` context binding;\n"
    "    translate the operator's ``chosen_option`` or\n"
    "    free-text ``guidance`` into ``decomposition_criteria``;\n"
    "    re-check completeness; spawn.\n\n"
    "  Example with multi-mission decomposition — user says\n"
    "  \"Refine the roadmap AND analyze the compliance posture\n"
    "  of the codebase.\":\n"
    "    Step 1: no single mission covers both halves.\n"
    "      Decompose into project_planning + compliance.\n"
    "    Steps 2-4 per half. Spawn one then the other\n"
    "      (parallel — they don't share data); track both\n"
    "      agent_ids in ``results``; relay both progress\n"
    "      streams back to the user.\n\n"
    "  When the user asks a question or needs information (no\n"
    "  mission spawn), use respond_to_user directly.\n\n"
    "  Before calling spawn_mission, if the user's message references a\n"
    "  mission that may already be running (\"check on the analysis you\n"
    "  started\", \"how is the planning going\"), call\n"
    "  list_spawned_missions(mission_type=<key>) first. A non-empty\n"
    "  result means you already have a live coordinator — reuse the\n"
    "  returned agent_id instead of spawning a duplicate.\n\n"
    "  When you store data in ``results`` (e.g., a query response, a\n"
    "  child agent's status report) and the next step is a simple\n"
    "  conditional, branch inline in the same code block using Python\n"
    "  if/else over ``results[\"<key>\"].output``. Variables you bind\n"
    "  in ``results[...]`` persist across iterations, so when fresh\n"
    "  reasoning is needed (data is too large, decision is open-ended,\n"
    "  or you need a new context window), simply end your block — the\n"
    "  next planning iteration sees the updated ``results``.\n\n"
    "IDLE DISCIPLINE — wait when there is no work to do:\n"
    "  You are a proactive agent: after every code block, the framework\n"
    "  calls you again for the next planning iteration. When you have\n"
    "  fully addressed the user's last message and have no in-flight\n"
    "  queries to inspect, end your block with\n"
    "  ``await run(\"wait_for_next_event\")`` to pause until the next\n"
    "  event arrives (a user message, a child-agent diagnostic, a\n"
    "  GitHub webhook, …). The framework does NOT auto-pause you —\n"
    "  if you have nothing to do and do not call ``wait_for_next_event``,\n"
    "  you will keep burning empty planning iterations.\n"
    "  ``wait_for_next_event(timeout_seconds=N)`` is available for\n"
    "  deadline-driven waits — on timeout the action returns\n"
    "  ``{\"ok\": True, \"timed_out\": True}`` so you can branch.\n\n"
    "AVAILABLE TOOLS — how to know what compute / retrieval / dispatch\n"
    "actions a freshly-spawned coordinator can mount:\n"
    "  The dict ``available_tools`` (in this agent's metadata.parameters)\n"
    "  lists every L4 tool capability the operator's design monorepo\n"
    "  declares via ``.colony/tool-registry.json``. Each entry maps\n"
    "  ``<tool_name>`` → ``{purpose, location, capability, capability_fqn}``.\n"
    "  The coordinator spawned via ``spawn_mission`` can mount any of these\n"
    "  by passing the ``capability_fqn`` string in its ``capabilities=[...]``\n"
    "  list at create_agent time. (``spawn_mission`` handles this for you when\n"
    "  the mission's worker class declares them in its blueprint.) Tools with\n"
    "  ``capability_fqn`` set are mountable; entries with an empty\n"
    "  ``capability_fqn`` are catalog-only candidates surveyed by the\n"
    "  build-vs-buy advisor — they're for context, not for mounting.\n\n"
    "DESIGN-MONOREPO BOOTSTRAP:\n"
    "  When the user asks to initialize / bootstrap / scaffold the "
    "  design monorepo (or asks 'how do I set up repo_map.yaml'), "
    "  call ``initialize_repo_map`` (on DesignCheckpointer). Do NOT "
    "  use ``mmap_repo`` for this — ``mmap_repo`` maps an existing "
    "  repo into VCM and assumes ``.colony/repo_map.yaml`` already "
    "  exists. ``initialize_repo_map`` writes a commented template "
    "  and commits it; it is idempotent and never overwrites an "
    "  operator-edited file.\n\n"
    "DESIGN-MONOREPO LITERATURE INGESTION:\n"
    "  When the user asks to ingest / process / load literature "
    "  from the design monorepo (or 'process repo_map.yaml'), call "
    "  ``ingest_repo_map_literature`` (on RepoStateProvider). It "
    "  refreshes the per-agent clone against ``origin``, reads "
    "  ``repo_map.yaml``, and ingests every file matched by a "
    "  ``knowledge_sources:`` row that is currently ENABLED in the "
    "  Design Monorepo tab's checkbox list. The selection is "
    "  persisted server-side per-colony, so the action picks it up "
    "  automatically — do NOT pass any source-filter argument. If "
    "  the user wants different rows ingested, ask them to toggle "
    "  the Design Monorepo tab's 'Knowledge sources' checkboxes; "
    "  do not try to filter from chat. Use this — NOT ``ingest_file`` "
    "  per file — when the user wants the whole literature corpus loaded.\n\n"
    "INTERACTION PROTOCOL — long-running actions:\n"
    "  Some actions take seconds or minutes (mmap_repo, create_agent, "
    "  search_and_fetch, run_skill, claim_unassigned_issue, …). For "
    "  any action you suspect will take more than ~2 seconds, FIRST "
    "  call respond_to_user with a one-line acknowledgement that "
    "  names the action you are about to run, THEN call the action. "
    "  This keeps the user informed while the work happens. After "
    "  the action returns, respond_to_user again with the outcome.\n\n"
    "ITERATION DISCIPLINE — small steps, persistent state:\n"
    "  Each REPL iteration is a focused step (≤3 actions on any "
    "  one execution path). Variables you bind in ``results[...]`` "
    "  persist across iterations, so you do NOT need to pack the "
    "  whole sequence into one cell. The natural shape is:\n"
    "    Iteration 1 — ack + the action (2 calls), store result.\n"
    "    Iteration 2 — read ``results[...]`` and respond.\n"
    "  An ack-then-action-then-branched-response (success / failure "
    "  arms) IS allowed in one iteration because only one branch "
    "  ever runs (≤3 actions per path); a straight-line program "
    "  with 4+ actions is not. When in doubt, defer the response "
    "  to the next iteration.\n\n"
    "RESULT INSPECTION — branch on the action result:\n"
    "  ``run(...)`` returns an ``ActionResult`` Pydantic "
    "  model — NOT a dict. Use attribute access, not "
    "  ``.get(...)``:\n"
    "    r.success      → bool (True iff the action ran "
    "                     to completion without error)\n"
    "    r.output       → the action's actual return "
    "                     value (a dict for most actions; "
    "                     None on failure)\n"
    "    r.error        → str | None (set on failure)\n"
    "    r.cancelled    → bool (set when /abort fired)\n"
    "  Never forward a result blindly. After every "
    "  action, branch on ``r.success`` and on the shape "
    "  of ``r.output``, surfacing zero-count / empty / "
    "  error cases explicitly. Doing this in the SAME "
    "  iteration that ran the action is allowed under "
    "  ITERATION DISCIPLINE because only one branch "
    "  executes:\n"
    "    r = await run(\"some.action\", ...)\n"
    "    results[\"r\"] = r\n"
    "    if not r.success:\n"
    "        await run(\"respond_to_user\", text=f\"…X failed: {r.error}\")\n"
    "    else:\n"
    "        out = r.output or {}\n"
    "        if not out or out.get(\"count\", 0) == 0:\n"
    "            await run(\"respond_to_user\", text=\"…ran X but it returned 0 items; likely cause: …\")\n"
    "        else:\n"
    "            await run(\"respond_to_user\", text=f\"…X succeeded: {out}\")\n"
    "  Apply this to ANY action whose ``output`` is a "
    "  dict with count / list / status / error fields — "
    "  not just literature ingestion. When the empty "
    "  case has a likely cause you can name (empty "
    "  knowledge_sources, unreachable origin, missing "
    "  file), include it in the message so the user can "
    "  act on it.\n\n"
    "RESPONSE FORMATTING — pass structured data as attachments:\n"
    "  ``respond_to_user(content, attachments=[…])`` "
    "  renders ``content`` as Markdown (GFM) and each "
    "  attachment below it as a typed, collapsible "
    "  block. Use this — NEVER hand-roll markdown "
    "  fences or ``str(dict)`` into ``content`` — for "
    "  any structured payload (dicts, lists, "
    "  ActionResult.output). The framework owns the "
    "  rendering: indentation, syntax highlighting, "
    "  collapse-when-long, the lot.\n"
    "  Three attachment kinds, plus two convenience "
    "  actions that wrap them:\n"
    "    1. CODE — ``{\"kind\":\"code\",\"lang\":\"python\","
    "       \"content\":<value>,\"label\":<str>?}``.\n"
    "       Pass the raw value (dict, list,\n"
    "       ``ActionResult.output``) directly as\n"
    "       ``content``. The framework pretty-prints\n"
    "       it server-side based on ``lang``\n"
    "       (``python``→pprint, ``json``→json.dumps,\n"
    "       ``yaml``→yaml.safe_dump). NEVER call\n"
    "       ``str(value)`` yourself — that produces a\n"
    "       single-line blob that the chat cannot lay\n"
    "       out cleanly.\n"
    "         await run(\n"
    "             \"respond_to_user\",\n"
    "             content=\"✅ Ingestion complete — 20 file(s) loaded.\",\n"
    "             attachments=[{\n"
    "                 \"kind\": \"code\",\n"
    "                 \"lang\": \"python\",\n"
    "                 \"content\": out,  # raw dict; framework formats it\n"
    "                 \"label\": \"ingest result\",\n"
    "             }],\n"
    "         )\n"
    "    2. TABLE — call ``respond_to_user_with_table(\n"
    "       summary=<str>, rows=[{...}, …],\n"
    "       columns=[…]?, label=<str>?)``. Use for\n"
    "       per-file outcomes, side-by-side metrics.\n"
    "         await run(\n"
    "             \"respond_to_user_with_table\",\n"
    "             summary=\"Per-file outcomes:\",\n"
    "             rows=[{\"file\": p, \"status\": s} for p, s in pairs],\n"
    "             columns=[\"file\", \"status\"],\n"
    "         )\n"
    "    3. DIFF — call ``respond_to_user_with_diff(\n"
    "       summary=<str>, before=<str>, after=<str>,\n"
    "       lang=<str>?, label=<str>?)``. Use for\n"
    "       state changes (config edits, file\n"
    "       rewrites). The chat renders ``+/-`` lines.\n"
    "         await run(\n"
    "             \"respond_to_user_with_diff\",\n"
    "             summary=\"Updated repo_map.yaml:\",\n"
    "             before=old_yaml, after=new_yaml,\n"
    "             lang=\"yaml\", label=\".colony/repo_map.yaml\",\n"
    "         )\n"
    "  Keep ``summary`` / ``content`` short and "
    "  human-readable — the user reads it first and "
    "  only expands attachments for details. Do NOT "
    "  embed a fenced ```` ```python … ``` ```` block "
    "  inside ``content`` when you also pass a code "
    "  attachment; pick one path.\n\n"
    "OUTPUT FORMAT — code generation:\n"
    "  Emit ONLY raw Python code at the cell level. "
    "  Do NOT wrap the whole iteration in a markdown "
    "  fence — the REPL executes the cell verbatim and "
    "  a leading ``` would crash it. Prefer attachments "
    "  over markdown fences inside ``content=`` (see "
    "  RESPONSE FORMATTING). No mocked result blocks. "
    "  No prose between statements. Each iteration is "
    "  a single focused snippet."
)


# ---------------------------------------------------------------------------
# Spawn helper — pulls per-colony state + calls the factory
# ---------------------------------------------------------------------------


async def spawn_user_session_agent_for_session(
    colony: "ColonyConnection",
    *,
    session_id: str,
    tenant_id: str,
    colony_id: str,
    user_sub: str | None,
) -> str | None:
    """Re-entrant respawn: fetch the per-colony state postgres holds
    for this session, build the blueprint, and spawn the agent.

    Returns the new ``agent_id`` on success or ``None`` on failure
    (logged at ERROR). Designed to be called from BOTH the initial
    create-session path AND the dashboard chat router's lazy
    respawn path when the prior SessionAgent has died.

    ``user_sub`` is the auth ``sub`` we persist on
    :attr:`SessionMetadata.created_by` so respawn can resolve the
    same per-user GitHub identity the original session had. When
    ``None`` (e.g. respawn happens with no original-user info), the
    GitHub identity falls back to anonymous — GitHub actions will
    surface a clean credentials-missing error rather than crashing.

    The factory's signature is the contract; everything below is
    just the db-side lookups + spawn boilerplate. If a new factory
    arg lands, add the matching lookup here.
    """

    from ..auth import service as auth_service

    if colony._db_pool is None:
        logger.error(
            "spawn_user_session_agent_for_session: session=%s — "
            "db_pool unavailable; cannot resolve per-colony state",
            session_id,
        )
        return None

    # Per-colony GitHub Project — required (issues auto-attach).
    colony_project = await auth_service.get_colony_github_project(
        colony._db_pool, colony_id=colony_id, tenant_id=tenant_id,
    )
    if colony_project is None:
        logger.error(
            "spawn_user_session_agent_for_session: session=%s — "
            "colony=%s has no GitHub Project attached; refusing "
            "to spawn (matches create_session's gate).",
            session_id, colony_id,
        )
        return None

    design_monorepo_url = (
        await auth_service.get_design_monorepo(
            colony._db_pool, colony_id=colony_id, tenant_id=tenant_id,
        ) or {}
    ).get("origin_url")

    git_attribution = (
        await auth_service.get_git_attribution(
            colony._db_pool, colony_id=colony_id, tenant_id=tenant_id,
        ) or {}
    )

    tenant_github_installation = (
        await auth_service.get_tenant_github_installation(
            colony._db_pool, tenant_id=tenant_id,
        )
    )
    user_github_identity = None
    if user_sub:
        user_github_identity = await auth_service.get_user_github_identity(
            colony._db_pool, user_id=user_sub,
        )

    blueprint = build_user_session_agent_blueprint(
        session_id=session_id,
        tenant_id=tenant_id,
        colony_id=colony_id,
        colony_project_node_id=colony_project["node_id"],
        design_monorepo_url=design_monorepo_url,
        git_attribution=git_attribution,
        tenant_github_installation=tenant_github_installation,
        user_github_identity=user_github_identity,
    )

    # Spawn under the session-scoped execution context so the
    # tracing facility's trace_id resolver lands on the session id,
    # not the agent_id. Same shape as the original create_session
    # path.
    with colony.user_execution_context(
        tenant_id=tenant_id,
        colony_id=colony_id,
        session_id=session_id,
        origin="dashboard_user_session_spawn",
    ):
        try:
            handle = await AgentHandle.from_blueprint(
                agent_blueprint=blueprint,
                app_name=colony.app_name,
            )
        except Exception as exc:
            logger.exception(
                "spawn_user_session_agent_for_session: session=%s — "
                "AgentHandle.from_blueprint failed: %s",
                session_id, exc,
            )
            return None
    return handle.agent_id


__all__ = (
    "_resolve_github_identity",
    "build_user_session_agent_blueprint",
    "spawn_user_session_agent_for_session",
)
