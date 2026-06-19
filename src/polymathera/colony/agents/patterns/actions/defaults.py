
from ...base import Agent, ActionPolicy

_DEFAULT_POLICY = "CODE_GEN" # Options: "CODE_GEN", "CACHE_AWARE", "MINIMAL"

async def create_default_action_policy(agent: Agent, **kwargs) -> ActionPolicy:
    if _DEFAULT_POLICY == "CODE_GEN":

        from .code_generation import create_code_generation_action_policy
        from .code_constraints import (
            FreeFormCodeGenerator,
            APIKnowledgeBaseValidator,
            ImportWhitelistValidator,
            InMemorySkillLibrary,
            DeterministicRecovery,
            IterationShapeValidator,
            NoGuardrail,
        )
        from ..planning.context import PlanningContextBuilder

        # Every dimension of the constrained code-gen design space is
        # overridable via the kwargs bag (which itself flows from the
        # agent's ``action_policy_blueprints`` dict — already
        # cloudpickle-safe and resolved through
        # ``Blueprint.local_instance``). Previously these were
        # hardcoded and silently ignored any overrides — a class-
        # level footgun that broke ``runtime_guardrail`` mounting in
        # particular (see mission-and-action-guardrails plan, step
        # 5). ``kwargs.get(..., factory())`` lets callers override
        # any one dimension without restating the rest, and
        # consistently keeps the previous default for unset
        # dimensions.
        code_generator = kwargs.get(
            "code_generator", FreeFormCodeGenerator(),
        )
        code_validators = kwargs.get(
            "code_validators",
            [
                APIKnowledgeBaseValidator(agent),
                ImportWhitelistValidator(),
                IterationShapeValidator(),
            ],
        )
        skill_library = kwargs.get(
            "skill_library", InMemorySkillLibrary(),
        )
        recovery_strategy = kwargs.get(
            "recovery_strategy", DeterministicRecovery(),
        )
        runtime_guardrail = kwargs.get(
            "runtime_guardrail", NoGuardrail(),
        )
        return await create_code_generation_action_policy(
            agent=agent,
            action_map=kwargs.get("action_map", None),
            action_providers=kwargs.get("action_providers", []),
            io=kwargs.get("io"),
            context_builder=PlanningContextBuilder(agent),
            code_generator=code_generator,
            code_validators=code_validators,
            skill_library=skill_library,
            recovery_strategy=recovery_strategy,
            runtime_guardrail=runtime_guardrail,
            completion_validator=kwargs.get("completion_validator", None),
            max_retries=kwargs.get("max_retries", 2),
            code_timeout=kwargs.get("code_timeout", 30.0),
            max_code_iterations=kwargs.get("max_code_iterations", 50),
            allow_self_termination=kwargs.get("allow_self_termination", True),
            planning_capability_blueprints=kwargs.get("planning_capability_blueprints", None),
            consciousness_streams=kwargs.get("consciousness_streams", None),
        )

    elif _DEFAULT_POLICY == "CACHE_AWARE":

        from .policies import create_cache_aware_action_policy

        return await create_cache_aware_action_policy(
            agent=agent,
            action_map=kwargs.get("action_map", None),
            action_providers=kwargs.get("action_providers", []),
            io=kwargs.get("io"),
            max_iterations=kwargs.get("max_iterations", 50),
            quality_threshold=kwargs.get("quality_threshold", 0.9),
            planning_horizon=kwargs.get("planning_horizon", 5),
            ideal_cache_size=kwargs.get("ideal_cache_size", 10),
            consciousness_streams=kwargs.get("consciousness_streams", None),
        )
    elif _DEFAULT_POLICY == "MINIMAL":

        from .minimal import create_minimal_action_policy

        return await create_minimal_action_policy(
            agent=agent,
            max_iterations=kwargs.get("max_iterations", 50),
            temperature=kwargs.get("temperature", 0.3),
            max_tokens=kwargs.get("max_tokens", 512),
            consciousness_streams=kwargs.get("consciousness_streams", None),
        )
    else:
        raise ValueError(f"Invalid default policy: {_DEFAULT_POLICY}")
