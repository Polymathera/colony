
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

        return await create_code_generation_action_policy(
            agent=agent,
            action_map=kwargs.get("action_map", None),
            action_providers=kwargs.get("action_providers", []),
            io=kwargs.get("io"),
            context_builder=PlanningContextBuilder(agent),
            code_generator=FreeFormCodeGenerator(),
            code_validators=[APIKnowledgeBaseValidator(agent), ImportWhitelistValidator(), IterationShapeValidator()],
            skill_library=InMemorySkillLibrary(),
            recovery_strategy=DeterministicRecovery(),
            runtime_guardrail=NoGuardrail(),
            max_retries=kwargs.get("max_retries", 2),
            code_timeout=kwargs.get("code_timeout", 30.0),
            max_code_iterations=kwargs.get("max_code_iterations", 50),
            allow_self_termination=kwargs.get("allow_self_termination", True),
            reactive_only=kwargs.get("reactive_only", False),
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
        )
    elif _DEFAULT_POLICY == "MINIMAL":

        from .policies import create_minimal_action_policy

        return await create_minimal_action_policy(
            agent=agent,
            max_iterations=kwargs.get("max_iterations", 50),
            temperature=kwargs.get("temperature", 0.3),
            max_tokens=kwargs.get("max_tokens", 512),
        )
    else:
        raise ValueError(f"Invalid default policy: {_DEFAULT_POLICY}")
