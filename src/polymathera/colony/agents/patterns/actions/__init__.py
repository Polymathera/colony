"""Domain-agnostic patterns for distributed agent action policies."""

from .dispatcher import (
    ActionExecutor,
    MethodWrapperActionExecutor,
    ActionDispatcher,
    SchemaDetail,
    action_executor,
)
from .policies import (
    CacheAwareActionPolicy,
    EventDrivenActionPolicy,
)
from .defaults import create_default_action_policy

from .repl import (
    REPLVariable,
    PolicyPythonREPL,
    REPLCapability,
    get_repl_guidance,
)

from .backing_store import (
    BackingStore,
    BlackboardBackingStore,
    StorageHint,
)

from .code_generation import (
    CodeGenerationActionPolicy,
    create_code_generation_action_policy,
)

from .code_constraints import (
    CodeGenerator,
    FreeFormCodeGenerator,
    CodeValidator,
    NoOpValidator,
    ImportWhitelistValidator,
    APIKnowledgeBaseValidator,
    IterationShapeValidator,
    SkillLibrary,
    NoOpSkillLibrary,
    InMemorySkillLibrary,
    RecoveryStrategy,
    NoRecovery,
    DeterministicRecovery,
    RuntimeGuardrail,
    NoGuardrail,
    CapabilityBoundaryGuardrail,
    TemporalOrderGuardrail,
    ArgsAwareOrderingRule,
    ArgsAwareTemporalOrderGuardrail,
    ApprovalRequiredGuardrail,
    CompositeGuardrail,
    ValidationResult,
    CodeSkill,
    RecoveryResult,
    GuardrailDecision,
)

from .minimal import (
    MinimalActionPolicy,
    create_minimal_action_policy,
)

from .deferred import (
    DeferredClosure,
    deferred,
    eager_execution,
    is_eager_execution,
)

__all__ = [
    # Policies
    "ActionExecutor",
    "MethodWrapperActionExecutor",
    "ActionDispatcher",
    "SchemaDetail",
    "action_executor",
    "CacheAwareActionPolicy",
    "EventDrivenActionPolicy",
    "create_default_action_policy",
    # REPL
    "REPLVariable",
    "PolicyPythonREPL",
    "REPLCapability",
    "get_repl_guidance",
    # Backing Store
    "BackingStore",
    "BlackboardBackingStore",
    "StorageHint",
    # Code Generation
    "CodeGenerationActionPolicy",
    "create_code_generation_action_policy",
    # Code Generation Constraints
    "CodeGenerator",
    "FreeFormCodeGenerator",
    "CodeValidator",
    "NoOpValidator",
    "ImportWhitelistValidator",
    "APIKnowledgeBaseValidator",
    "IterationShapeValidator",
    "SkillLibrary",
    "NoOpSkillLibrary",
    "InMemorySkillLibrary",
    "RecoveryStrategy",
    "NoRecovery",
    "DeterministicRecovery",
    "RuntimeGuardrail",
    "NoGuardrail",
    "CapabilityBoundaryGuardrail",
    "TemporalOrderGuardrail",
    "ArgsAwareOrderingRule",
    "ArgsAwareTemporalOrderGuardrail",
    "ApprovalRequiredGuardrail",
    "CompositeGuardrail",
    "ValidationResult",
    "CodeSkill",
    "RecoveryResult",
    "GuardrailDecision",
    # Minimal
    "MinimalActionPolicy",
    "create_minimal_action_policy",
    # Deferred-closure extraction primitive
    "DeferredClosure",
    "deferred",
    "eager_execution",
    "is_eager_execution",
]
