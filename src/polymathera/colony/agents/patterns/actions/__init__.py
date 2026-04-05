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
    create_default_action_policy,
)

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

from .minimal import (
    MinimalActionPolicy,
    create_minimal_action_policy,
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
    # Minimal
    "MinimalActionPolicy",
    "create_minimal_action_policy",
]
