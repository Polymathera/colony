"""Agent system for autonomous computational entities.

This module provides the multi-agent system infrastructure for distributed
inference over extremely long contexts.
This module provides the complete agent infrastructure:
- Base agent classes and lifecycle management
- Agent system coordination and discovery
- Communication layer (exactly-once, sync/async)
- Tool discovery and execution
- Job submission and management
- Rich action tracking with reasoning traces
- Blackboard working memory
- Standalone agent deployment

Example:
    ```python
    from polymathera.colony.vcm.sources import BuilInContextPageSourceType
    from polymathera.colony.system import get_vcm

    # Create context page source
    vcm_handle = get_vcm()
    mmap_result: MmapResult = await vcm_handle.mmap_application_scope(
        scope_id="repo-123",
        source_type=BuilInContextPageSourceType.FILE_GROUPER.value,
        config=MmapConfig(),
        tenant_id="tenant-1",
        repo_path="/path/to/repo",
    )
    ```
"""

# Base classes
from .base import Agent, AgentManagerBase, AgentState

# Agent system coordination
from .system import AgentSystemDeployment, AgentSystemState

from .config import AgentSystemConfig

# Blackboard (new enhanced implementation)
from .blackboard import (
    EnhancedBlackboard,
    BlackboardScope,
    KeyPatternFilter,
    EventTypeFilter,
    AgentFilter,
    BlackboardEvent,
)

# Backward compatibility alias
Blackboard = EnhancedBlackboard

# Tool system
from .tools import ToolManagerDeployment, ToolResultCache, ToolSystemState

# Standalone deployment
from .standalone import StandaloneAgentDeployment

# Data models
from .models import (
    Action,
    ActionCheckpoint,
    ActionResult,
    ActionStatus,
    ActionType,
    ActionPolicyExecutionState,
    ActionPolicyIO,
    ActionPolicyIterationResult,
    AgentSpawnSpec,
    ActionPlan,
    PolicyREPL,
    Ref,
    TodoItem,
    TodoItemStatus,
    ToolCall,
    ToolMetadata,
    ToolParameterSchema,
)

__all__ = [
    # Base
    "Agent",
    "AgentManagerBase",
    "AgentState",
    # System
    "AgentSystemDeployment",
    "AgentSystemState",
    "AgentSystemConfig",
    # Blackboard
    "Blackboard",
    "EnhancedBlackboard",
    "BlackboardScope",
    "KeyPatternFilter",
    "EventTypeFilter",
    "AgentFilter",
    "BlackboardEvent",
    # Tools
    "ToolManagerDeployment",
    "ToolResultCache",
    "ToolSystemState",
    # Standalone
    "StandaloneAgentDeployment",
    # Models
    "Action",
    "ActionCheckpoint",
    "ActionResult",
    "ActionStatus",
    "ActionType",
    "ActionPolicyExecutionState",
    "ActionPolicyIO",
    "ActionPolicyIterationResult",
    "AgentSpawnSpec",
    "ActionPlan",
    "PolicyREPL",
    "Ref",
    "TodoItem",
    "TodoItemStatus",
    "ToolCall",
    "ToolMetadata",
    "ToolParameterSchema",
]