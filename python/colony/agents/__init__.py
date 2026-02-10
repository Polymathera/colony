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
    from polymathera.colony.samples.code_analysis import CodeAnalysisCoordinator
    from polymathera.colony.vcm.sources import ContextPageSourceFactory

    # Create context page source
    source = ContextPageSourceFactory.create(
        source_type="file_grouper",
        group_id="repo-123",
        repo_path="/path/to/repo",
        tenant_id="tenant-1"
    )
    await source.initialize()
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