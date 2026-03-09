"""Declarative blackboard key schemas and publishable protocol.

This module replaces scattered f-string key construction across capabilities
with a single declaration that provides: key formatting, pattern generation,
and validation.

Instead of each capability manually building keys with f-strings and helper
methods, declare key schemas once in this module. This replaces both the
class-level ``_KEY`` constants AND the ``_get_*_key()`` helper methods.

Example::

    from polymathera.colony.agents.blackboard.keys import RESULT_KEY

    # Format a specific key
    key = RESULT_KEY.format(scope_id="agent-123", result_id="abc")
    # -> "agent-123:result:analysis:abc"

    # Generate a subscription pattern (wildcards for unspecified parts)
    pattern = RESULT_KEY.pattern(scope_id="agent-123")
    # -> "agent-123:result:analysis:*"

Why this is useful (not just a wrapper):
- ``pattern()`` auto-generates subscription patterns from the same template —
  eliminates the paired ``get_blackboard_key()`` + ``get_key_pattern()`` on every model.
- ``auto_parts`` + ``format_auto()`` eliminates the ``self.agent.tenant_id``
  boilerplate that appears in every helper.
- Canonical schemas in one file make key drift impossible.
- The ``description`` field enables future tooling (key documentation, monitoring).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    from ..base import Agent


# =============================================================================
# BlackboardKeySchema — Declarative Key Definitions
# =============================================================================


@dataclass(frozen=True)
class BlackboardKeySchema:
    """Declarative blackboard key definition.

    Replaces scattered f-string key construction with a single declaration
    that provides: key formatting, pattern generation, and validation.

    Example::

        RESULT_KEY = BlackboardKeySchema(
            template="{scope_id}:analysis:result:{result_id}",
            description="Analysis result for a specific page",
        )

        # Format a specific key
        key = RESULT_KEY.format(scope_id="agent-123", result_id="abc")
        # -> "agent-123:analysis:result:abc"

        # Generate a subscription pattern (wildcards for unspecified parts)
        pattern = RESULT_KEY.pattern(scope_id="agent-123")
        # -> "agent-123:analysis:result:*"

    Attributes:
        template: Key template with ``{variable}`` placeholders.
        description: Human-readable description of what this key represents.
        auto_parts: Parts that should be filled from agent context automatically
            by ``format_auto()`` and ``pattern_auto()``.
    """

    template: str
    description: str = ""
    auto_parts: frozenset[str] = field(default_factory=frozenset)

    def format(self, **parts: str) -> str:
        """Format a specific key with all parts provided.

        Args:
            **parts: Template variable values.

        Returns:
            Fully resolved key string.

        Raises:
            KeyError: If a required template variable is missing.
        """
        return self.template.format(**parts)

    def pattern(self, **parts: str) -> str:
        """Generate a glob pattern, replacing missing parts with ``*``.

        Only replaces template variables that are NOT provided.

        Args:
            **parts: Known template variable values.

        Returns:
            Glob pattern string suitable for blackboard subscriptions.
        """
        result = self.template
        variables = set(re.findall(r'\{(\w+)\}', self.template))
        for var in variables:
            if var not in parts:
                result = result.replace(f"{{{var}}}", "*")
            else:
                result = result.replace(f"{{{var}}}", parts[var])
        return result

    def format_auto(self, agent: "Agent", **parts: str) -> str:
        """Format with auto-parts filled from agent context.

        Auto-parts mapping:
        - ``tenant_id`` -> ``agent.tenant_id``
        - ``agent_id`` -> ``agent.agent_id``

        Other parts must be provided explicitly.

        Args:
            agent: Agent to extract auto-parts from.
            **parts: Additional template variable values.

        Returns:
            Fully resolved key string.
        """
        auto: dict[str, str] = {}
        if "tenant_id" in self.auto_parts:
            auto["tenant_id"] = agent.tenant_id
        if "agent_id" in self.auto_parts:
            auto["agent_id"] = agent.agent_id
        return self.template.format(**{**auto, **parts})

    def pattern_auto(self, agent: "Agent", **parts: str) -> str:
        """Generate pattern with auto-parts filled from agent context.

        Args:
            agent: Agent to extract auto-parts from.
            **parts: Known template variable values.

        Returns:
            Glob pattern string with auto-parts resolved and missing
            parts replaced with ``*``.
        """
        auto: dict[str, str] = {}
        if "tenant_id" in self.auto_parts:
            auto["tenant_id"] = agent.tenant_id
        if "agent_id" in self.auto_parts:
            auto["agent_id"] = agent.agent_id
        all_parts = {**auto, **parts}
        result = self.template
        variables = set(re.findall(r'\{(\w+)\}', self.template))
        for var in variables:
            if var in all_parts:
                result = result.replace(f"{{{var}}}", all_parts[var])
            else:
                result = result.replace(f"{{{var}}}", "*")
        return result


# =============================================================================
# BlackboardPublishable — Protocol for models with key schemas
# =============================================================================


@runtime_checkable
class BlackboardPublishable(Protocol):
    """Protocol for models that can be published to blackboard.

    Models implementing this protocol provide their own key schema,
    enabling automatic key construction and pattern matching.
    Works alongside the existing ``get_blackboard_key(scope_id)`` convention.

    Example::

        class CritiqueRequest(BaseModel, BlackboardPublishable):
            request_id: str
            ...

            @classmethod
            def key_schema(cls) -> BlackboardKeySchema:
                return BlackboardKeySchema(
                    template="{scope_id}:critique_request:{request_id}",
                    description="Critique request",
                )

            def key_parts(self) -> dict[str, str]:
                return {"request_id": self.request_id}
    """

    @classmethod
    def key_schema(cls) -> BlackboardKeySchema: ...

    def key_parts(self) -> dict[str, str]: ...


# =============================================================================
# Canonical Key Schemas
# =============================================================================
# These replace scattered constants across capabilities. Changing a key
# shape is a single-line edit here.

REQUEST_KEY = BlackboardKeySchema(
    template="{sender_id}:request:{request_type}:{request_id}",
    description="Generic capability request",
)

RESULT_KEY = BlackboardKeySchema(
    template="{scope_id}:result:{result_type}:{result_id}",
    description="Generic capability result",
)

# VCM working set (used by WorkingSetCapability)
WORKING_SET_KEY = BlackboardKeySchema(
    template="vcm:working_set:{tenant_id}",
    description="Cluster-wide working set state",
    auto_parts=frozenset({"tenant_id"}),
)

PAGE_STATUS_KEY = BlackboardKeySchema(
    template="vcm:page_status:{tenant_id}",
    description="Cluster-wide page status",
    auto_parts=frozenset({"tenant_id"}),
)

# VCM analysis (used by VCMAnalysisCapability)
VCM_ANALYSIS_RESULT_KEY = BlackboardKeySchema(
    template="vcm_analysis:{tenant_id}:{scope_id}:result:{page_id}",
    description="VCM analysis result for a page",
    auto_parts=frozenset({"tenant_id"}),
)

VCM_ANALYSIS_STATE_KEY = BlackboardKeySchema(
    template="vcm_analysis:{tenant_id}:{scope_id}:state",
    description="VCM analysis state",
    auto_parts=frozenset({"tenant_id"}),
)

# Results (used by ResultCapability)
PARTIAL_RESULT_KEY = BlackboardKeySchema(
    template="results:partial:{tenant_id}:{result_id}",
    description="Partial analysis result",
    auto_parts=frozenset({"tenant_id"}),
)

RESULTS_INDEX_KEY = BlackboardKeySchema(
    template="results:index:{tenant_id}",
    description="Index of all partial results",
    auto_parts=frozenset({"tenant_id"}),
)

# Page registry (used by GlobalPageKeyRegistry)
PAGE_REGISTRY_KEY = BlackboardKeySchema(
    template="key:{tenant_id}:{page_id}",
    description="Page key in global registry",
    auto_parts=frozenset({"tenant_id"}),
)

CLUSTER_SUMMARY_KEY = BlackboardKeySchema(
    template="cluster:{tenant_id}:{cluster_id}",
    description="Cluster summary in global registry",
    auto_parts=frozenset({"tenant_id"}),
)

# Relationships (used by PageGraphCapability)
RELATIONSHIP_KEY = BlackboardKeySchema(
    template="relationship:{source_id}:{target_id}:{rel_type}",
    description="Inter-page relationship",
)
