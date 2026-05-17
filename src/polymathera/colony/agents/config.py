
"""Configuration for Agent System."""

from __future__ import annotations

from pydantic import ConfigDict, Field

from ..distributed.config import (
    ConfigComponent,
    Mutability,
    Tier,
    register_polymathera_config,
    tier_metadata,
)
from ..distributed.observability.config import TracingConfig
from ..distributed.ray_utils import serving


@register_polymathera_config(path="agent_system")
class AgentSystemConfig(ConfigComponent):
    """Configuration for Agent System deployment.

    Tier: ``L1_OPERATOR`` for the structural fields (e.g. backend type);
    ``RELOADABLE`` so an operator can update them without re-deploying.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    max_retries: int = Field(
        default=3,
        json_schema_extra=tier_metadata(
            tier=Tier.L1_OPERATOR, mutability=Mutability.RELOADABLE,
        ),
    )

    # Session management configuration
    enable_sessions: bool = Field(
        default=True,
        json_schema_extra=tier_metadata(tier=Tier.L1_OPERATOR),
    )
    default_session_ttl: float = Field(
        default=86400.0,  # 24 hours
        json_schema_extra=tier_metadata(
            tier=Tier.L1_OPERATOR, mutability=Mutability.RELOADABLE,
        ),
    )

    # Blackboard configuration
    blackboard_backend_type: str = Field(
        default="distributed",
        description='"distributed", "redis", or "memory"',
        json_schema_extra=tier_metadata(tier=Tier.L1_OPERATOR),
    )

    # Observability / tracing
    tracing: TracingConfig = Field(default_factory=TracingConfig)

    def add_deployments_to_app(self, app: serving.Application, top_level: bool) -> None:
        if not top_level:
            from .system import AgentSystemDeployment
            app.add_deployment(
                AgentSystemDeployment.bind(
                    blackboard_backend_type=self.blackboard_backend_type,
                ),
                name="agent_system"
            )

        from .standalone import StandaloneAgentDeployment

        app.add_deployment(
            StandaloneAgentDeployment.bind(),
            name="standalone_agents"
        )

        # Session management
        if self.enable_sessions:
            from .sessions.manager import SessionManagerDeployment
            app.add_deployment(
                SessionManagerDeployment.bind(
                    default_session_ttl=self.default_session_ttl,
                ),
                name="session_manager"
            )


