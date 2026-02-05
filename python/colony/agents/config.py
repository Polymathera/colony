
"""Configuration for Agent System."""

from __future__ import annotations

from dataclasses import dataclass

from ..distributed.ray_utils import serving


@dataclass
class AgentSystemConfig:
    """Configuration for Agent System deployment.

    Attributes:
        max_retries: Maximum number of retries for agent operations
        enable_sessions: Whether to enable session management
        default_session_ttl: Default session TTL in seconds (24 hours)
        max_sessions_per_tenant: Maximum sessions per tenant
    """

    max_retries: int = 3

    # Session management configuration
    enable_sessions: bool = True
    default_session_ttl: float = 86400.0  # 24 hours
    max_sessions_per_tenant: int = 100

    def add_deployments_to_app(self, app: serving.Application, top_level: bool) -> None:
        if not top_level:
            from .system import AgentSystemDeployment
            app.add_deployment(
                AgentSystemDeployment.bind(),
                name="agent_system"
            )

        from .tools import ToolManagerDeployment
        from .standalone import StandaloneAgentDeployment

        app.add_deployment(
            ToolManagerDeployment.bind(),
            name="tool_manager"
        )

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
                    max_sessions_per_tenant=self.max_sessions_per_tenant,
                ),
                name="session_manager"
            )


