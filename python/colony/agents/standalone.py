"""Standalone Agent Deployment for agents not tied to VLLM replicas.

These agents can:
- Run without page context
- Submit inference to any VLLM replica
- Perform coordination, supervision, or tool management
"""

import logging

from ..distributed.ray_utils import serving
from ..cluster.config import LLMDeploymentConfig
from .base import AgentManagerBase

logger = logging.getLogger(__name__)


@serving.deployment
class StandaloneAgentDeployment(AgentManagerBase):
    """Deployment for standalone agents not tied to VLLM replicas.

    Example:
        ```python
        from colony.distributed.ray_utils import serving
        from colony.agents import StandaloneAgentDeployment

        app = serving.Application(name="agent-app")
        app.add_deployment(
            StandaloneAgentDeployment.bind(),
            name="standalone_agents",
        )
        await app.start()
        ```
    """

    def __init__(self, deployment_config: LLMDeploymentConfig | None = None):
        """Initialize standalone agent deployment.

        Uses environment-based discovery to find all services.
        """
        # Initialize parent AgentManagerBase
        super().__init__(deployment_config=deployment_config)

    @serving.initialize_deployment
    async def initialize(self):
        """Initialize self-contained state after deployment starts."""
        logger.info("Initializing StandaloneAgentDeployment")
        await super().initialize()
        logger.info("StandaloneAgentDeployment initialized (awaiting app ready for handle discovery)")

    @serving.on_app_ready
    async def on_ready(self):
        """Discover sibling deployment handles after all deployments are started: LLMCluster, VCM, AgentSystem, ToolManager, ConsciousnessManager."""
        await self.discover_handles()
        logger.info("StandaloneAgentDeployment handle discovery complete")

    def _get_deployment_replica_id(self) -> str:
        """Get deployment replica ID for this manager."""
        return serving.get_my_replica_id()

    # Inherits start_agent, stop_agent, list_agents from AgentManagerBase

