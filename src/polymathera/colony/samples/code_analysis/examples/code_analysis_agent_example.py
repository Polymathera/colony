"""Example: Using code analysis agents to analyze a git repository.

This demonstrates:
1. Creating a ContextPageSource from a git repo
2. Spawning a CodeAnalysisCoordinator agent
3. Monitoring the analysis progress
4. Retrieving the final report

Prerequisites:
- Ray cluster running with polymathera application deployed
- Git repository cloned locally
- Repository sharded into VCM pages
"""

import asyncio
import logging

from polymathera.colony.vcm.sources import BuilInContextPageSourceType
from polymathera.colony.vcm.models import MmapConfig, MmapResult
from polymathera.colony.agents.models import AgentMetadata
from polymathera.colony.samples.code_analysis.basic.coordinator import CodeAnalysisCoordinator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def analyze_repository(
    repo_path: str,
    repo_id: str,
    session_id: str,
    run_id: str,
):
    """Analyze a git repository using code analysis agents.

    Args:
        repo_path: Local path to cloned git repository
        repo_id: Unique identifier for this repository
        app_name: Polymmathera serving application name

    Returns:
        Analysis report dictionary
    """
    logger.info(f"Starting code analysis for repo: {repo_id}")

    # 1. Create ContextPageSource
    logger.info("Creating ContextPageSource...")
    # Map the repository to VCM pages using the built-in file grouper source type
    from polymathera.colony.system import get_agent_system, get_vcm
    vcm_handle = get_vcm()
    mmap_result: MmapResult = await vcm_handle.mmap_application_scope(
        scope_id="repo-123",
        group_id="vmr-456",
        tenant_id="tenant-1",
        source_type=BuilInContextPageSourceType.FILE_GROUPER.value,
        config=MmapConfig(),
        repo_path=repo_path,
    )

    # TODO: Build the page graph and persist it so the coordinator can load it (or let coordinator build it - needs repo access)

    logger.info("ContextPageSource initialized")

    # 2. Get AgentSystemDeployment handle
    agent_system = get_agent_system()

    # 3. Spawn CodeAnalysisCoordinator
    logger.info("Spawning CodeAnalysisCoordinator...")
    coordinator_bp = CodeAnalysisCoordinator.bind(
        metadata=AgentMetadata(
            session_id=session_id,
            run_id=run_id,
            parameters={"repo_id": repo_id},
        ),
        bound_pages=[],  # Coordinator doesn't need pages
    )

    # TODO: Pass LLM requirements to spawn_agents
    agent_ids = await agent_system.spawn_agents(
        blueprints=[coordinator_bp],
        soft_affinity=False
    )

    coordinator_id = agent_ids[0]
    logger.info(f"Coordinator spawned: {coordinator_id}")

    # 4. Monitor progress (poll agent state)
    # TODO: Implement proper monitoring via agent system
    # For now, just wait
    logger.info("Monitoring analysis progress...")
    await asyncio.sleep(60)  # Wait for analysis to complete

    # 5. Retrieve results
    # TODO: Implement result retrieval from blackboard or agent state
    logger.info("Analysis complete (TODO: retrieve actual results)")

    return {
        "repo_id": repo_id,
        "coordinator_id": coordinator_id,
        "status": "completed"
    }


async def main():
    """Example usage."""
    # Example: Analyze polymathera repository itself
    result = await analyze_repository(
        repo_path="/path/to/polymathera",  # Update with actual path
        repo_id="polymathera-self",
        app_name="polymathera-app"
    )

    logger.info(f"Analysis result: {result}")


if __name__ == "__main__":
    asyncio.run(main())

