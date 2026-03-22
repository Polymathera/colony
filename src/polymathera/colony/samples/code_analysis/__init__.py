"""Code analysis agents demonstrating multi-shard inference and distributed orchestration.

This module provides:
- ClusterAnalyzer: Multi-shard inference over page clusters
- CodeAnalysisCoordinator: Distributed orchestration of analysis

Example:
    ```python
    from polymathera.colony.samples.code_analysis import CodeAnalysisCoordinator
    from polymathera.colony.vcm.sources import BuilInContextPageSourceType
    from polymathera.colony.system import get_vcm

    from polymathera.colony.distributed.ray_utils.serving.context import execution_context

    with execution_context(
        ring=Ring.USER,
        colony_id="colony-456",
        tenant_id="tenant-1",
        session_id="session-789",
        run_id="run-abc",
        origin="cli",
    ):
        # Create context page source
        vcm_handle = get_vcm()
        mmap_result: MmapResult = await vcm_handle.mmap_application_scope(
            scope_id="repo-123",
            source_type=BuilInContextPageSourceType.FILE_GROUPER.value,
            config=MmapConfig(),
            repo_path="/path/to/repo",
        )

        # Spawn coordinator agent
        from polymathera.colony.system import spawn_agents
        coordinator_bp = CodeAnalysisCoordinator.bind(
            metadata=AgentMetadata(
                parameters={"repo_id": "repo-123"},
            ),
        )
        agent_ids = await spawn_agents(blueprints=[coordinator_bp])
    ```
"""

from .basic.cluster_analyzer import ClusterAnalyzer
from .basic.coordinator import CodeAnalysisCoordinator

__all__ = [
    "ClusterAnalyzer",
    "CodeAnalysisCoordinator",
]

# Register agent classes for Ray distribution
# Users should include this module in ray.init(runtime_env={"py_modules": [...]})
_AGENT_CLASSES = {
    "ClusterAnalyzer": ClusterAnalyzer,
    "CodeAnalysisCoordinator": CodeAnalysisCoordinator,
}
