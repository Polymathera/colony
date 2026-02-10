"""Code analysis agents demonstrating multi-shard inference and distributed orchestration.

This module provides:
- ClusterAnalyzer: Multi-shard inference over page clusters
- CodeAnalysisCoordinator: Distributed orchestration of analysis

Example:
    ```python
    from polymathera.colony.samples.code_analysis import CodeAnalysisCoordinator
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

    # Spawn coordinator agent
    agent_system = serving.get_deployment(app_name, names.agent_system)
    agent_ids = await agent_system.spawn_agents([{
        "agent_type": "polymathera.colony.samples.code_analysis.CodeAnalysisCoordinator",
        "metadata": {
            "repo_id": "repo-123",
        }
    }])
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
