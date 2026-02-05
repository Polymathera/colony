"""Code analysis agents demonstrating multi-shard inference and distributed orchestration.

This module provides:
- ClusterAnalyzer: Multi-shard inference over page clusters
- CodeAnalysisCoordinator: Distributed orchestration of analysis

Example:
    ```python
    from polymathera.colony.samples.code_analysis import (
        CodeAnalysisCoordinator,
        ContextPageSourceFactory
    )

    # Create context page source
    source = ContextPageSourceFactory.create(
        source_type="file_grouper",
        group_id="repo-123",
        repo_path="/path/to/repo",
        tenant_id="tenant-1"
    )
    await source.initialize()

    # Spawn coordinator agent
    agent_system = serving.get_deployment(app_name, names.agent_system)
    agent_ids = await agent_system.spawn_agents([{
        "agent_type": "polymathera.colony.samples.code_analysis.CodeAnalysisCoordinator",
        "metadata": {
            "repo_id": "repo-123",
            "context_page_source_config": source.get_config()
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
