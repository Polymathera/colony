"""Design-monorepo Layer-0 abstraction for Polymathera Colony.

This package implements the per-program git monorepo discipline
described in the design-automation architecture (master §3.1.6, §3.5.1,
§8, §9, with the full critique in
``colony_docs/markdown/apps/git_as_design_state_engine.md``):

- ``DesignMonorepoManifest`` — committed at ``.colony/manifest.json``;
  carries the topology + tooling-carry-over + LFS + webhook config.
- ``DesignMonorepoClient`` — a thin GitPython wrapper exposing the
  exact operations the framework needs (clone, commit-with-identity,
  tag/list checkpoints, fork/list forks, restore, merge, cherry-pick,
  diff, find-existing-tool, list-recent-decisions, branch topology).
- ``bootstrap_design_monorepo`` — initialize a fresh monorepo with the
  framework's directory layout, ``.gitattributes`` (LFS + merge
  drivers), and an opening commit.
- Three ``AgentCapability`` subclasses — ``RepoStateProvider`` (read-
  only), ``DesignCheckpointer`` (write side), ``ToolBuilder``
  (``bootstrap_repo``).
- Custom git merge drivers under ``git_merge/`` for the structured
  artifact types ``.gitattributes`` declares.
- Language-agnostic scaffolds under ``scaffolds/``.

The package is colony-generic: any multi-agent system that wants
git-backed durable design state benefits, not just CPS.
"""

from __future__ import annotations

from .blueprints import design_monorepo_capability_blueprints
from .bootstrap import SCAFFOLD_DIRS, bootstrap_design_monorepo
from .capabilities import (
    BranchScopedCapabilityBase,
    DesignCheckpointer,
    DesignMonorepoCapabilityBase,
    ProjectAuthoringCapability,
    RepoStateProvider,
    ToolBuilder,
)
from .clones import resolve_clone_path
from .client import (
    AGENT_BRANCH_PREFIX,
    BranchExistsError,
    CHECKPOINT_TAG_PREFIX,
    CheckpointNotFoundError,
    DECISIONS_DIR,
    DesignMonorepoClient,
    DesignMonorepoError,
    FORK_BRANCH_PREFIX,
    MAX_DIFF_BYTES,
    MERGE_DRIVER_ENTRY_POINTS,
    MERGE_DRIVER_PATTERNS,
    SESSION_BRANCH_PREFIX,
    TOOL_BRANCH_PREFIX,
    UncommittedChangesError,
)
from .identity import AgentIdentity
from .manifest import (
    DEFAULT_SURFACE_DIRS,
    DesignMonorepoManifest,
    ExtensionsConfig,
    LFSConfig,
    MANIFEST_RELATIVE_PATH,
    MANIFEST_SCHEMA_VERSION,
    ManifestSchemaError,
    SurfaceConfig,
    WebhookConfig,
)
from .models import (
    PROJECT_ACTION_KINDS,
    BootstrapResult,
    BranchNode,
    BranchTopology,
    Checkpoint,
    DecisionEntry,
    DesignDiff,
    DesignDiffEntry,
    ExtensionAuthoredPayload,
    ForkBranch,
    ImportedRemote,
    PageChangeEvent,
    ProjectArtifactAuthoredPayload,
    ProjectArtifactValidationResult,
    RepoBootstrapSpec,
    RepoState,
    ToolEntry,
    ToolMatch,
)
from .registry import (
    REGISTRY_RELATIVE_PATH,
    REGISTRY_SCHEMA_VERSION,
    ToolRegistryError,
    load_registry,
    upsert_tool,
    write_registry,
)
from .manifest_migrate import (
    MigrationResult,
    migrate_manifest,
)


__all__ = (
    # Models
    "PROJECT_ACTION_KINDS",
    "BootstrapResult",
    "BranchNode",
    "BranchTopology",
    "Checkpoint",
    "DecisionEntry",
    "DesignDiff",
    "DesignDiffEntry",
    "ExtensionAuthoredPayload",
    "ForkBranch",
    "ImportedRemote",
    "PageChangeEvent",
    "ProjectArtifactAuthoredPayload",
    "ProjectArtifactValidationResult",
    "RepoBootstrapSpec",
    "RepoState",
    "ToolEntry",
    "ToolMatch",
    # Manifest
    "DEFAULT_SURFACE_DIRS",
    "DesignMonorepoManifest",
    "ExtensionsConfig",
    "LFSConfig",
    "WebhookConfig",
    "MANIFEST_RELATIVE_PATH",
    "MANIFEST_SCHEMA_VERSION",
    "ManifestSchemaError",
    "SurfaceConfig",
    # Identity
    "AgentIdentity",
    # Registry
    "REGISTRY_RELATIVE_PATH",
    "REGISTRY_SCHEMA_VERSION",
    "ToolRegistryError",
    "load_registry",
    "upsert_tool",
    "write_registry",
    # Manifest migration
    "MigrationResult",
    "migrate_manifest",
    # Client + bootstrap
    "DesignMonorepoClient",
    "DesignMonorepoError",
    "BranchExistsError",
    "CheckpointNotFoundError",
    "UncommittedChangesError",
    "CHECKPOINT_TAG_PREFIX",
    "FORK_BRANCH_PREFIX",
    "SESSION_BRANCH_PREFIX",
    "AGENT_BRANCH_PREFIX",
    "TOOL_BRANCH_PREFIX",
    "DECISIONS_DIR",
    "MAX_DIFF_BYTES",
    "MERGE_DRIVER_PATTERNS",
    "MERGE_DRIVER_ENTRY_POINTS",
    "bootstrap_design_monorepo",
    "SCAFFOLD_DIRS",
    # Capabilities
    "RepoStateProvider",
    "BranchScopedCapabilityBase",
    "DesignCheckpointer",
    "DesignMonorepoCapabilityBase",
    "ProjectAuthoringCapability",
    "ToolBuilder",
    "design_monorepo_capability_blueprints",
    "resolve_clone_path",
)
