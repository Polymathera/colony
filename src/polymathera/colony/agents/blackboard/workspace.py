"""Git-like workspace versioning for multi-agent collaboration.

As specified in AGENT_FRAMEWORK.md:
"The blackboard should allow a git-like workflow that supports a versioning
mechanism to track changes to the blackboard (or designated data structure)
and revert to previous versions: branching, merging, conflict resolution,
rebase, cherry-pick, reset, clean."

This enables:
- Multiple agents working on separate branches
- Speculative exploration without affecting main branch
- Merging workspaces with conflict detection
- Rollback to previous versions
- Cherry-picking insights from experiments

Integration:
- Stored in EnhancedBlackboard
- Uses event system for notifications
- Supports atomic operations via transactions
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class BranchStatus(str, Enum):
    """Status of a workspace branch."""

    ACTIVE = "active"  # Currently in use
    MERGED = "merged"  # Merged into another branch
    ABANDONED = "abandoned"  # No longer used
    ARCHIVED = "archived"  # Archived for reference


class ConflictType(str, Enum):
    """Type of merge conflict."""

    CONTENT_CONFLICT = "content_conflict"  # Same key, different values
    DELETE_MODIFY_CONFLICT = "delete_modify_conflict"  # Deleted in one, modified in other
    TYPE_CONFLICT = "type_conflict"  # Same key, incompatible types
    DEPENDENCY_CONFLICT = "dependency_conflict"  # Conflicting dependencies


class WorkspaceCommit(BaseModel):
    """A commit in workspace history.

    Similar to git commits, tracks changes to workspace data.
    """

    commit_id: str = Field(
        default_factory=lambda: f"commit_{uuid.uuid4().hex}",
        description="Unique commit identifier"
    )

    parent_commit_id: str | None = Field(
        default=None,
        description="Parent commit (None for initial commit)"
    )

    branch_name: str = Field(
        description="Branch this commit belongs to"
    )

    # Changes in this commit
    changes: dict[str, Any] = Field(
        default_factory=dict,
        description="Changed key-value pairs"
    )

    deletions: list[str] = Field(
        default_factory=list,
        description="Keys deleted in this commit"
    )

    # Metadata
    message: str = Field(
        description="Commit message explaining changes"
    )

    author_agent_id: str = Field(
        description="Agent that made this commit"
    )

    timestamp: float = Field(
        default_factory=time.time,
        description="When commit was created"
    )

    tags: list[str] = Field(
        default_factory=list,
        description="Tags for this commit (milestone, release, etc.)"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional commit metadata"
    )


class WorkspaceBranch(BaseModel):
    """A branch in a workspace.

    Branches provide isolated work environments. Each branch tracks
    its own history and can be merged with other branches.
    """

    branch_name: str = Field(
        description="Unique branch name within workspace"
    )

    parent_branch: str | None = Field(
        default=None,
        description="Parent branch (None for main branch)"
    )

    # Branch data (current state)
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Current branch data"
    )

    # Versioning
    head_commit_id: str | None = Field(
        default=None,
        description="Current commit (HEAD)"
    )

    base_commit_id: str | None = Field(
        default=None,
        description="Base commit where branch diverged from parent"
    )

    commit_history: list[str] = Field(
        default_factory=list,
        description="Commit IDs in chronological order"
    )

    # Ownership
    owner_agent_id: str = Field(
        description="Agent that owns this branch"
    )

    collaborators: list[str] = Field(
        default_factory=list,
        description="Agent IDs with write access"
    )

    # Status
    status: BranchStatus = Field(
        default=BranchStatus.ACTIVE,
        description="Branch status"
    )

    # Metadata
    description: str | None = Field(
        default=None,
        description="Branch description/purpose"
    )

    created_at: float = Field(
        default_factory=time.time,
        description="When branch was created"
    )

    updated_at: float = Field(
        default_factory=time.time,
        description="When branch was last updated"
    )

    tags: list[str] = Field(
        default_factory=list,
        description="Branch tags"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    def can_commit(self, agent_id: str) -> bool:
        """Check if agent can commit to this branch.

        Args:
            agent_id: Agent ID

        Returns:
            True if agent has write access
        """
        return agent_id == self.owner_agent_id or agent_id in self.collaborators

    def add_collaborator(self, agent_id: str) -> None:
        """Add collaborator to branch.

        Args:
            agent_id: Agent to add
        """
        if agent_id not in self.collaborators:
            self.collaborators.append(agent_id)
            self.updated_at = time.time()


class Workspace(BaseModel):
    """A workspace containing multiple branches for collaboration.

    A Workspace is like a Git repository - it contains multiple branches
    where agents can work independently and merge their changes.

    Examples:
        Create workspace with main branch:
        ```python
        workspace = Workspace(
            workspace_id="my_project",
            name="My Analysis Project",
            description="Multi-agent code analysis workspace"
        )
        # Automatically has a 'main' branch
        ```

        Add experimental branch:
        ```python
        branch = workspace.create_branch(
            branch_name="experiment/new_approach",
            parent_branch="main",
            owner_agent_id="researcher_001"
        )
        ```
    """

    workspace_id: str = Field(
        default_factory=lambda: f"workspace_{uuid.uuid4().hex}",
        description="Unique workspace identifier"
    )

    name: str = Field(
        description="Workspace name"
    )

    # Branches (like a git repo with multiple branches)
    branches: dict[str, WorkspaceBranch] = Field(
        default_factory=dict,
        description="Branch name -> branch mapping"
    )

    current_branch: str = Field(
        default="main",
        description="Currently active branch"
    )

    # All commits across all branches
    commits: dict[str, WorkspaceCommit] = Field(
        default_factory=dict,
        description="Commit ID -> commit mapping"
    )

    # Workspace-level metadata
    description: str | None = Field(
        default=None,
        description="Workspace description"
    )

    owner_agent_id: str = Field(
        default="system",
        description="Workspace owner"
    )

    created_at: float = Field(
        default_factory=time.time,
        description="When workspace was created"
    )

    updated_at: float = Field(
        default_factory=time.time,
        description="When workspace was last updated"
    )

    tags: list[str] = Field(
        default_factory=list,
        description="Workspace tags"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    def __init__(self, **data):
        """Initialize workspace with main branch."""
        super().__init__(**data)

        # Create main branch if not exists
        if "main" not in self.branches:
            self.branches["main"] = WorkspaceBranch(
                branch_name="main",
                parent_branch=None,
                owner_agent_id=self.owner_agent_id,
                description="Main branch"
            )

    def create_branch(
        self,
        branch_name: str,
        parent_branch: str,
        owner_agent_id: str,
        description: str | None = None
    ) -> WorkspaceBranch:
        """Create a new branch.

        Args:
            branch_name: New branch name
            parent_branch: Parent branch to fork from
            owner_agent_id: Branch owner
            description: Optional description

        Returns:
            Created branch

        Raises:
            ValueError: If branch exists or parent doesn't exist
        """
        if branch_name in self.branches:
            raise ValueError(f"Branch '{branch_name}' already exists")

        if parent_branch not in self.branches:
            raise ValueError(f"Parent branch '{parent_branch}' doesn't exist")

        parent = self.branches[parent_branch]

        # Create new branch with parent's current data
        branch = WorkspaceBranch(
            branch_name=branch_name,
            parent_branch=parent_branch,
            data=parent.data.copy(),
            base_commit_id=parent.head_commit_id,
            owner_agent_id=owner_agent_id,
            description=description
        )

        self.branches[branch_name] = branch
        self.updated_at = time.time()

        return branch

    def get_branch(self, branch_name: str) -> WorkspaceBranch | None:
        """Get branch by name.

        Args:
            branch_name: Branch name

        Returns:
            Branch or None
        """
        return self.branches.get(branch_name)

    def switch_branch(self, branch_name: str) -> None:
        """Switch current branch.

        Args:
            branch_name: Branch to switch to

        Raises:
            ValueError: If branch doesn't exist
        """
        if branch_name not in self.branches:
            raise ValueError(f"Branch '{branch_name}' doesn't exist")

        self.current_branch = branch_name
        self.updated_at = time.time()

    def list_branches(self) -> list[str]:
        """List all branch names.

        Returns:
            List of branch names
        """
        return list(self.branches.keys())

    def get_active_branches(self) -> list[str]:
        """Get active branch names.

        Returns:
            List of active branch names
        """
        return [
            name for name, branch in self.branches.items()
            if branch.status == BranchStatus.ACTIVE
        ]


class MergeConflict(BaseModel):
    """A conflict during workspace merge."""

    conflict_type: ConflictType = Field(
        description="Type of conflict"
    )

    key: str = Field(
        description="Conflicting key"
    )

    source_value: Any = Field(
        description="Value in source workspace"
    )

    target_value: Any = Field(
        description="Value in target workspace"
    )

    resolution: str | None = Field(
        default=None,
        description="How to resolve: 'use_source', 'use_target', 'merge', 'manual'"
    )

    resolved_value: Any | None = Field(
        default=None,
        description="Resolved value after conflict resolution"
    )


class MergeResult(BaseModel):
    """Result of merging two branches."""

    success: bool = Field(
        description="Whether merge succeeded"
    )

    merged_branch: str | None = Field(
        default=None,
        description="Name of merged branch (if successful)"
    )

    conflicts: list[MergeConflict] = Field(
        default_factory=list,
        description="Conflicts encountered"
    )

    auto_resolved: int = Field(
        default=0,
        description="Number of conflicts auto-resolved"
    )

    manual_resolution_needed: int = Field(
        default=0,
        description="Number of conflicts needing manual resolution"
    )

    changes_applied: int = Field(
        default=0,
        description="Number of changes applied"
    )


class WorkspaceManager:
    """Manages workspaces in blackboard.

    Provides git-like operations:
    - Versioning: Track changes over time
    - create_workspace: Create new workspace
    - create_branch: Create branch in workspace
    - commit: Commit changes to branch
    - merge: Merge branches
    - rebase: Rebase branch on another
    - cherry-pick: Cherry-pick commits
    - reset: Reset to previous commit
    - Conflict resolution: Resolve merge conflicts
    """

    def __init__(self, blackboard: Any):
        """Initialize workspace manager.

        Args:
            blackboard: EnhancedBlackboard instance
        """
        self.blackboard = blackboard
        self.namespace = "workspace"
        self.commit_namespace = "workspace_commit"

    async def create_workspace(
        self,
        name: str,
        owner_agent_id: str,
        description: str | None = None
    ) -> Workspace:
        """Create a new workspace.

        Args:
            name: Workspace name
            owner_agent_id: Owner agent ID
            description: Optional description

        Returns:
            Created workspace
        """
        workspace = Workspace(
            name=name,
            owner_agent_id=owner_agent_id,
            description=description
        )

        # Store workspace
        await self._store_workspace(workspace)

        return workspace

    async def get_workspace(self, workspace_id: str) -> Workspace | None:
        """Get workspace by ID.

        Args:
            workspace_id: Workspace ID

        Returns:
            Workspace or None if not found
        """
        from ..scopes import ScopeUtils
        key = ScopeUtils.format_key(namespace=self.namespace, workspace_id=workspace_id)
        data = await self.blackboard.read(key)

        if data is None:
            return None

        return Workspace(**data)

    async def commit(
        self,
        workspace_id: str,
        branch_name: str,
        changes: dict[str, Any],
        deletions: list[str],
        message: str,
        author_agent_id: str
    ) -> WorkspaceCommit:
        """Commit changes to a branch.

        Args:
            workspace_id: Workspace ID
            branch_name: Branch name
            changes: Changed key-value pairs
            deletions: Deleted keys
            message: Commit message
            author_agent_id: Committing agent

        Returns:
            Created commit
        """
        workspace = await self.get_workspace(workspace_id)
        if not workspace:
            raise ValueError(f"Workspace {workspace_id} not found")

        branch = workspace.get_branch(branch_name)
        if not branch:
            raise ValueError(f"Branch {branch_name} not found in workspace {workspace_id}")

        if not branch.can_commit(author_agent_id):
            raise PermissionError(f"Agent {author_agent_id} cannot commit to branch {branch_name}")

        # Create commit
        commit = WorkspaceCommit(
            branch_name=branch_name,
            parent_commit_id=branch.head_commit_id,
            changes=changes,
            deletions=deletions,
            message=message,
            author_agent_id=author_agent_id
        )

        # Apply changes to branch data
        for key, value in changes.items():
            branch.data[key] = value
        for key in deletions:
            branch.data.pop(key, None)

        # Update branch
        branch.head_commit_id = commit.commit_id
        branch.commit_history.append(commit.commit_id)
        branch.updated_at = time.time()

        # Store commit in workspace
        workspace.commits[commit.commit_id] = commit
        workspace.updated_at = time.time()

        # Store workspace
        await self._store_workspace(workspace)

        return commit

    async def merge_branches(
        self,
        workspace_id: str,
        source_branch: str,
        target_branch: str,
        merge_strategy: str = "auto",
        author_agent_id: str | None = None
    ) -> MergeResult:
        """Merge source branch into target branch.

        Args:
            workspace_id: Workspace ID
            source_branch: Source branch name
            target_branch: Target branch name
            merge_strategy: Merge strategy: 'auto', 'manual', 'ours', 'theirs'
            author_agent_id: Agent performing merge

        Returns:
            Merge result
        """
        workspace = await self.get_workspace(workspace_id)
        if not workspace:
            return MergeResult(
                success=False,
                conflicts=[],
                manual_resolution_needed=1
            )

        source = workspace.get_branch(source_branch)
        target = workspace.get_branch(target_branch)

        if not source or not target:
            return MergeResult(
                success=False,
                conflicts=[],
                manual_resolution_needed=1
            )

        # Find common ancestor commit
        common_ancestor = self._find_common_ancestor(workspace, source, target)

        # Three-way merge: compare source, target, and ancestor
        conflicts = await self._detect_conflicts_three_way(source, target, common_ancestor)

        # Resolve conflicts based on strategy
        if merge_strategy == "auto":
            resolved = await self._auto_resolve_conflicts(conflicts)
        elif merge_strategy == "ours":
            resolved = [self._resolve_with_ours(c) for c in conflicts]
        elif merge_strategy == "theirs":
            resolved = [self._resolve_with_theirs(c) for c in conflicts]
        else:
            resolved = conflicts  # Manual resolution needed

        # Check if any conflicts remain unresolved
        unresolved = [c for c in resolved if c.resolution is None]

        if unresolved:
            return MergeResult(
                success=False,
                conflicts=unresolved,
                manual_resolution_needed=len(unresolved)
            )

        # Apply merge
        changes = {}
        for conflict in resolved:
            if conflict.resolved_value is not None:
                changes[conflict.key] = conflict.resolved_value

        # Add non-conflicting changes from source
        for key, value in source.data.items():
            if key not in target.data:
                changes[key] = value

        # Commit merge to target
        if author_agent_id:
            await self.commit(
                workspace_id=workspace_id,
                branch_name=target_branch,
                changes=changes,
                deletions=[],
                message=f"Merge {source_branch} into {target_branch}",
                author_agent_id=author_agent_id
            )

        # Update source status
        source.status = BranchStatus.MERGED
        workspace.updated_at = time.time()
        await self._store_workspace(workspace)

        return MergeResult(
            success=True,
            merged_branch=target_branch,
            conflicts=conflicts,
            auto_resolved=len(resolved),
            changes_applied=len(changes)
        )

    def _find_common_ancestor(
        self,
        workspace: Workspace,
        source: WorkspaceBranch,
        target: WorkspaceBranch
    ) -> dict[str, Any] | None:
        """Find common ancestor data for three-way merge.

        Args:
            workspace: Workspace containing branches
            source: Source branch
            target: Target branch

        Returns:
            Common ancestor data or None
        """
        # If source was branched from target, use target's base
        if source.parent_branch == target.branch_name and source.base_commit_id:
            commit = workspace.commits.get(source.base_commit_id)
            if commit:
                # Reconstruct data at that commit
                # This is simplified - real implementation would replay commits
                return {}

        # For now, return empty dict as base
        return {}

    async def _detect_conflicts_three_way(
        self,
        source: WorkspaceBranch,
        target: WorkspaceBranch,
        ancestor: dict[str, Any] | None
    ) -> list[MergeConflict]:
        """Detect conflicts using three-way merge.

        Args:
            source: Source branch
            target: Target branch
            ancestor: Common ancestor data

        Returns:
            List of conflicts
        """
        conflicts = []
        ancestor = ancestor or {}

        # Check all keys that exist in any version
        all_keys = set(source.data.keys()) | set(target.data.keys()) | set(ancestor.keys())

        for key in all_keys:
            source_value = source.data.get(key)
            target_value = target.data.get(key)
            ancestor_value = ancestor.get(key)

            # Both modified differently from ancestor
            if source_value != target_value:
                if source_value != ancestor_value and target_value != ancestor_value:
                    # Real conflict - both sides changed
                    conflict_type = self._classify_conflict(source_value, target_value)
                    conflicts.append(MergeConflict(
                        conflict_type=conflict_type,
                        key=key,
                        source_value=source_value,
                        target_value=target_value
                    ))
                # If only one side changed, no conflict (take the change)

        return conflicts

    def _classify_conflict(self, source_value: Any, target_value: Any) -> ConflictType:
        """Classify type of conflict.

        Args:
            source_value: Value in source
            target_value: Value in target

        Returns:
            Conflict type
        """
        # Check type compatibility
        if type(source_value) != type(target_value):
            return ConflictType.TYPE_CONFLICT

        # Default to content conflict
        return ConflictType.CONTENT_CONFLICT

    async def _auto_resolve_conflicts(
        self,
        conflicts: list[MergeConflict]
    ) -> list[MergeConflict]:
        """Auto-resolve conflicts where possible.

        Args:
            conflicts: Conflicts to resolve

        Returns:
            Conflicts with resolutions
        """
        resolved = []

        for conflict in conflicts:
            # Try to auto-resolve based on conflict type
            if conflict.conflict_type == ConflictType.CONTENT_CONFLICT:
                # Use heuristics for auto-resolution
                # For now, keep as unresolved
                resolved.append(conflict)
            else:
                resolved.append(conflict)

        return resolved

    def _resolve_with_ours(self, conflict: MergeConflict) -> MergeConflict:
        """Resolve conflict by keeping target value.

        Args:
            conflict: Conflict to resolve

        Returns:
            Resolved conflict
        """
        conflict.resolution = "use_target"
        conflict.resolved_value = conflict.target_value
        return conflict

    def _resolve_with_theirs(self, conflict: MergeConflict) -> MergeConflict:
        """Resolve conflict by using source value.

        Args:
            conflict: Conflict to resolve

        Returns:
            Resolved conflict
        """
        conflict.resolution = "use_source"
        conflict.resolved_value = conflict.source_value
        return conflict

    async def rebase(
        self,
        workspace_id: str,
        branch_name: str,
        onto_branch: str,
        author_agent_id: str
    ) -> bool:
        """Rebase branch onto another branch.

        Replays commits from branch on top of onto_branch.

        Args:
            workspace_id: Workspace ID
            branch_name: Branch to rebase
            onto_branch: Branch to rebase onto
            author_agent_id: Agent performing rebase

        Returns:
            True if successful
        """
        workspace = await self.get_workspace(workspace_id)
        if not workspace:
            return False

        branch = workspace.get_branch(branch_name)
        onto = workspace.get_branch(onto_branch)

        if not branch or not onto:
            return False

        # Find commits to replay (after divergence point)
        commits_to_replay = []
        for commit_id in branch.commit_history:
            commit = workspace.commits.get(commit_id)
            if commit and commit_id not in onto.commit_history:
                commits_to_replay.append(commit)

        # Reset branch to onto's HEAD
        branch.data = onto.data.copy()
        branch.base_commit_id = onto.head_commit_id

        # Replay commits
        for commit in commits_to_replay:
            # Apply commit's changes
            for key, value in commit.changes.items():
                branch.data[key] = value
            for key in commit.deletions:
                branch.data.pop(key, None)

            # Create new commit
            new_commit = WorkspaceCommit(
                branch_name=branch_name,
                parent_commit_id=branch.head_commit_id,
                changes=commit.changes,
                deletions=commit.deletions,
                message=f"Rebase: {commit.message}",
                author_agent_id=author_agent_id
            )

            workspace.commits[new_commit.commit_id] = new_commit
            branch.head_commit_id = new_commit.commit_id
            branch.commit_history.append(new_commit.commit_id)

        branch.updated_at = time.time()
        workspace.updated_at = time.time()
        await self._store_workspace(workspace)

        return True

    async def cherry_pick(
        self,
        workspace_id: str,
        target_branch: str,
        commit_id: str,
        author_agent_id: str
    ) -> bool:
        """Cherry-pick a commit into branch.

        Args:
            workspace_id: Workspace ID
            target_branch: Target branch name
            commit_id: Commit to cherry-pick
            author_agent_id: Agent performing cherry-pick

        Returns:
            True if successful
        """
        workspace = await self.get_workspace(workspace_id)
        if not workspace:
            return False

        commit = workspace.commits.get(commit_id)
        if not commit:
            return False

        branch = workspace.get_branch(target_branch)
        if not branch:
            return False

        # Apply commit changes to target
        await self.commit(
            workspace_id=workspace_id,
            branch_name=target_branch,
            changes=commit.changes,
            deletions=commit.deletions,
            message=f"Cherry-pick: {commit.message}",
            author_agent_id=author_agent_id
        )

        return True

    async def reset(
        self,
        workspace_id: str,
        branch_name: str,
        commit_id: str,
        author_agent_id: str,
        hard: bool = False
    ) -> bool:
        """Reset branch to a specific commit.

        Args:
            workspace_id: Workspace ID
            branch_name: Branch to reset
            commit_id: Commit to reset to
            author_agent_id: Agent performing reset
            hard: If True, discard changes; if False, keep changes

        Returns:
            True if successful
        """
        workspace = await self.get_workspace(workspace_id)
        if not workspace:
            return False

        branch = workspace.get_branch(branch_name)
        if not branch:
            return False

        commit = workspace.commits.get(commit_id)
        if not commit or commit.branch_name != branch_name:
            return False

        if hard:
            # Hard reset: reconstruct data at that commit
            # Start from initial state and replay commits up to target
            branch.data = {}
            for cid in branch.commit_history:
                if cid == commit_id:
                    break
                c = workspace.commits.get(cid)
                if c:
                    for key, value in c.changes.items():
                        branch.data[key] = value
                    for key in c.deletions:
                        branch.data.pop(key, None)

        # Update HEAD
        branch.head_commit_id = commit_id
        # Truncate history after reset point
        if commit_id in branch.commit_history:
            idx = branch.commit_history.index(commit_id)
            branch.commit_history = branch.commit_history[:idx+1]

        branch.updated_at = time.time()
        workspace.updated_at = time.time()
        await self._store_workspace(workspace)

        return True

    async def _store_workspace(self, workspace: Workspace) -> None:
        """Store workspace in blackboard.

        Args:
            workspace: Workspace to store
        """
        from ..scopes import ScopeUtils
        key = ScopeUtils.format_key(namespace=self.namespace, workspace_id=workspace.workspace_id)

        # Tags for filtering
        tags = {"workspace", workspace.name}
        tags.update(workspace.tags)

        await self.blackboard.write(
            key=key,
            value=workspace.model_dump(),
            tags=tags,
            created_by=workspace.owner_agent_id
        )


# Utility functions

async def create_project_workspace(
    blackboard: Any,
    name: str,
    owner_agent_id: str = "system",
    description: str | None = None
) -> Workspace:
    """Create a new project workspace.

    Args:
        blackboard: Blackboard instance
        name: Workspace name
        owner_agent_id: Owner agent ID
        description: Workspace description

    Returns:
        Created workspace
    """
    manager = WorkspaceManager(blackboard)
    return await manager.create_workspace(
        name=name,
        owner_agent_id=owner_agent_id,
        description=description
    )


async def create_experiment_branch(
    blackboard: Any,
    workspace_id: str,
    experiment_name: str,
    agent_id: str
) -> WorkspaceBranch:
    """Create experimental branch in workspace.

    Args:
        blackboard: Blackboard instance
        workspace_id: Workspace ID
        experiment_name: Experiment name
        agent_id: Agent creating branch

    Returns:
        Created branch
    """
    manager = WorkspaceManager(blackboard)
    workspace = await manager.get_workspace(workspace_id)

    if not workspace:
        raise ValueError(f"Workspace {workspace_id} not found")

    branch = workspace.create_branch(
        branch_name=f"experiment/{experiment_name}",
        parent_branch="main",
        owner_agent_id=agent_id,
        description=f"Experimental branch for {experiment_name}"
    )

    await manager._store_workspace(workspace)
    return branch

