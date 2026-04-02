
from abc import ABC, abstractmethod

from ...models import ActionPlan


class PlanAccessPolicy(ABC):
    """Policy protocol for plan access control.

    Controls which agents can create, read, update, and approve plans based on
    agent hierarchy, team structure, and plan scope.
    """

    @abstractmethod
    def can_create_plan(self, agent_id: str, plan: ActionPlan) -> bool:
        """Check if agent can create this plan.

        Args:
            agent_id: ID of agent attempting to create plan
            plan: ActionPlan to be created

        Returns:
            True if agent can create plan
        """
        ...

    @abstractmethod
    def can_read_plan(self, agent_id: str, plan: ActionPlan) -> bool:
        """Check if agent can read this plan.

        Args:
            agent_id: ID of agent attempting to read plan
            plan: ActionPlan to be read

        Returns:
            True if agent can read plan
        """
        ...

    @abstractmethod
    def can_update_plan(self, agent_id: str, plan: ActionPlan) -> bool:
        """Check if agent can update this plan.

        Args:
            agent_id: ID of agent attempting to update plan
            plan: ActionPlan to be updated

        Returns:
            True if agent can update plan
        """
        ...

    @abstractmethod
    def can_approve_plan(self, agent_id: str, plan: ActionPlan) -> bool:
        """Check if agent can approve this plan.

        Args:
            agent_id: ID of agent attempting to approve plan
            plan: ActionPlan to be approved

        Returns:
            True if agent can approve plan
        """
        ...


class HierarchicalAccessPolicy(PlanAccessPolicy):
    """Hierarchical access control for plans.

    Access rules:
    - Agents can always manage (create/read/update) their own plans
    - Parent agents can read/approve child plans
    - Agents in same team can read each other's plans
    - Only parent agents can approve plans requiring approval
    - System-scope plans require higher-level approval
    """

    def __init__(
        self,
        agent_hierarchy: dict[str, str] | None = None,
        team_structure: dict[str, set[str]] | None = None,
    ):
        """Initialize hierarchical access policy.

        Args:
            agent_hierarchy: Mapping of agent_id -> parent_agent_id
            team_structure: Mapping of team_name -> set of agent_ids
        """
        self.agent_hierarchy = agent_hierarchy or {}
        self.team_structure = team_structure or {}

    def can_create_plan(self, agent_id: str, plan: ActionPlan) -> bool:
        """Check if agent can create this plan.

        Rules:
        - Agents can create plans for themselves
        - Plans with parent_plan_id require approval from parent plan's agent

        Args:
            agent_id: ID of agent attempting to create plan
            plan: ActionPlan to be created

        Returns:
            True if agent can create plan
        """
        # Agents can always create plans for themselves
        if plan.agent_id == agent_id:
            return True

        # Cannot create plans for other agents
        return False

    def can_read_plan(self, agent_id: str, plan: ActionPlan) -> bool:
        """Check if agent can read this plan.

        Rules:
        - Agents can read their own plans
        - Parent agents can read child plans
        - Team members can read each other's plans
        - GLOBAL visibility plans can be read by anyone

        Args:
            agent_id: ID of agent attempting to read plan
            plan: ActionPlan to be read

        Returns:
            True if agent can read plan
        """
        # Global plans are readable by all
        if plan.visibility == "global":
            return True

        # Agents can read their own plans
        if plan.agent_id == agent_id:
            return True

        # Parent agents can read child plans
        if self._is_parent(agent_id, plan.agent_id):
            return True

        # Team members can read each other's plans
        if self._is_in_team(agent_id, plan.agent_id):
            return True

        # Agents in subscriber list can read
        if agent_id in plan.subscribers:
            return True

        return False

    def can_update_plan(self, agent_id: str, plan: ActionPlan) -> bool:
        """Check if agent can update this plan.

        Rules:
        - Agents can update their own plans
        - Parent agents can update child plans (for coordination)

        Args:
            agent_id: ID of agent attempting to update plan
            plan: ActionPlan to be updated

        Returns:
            True if agent can update plan
        """
        # Agents can update their own plans
        if plan.agent_id == agent_id:
            return True

        # Parent agents can update child plans
        if self._is_parent(agent_id, plan.agent_id):
            return True

        return False

    def can_approve_plan(self, agent_id: str, plan: ActionPlan) -> bool:
        """Check if agent can approve this plan.

        Rules:
        - Only parent agents can approve plans
        - System-scope plans require approval from system-level agents

        Args:
            agent_id: ID of agent attempting to approve plan
            plan: ActionPlan to be approved

        Returns:
            True if agent can approve plan
        """
        # Only parent can approve child's plan
        if self._is_parent(agent_id, plan.agent_id):
            # System-scope plans need higher-level approval
            if plan.scope == "system":
                # Check if approver is at system level (no parent)
                return agent_id not in self.agent_hierarchy
            return True

        return False

    def _is_parent(self, potential_parent: str, agent_id: str) -> bool:
        """Check if potential_parent is parent of agent_id.

        Args:
            potential_parent: Agent ID of potential parent
            agent_id: Agent ID to check

        Returns:
            True if potential_parent is direct parent of agent_id
        """
        return self.agent_hierarchy.get(agent_id) == potential_parent

    def _is_in_team(self, agent1: str, agent2: str) -> bool:
        """Check if two agents are in same team.

        Args:
            agent1: First agent ID
            agent2: Second agent ID

        Returns:
            True if agents are in same team
        """
        for team_members in self.team_structure.values():
            if agent1 in team_members and agent2 in team_members:
                return True
        return False

    def _is_in_hierarchy(self, agent1: str, agent2: str) -> bool:
        """Check if two agents are in same hierarchy chain.

        Args:
            agent1: First agent ID
            agent2: Second agent ID

        Returns:
            True if agents share a hierarchy chain
        """
        # Check if agent1 is ancestor of agent2
        current = agent2
        while current in self.agent_hierarchy:
            if self.agent_hierarchy[current] == agent1:
                return True
            current = self.agent_hierarchy[current]

        # Check if agent2 is ancestor of agent1
        current = agent1
        while current in self.agent_hierarchy:
            if self.agent_hierarchy[current] == agent2:
                return True
            current = self.agent_hierarchy[current]

        return False

