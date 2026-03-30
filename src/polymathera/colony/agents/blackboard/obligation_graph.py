"""ObligationGraph for compliance reasoning.

The ObligationGraph is a bipartite graph stored in blackboard linking
requirements to code/test/doc shards with LLM-derived edges labeled
'satisfies', 'missing', 'partial'. This is a very general abstraction that
supports Compliance reasoning (with specs, policies, regulations, etc.).

This abstraction is generalizable beyond code analysis to:
- Regulatory compliance (GDPR, HIPAA, SOC2)
- Specification conformance (API specs, protocols)
- Policy enforcement (security policies, architectural rules)
- Contract verification (business contracts, SLAs)

The bipartite structure:
- One side: Requirements (specs, regulations, policies)
- Other side: Artifacts (code, tests, documentation)
- Edges: Relationships (satisfies, missing, partial) with LLM-derived confidence
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .blackboard import EnhancedBlackboard

class NodeType(str, Enum):
    """Type of node in obligation graph."""

    REQUIREMENT = "requirement"  # Requirement, spec, regulation, policy
    ARTIFACT = "artifact"  # Code, test, doc, config


class ComplianceRelationship(str, Enum):
    """Relationship between requirement and artifact."""

    SATISFIES = "satisfies"  # Artifact fully satisfies requirement
    PARTIAL = "partial"  # Artifact partially satisfies requirement
    MISSING = "missing"  # Requirement not satisfied (artifact doesn't address it)
    VIOLATES = "violates"  # Artifact violates requirement


class ObligationNode(BaseModel):
    """A node in the obligation graph with full traceability.

    Can represent either a requirement or an artifact.

    Examples:
        Requirement node:
        ```python
        req = ObligationNode(
            node_type=NodeType.REQUIREMENT,
            content={
                "requirement_id": "GDPR_Art17",
                "title": "Right to Erasure",
                "description": "Users must be able to request deletion of their data",
                "source": "GDPR Article 17"
            },
            location=None
        )
        ```

        Artifact node (code):
        ```python
        artifact = ObligationNode(
            node_type=NodeType.ARTIFACT,
            content={
                "artifact_type": "code",
                "file": "api/user_data.py",
                "function": "delete_user_data",
                "description": "Deletes user data from database"
            },
            location=CodeLocation(file="api/user_data.py", line=42)
        )
        ```
    """

    node_id: str = Field(
        default_factory=lambda: f"node_{uuid.uuid4().hex}",
        description="Unique node identifier"
    )

    node_type: NodeType = Field(
        description="Type of node (requirement or artifact)"
    )

    # Node content (structure depends on type)
    content: dict[str, Any] = Field(
        description="Node content (requirement details or artifact details)"
    )

    # Location (for artifacts)
    location: dict[str, Any] | None = Field(
        default=None,
        description="Location of artifact (file, line, etc.)"
    )

    # Metadata
    title: str | None = Field(
        default=None,
        description="Human-readable title"
    )

    description: str | None = Field(
        default=None,
        description="Node description"
    )

    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorization"
    )

    # Version tracking
    version: int = Field(
        default=1,
        description="Node version number"
    )

    previous_version_id: str | None = Field(
        default=None,
        description="Previous version of this node"
    )

    # Audit trail
    created_at: float = Field(
        default_factory=time.time,
        description="When node was created"
    )

    created_by: str | None = Field(
        default=None,
        description="Agent that created this node"
    )

    updated_at: float = Field(
        default_factory=time.time,
        description="When node was last updated"
    )

    updated_by: str | None = Field(
        default=None,
        description="Agent that last updated this node"
    )

    # Traceability
    source_reference: str | None = Field(
        default=None,
        description="Source document/system reference"
    )

    external_id: str | None = Field(
        default=None,
        description="ID in external system"
    )

    change_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="History of changes to this node"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    def record_change(self, change_type: str, details: dict[str, Any], agent_id: str) -> None:
        """Record a change to this node.

        Args:
            change_type: Type of change
            details: Change details
            agent_id: Agent making the change
        """
        self.change_history.append({
            "timestamp": time.time(),
            "change_type": change_type,
            "details": details,
            "agent_id": agent_id,
            "version": self.version
        })
        self.updated_at = time.time()
        self.updated_by = agent_id


class ObligationEdge(BaseModel):
    """An edge in the obligation graph.

    Connects a requirement to an artifact with a relationship type.

    Examples:
        Full satisfaction:
        ```python
        edge = ObligationEdge(
            source_id="req_gdpr_art17",
            target_id="artifact_delete_user_data",
            relationship=ComplianceRelationship.SATISFIES,
            confidence=0.90,
            evidence=[
                "Function deletes all user data from database",
                "Includes cascade delete for related records",
                "Returns confirmation of deletion"
            ]
        )
        ```

        Partial satisfaction:
        ```python
        edge = ObligationEdge(
            source_id="req_gdpr_art17",
            target_id="artifact_api_endpoint",
            relationship=ComplianceRelationship.PARTIAL,
            confidence=0.70,
            evidence=["API endpoint exists for deletion"],
            gaps=["Does not delete S3 backups", "Missing audit trail"]
        )
        ```
    """

    edge_id: str = Field(
        default_factory=lambda: f"edge_{uuid.uuid4().hex}",
        description="Unique edge identifier"
    )

    # Bipartite structure: requirement -> artifact
    source_id: str = Field(
        description="Requirement node ID"
    )

    target_id: str = Field(
        description="Artifact node ID"
    )

    # Relationship
    relationship: ComplianceRelationship = Field(
        description="Type of relationship"
    )

    confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence in this relationship (LLM-derived)"
    )

    # Supporting information
    evidence: list[str] = Field(
        default_factory=list,
        description="Evidence supporting this relationship"
    )

    gaps: list[str] = Field(
        default_factory=list,
        description="Gaps if relationship is PARTIAL or MISSING"
    )

    # Metadata
    derived_by: str | None = Field(
        default=None,
        description="Agent or analysis that derived this edge"
    )

    created_at: float = Field(
        default_factory=time.time,
        description="When edge was created"
    )

    validated: bool = Field(
        default=False,
        description="Whether this edge has been validated"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class ComplianceReport(BaseModel):
    """Compliance status report."""

    total_requirements: int = Field(
        description="Total number of requirements"
    )

    satisfied_requirements: int = Field(
        description="Number of fully satisfied requirements"
    )

    partially_satisfied: int = Field(
        description="Number of partially satisfied requirements"
    )

    unsatisfied_requirements: int = Field(
        description="Number of unsatisfied requirements"
    )

    compliance_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall compliance score"
    )

    critical_gaps: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Critical compliance gaps"
    )

    generated_at: float = Field(
        default_factory=time.time,
        description="When report was generated"
    )


class ObligationGraph:
    """Manages obligation graph in blackboard with full traceability.

    Provides:
    - Adding requirements and artifacts with versioning
    - Linking with compliance relationships
    - Querying compliance status
    - Identifying gaps and violations
    - Full audit trail and traceability
    - Impact analysis for changes
    - Coverage metrics
    """

    def __init__(self, blackboard: EnhancedBlackboard):
        """Initialize obligation graph.

        Args:
            blackboard: EnhancedBlackboard instance
        """
        self.blackboard = blackboard
        self.node_namespace = "obligation_node"
        self.edge_namespace = "obligation_edge"
        self.trace_namespace = "obligation_trace"
        self._node_cache: dict[str, ObligationNode] = {}  # Cache for performance
        self._edge_cache: dict[str, ObligationEdge] = {}  # Cache for performance

    async def add_requirement(
        self,
        content: dict[str, Any],
        title: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        created_by: str | None = None
    ) -> ObligationNode:
        """Add requirement node to graph.

        Args:
            content: Requirement content
            title: Optional title
            description: Optional description
            tags: Optional tags
            created_by: Optional creator agent ID

        Returns:
            Created requirement node
        """
        node = ObligationNode(
            node_type=NodeType.REQUIREMENT,
            content=content,
            title=title,
            description=description,
            tags=tags or [],
            created_by=created_by
        )

        await self._store_node(node)
        return node

    async def add_artifact(
        self,
        content: dict[str, Any],
        location: dict[str, Any] | None = None,
        title: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        created_by: str | None = None
    ) -> ObligationNode:
        """Add artifact node to graph.

        Args:
            content: Artifact content
            location: Optional location info
            title: Optional title
            description: Optional description
            tags: Optional tags
            created_by: Optional creator agent ID

        Returns:
            Created artifact node
        """
        node = ObligationNode(
            node_type=NodeType.ARTIFACT,
            content=content,
            location=location,
            title=title,
            description=description,
            tags=tags or [],
            created_by=created_by
        )

        await self._store_node(node)
        return node

    async def link(
        self,
        requirement_id: str,
        artifact_id: str,
        relationship: ComplianceRelationship,
        confidence: float,
        evidence: list[str] | None = None,
        gaps: list[str] | None = None,
        derived_by: str | None = None
    ) -> ObligationEdge:
        """Link requirement to artifact.

        Args:
            requirement_id: Requirement node ID
            artifact_id: Artifact node ID
            relationship: Type of relationship
            confidence: Confidence in relationship
            evidence: Optional evidence
            gaps: Optional gaps (for PARTIAL/MISSING)
            derived_by: Optional agent that derived this

        Returns:
            Created edge
        """
        edge = ObligationEdge(
            source_id=requirement_id,
            target_id=artifact_id,
            relationship=relationship,
            confidence=confidence,
            evidence=evidence or [],
            gaps=gaps or [],
            derived_by=derived_by
        )

        await self._store_edge(edge)
        return edge

    async def get_unsatisfied_requirements(self) -> list[ObligationNode]:
        """Get all unsatisfied requirements.

        Returns:
            List of requirements with no SATISFIES edges
        """
        # Query all requirement nodes
        all_requirements = await self._get_all_nodes(NodeType.REQUIREMENT)

        unsatisfied = []
        for req in all_requirements:
            # Get edges from this requirement
            edges = await self.get_requirement_edges(req.node_id)

            # Check if any edge is SATISFIES
            has_satisfaction = any(
                e.relationship == ComplianceRelationship.SATISFIES
                for e in edges
            )

            if not has_satisfaction:
                unsatisfied.append(req)

        return unsatisfied

    async def get_partially_satisfied_requirements(self) -> list[tuple[ObligationNode, list[ObligationEdge]]]:
        """Get requirements that are only partially satisfied.

        Returns:
            List of (requirement, partial_edges) tuples
        """
        all_requirements = await self._get_all_nodes(NodeType.REQUIREMENT)

        partially_satisfied = []
        for req in all_requirements:
            edges = await self.get_requirement_edges(req.node_id)

            # Has partial but not full satisfaction
            has_partial = any(e.relationship == ComplianceRelationship.PARTIAL for e in edges)
            has_full = any(e.relationship == ComplianceRelationship.SATISFIES for e in edges)

            if has_partial and not has_full:
                partial_edges = [e for e in edges if e.relationship == ComplianceRelationship.PARTIAL]
                partially_satisfied.append((req, partial_edges))

        return partially_satisfied

    async def get_compliance_status(self) -> ComplianceReport:
        """Get overall compliance status.

        Returns:
            Compliance report
        """
        all_requirements = await self._get_all_nodes(NodeType.REQUIREMENT)
        total = len(all_requirements)

        satisfied = 0
        partial = 0
        unsatisfied = 0
        critical_gaps = []

        for req in all_requirements:
            edges = await self.get_requirement_edges(req.node_id)

            has_full = any(e.relationship == ComplianceRelationship.SATISFIES for e in edges)
            has_partial = any(e.relationship == ComplianceRelationship.PARTIAL for e in edges)

            if has_full:
                satisfied += 1
            elif has_partial:
                partial += 1
            else:
                unsatisfied += 1

                # Track as critical gap
                critical_gaps.append({
                    "requirement_id": req.node_id,
                    "title": req.title,
                    "description": req.description
                })

        # Calculate compliance score
        # Full satisfaction = 1.0, partial = 0.5, missing = 0.0
        score = (satisfied + 0.5 * partial) / total if total > 0 else 1.0

        return ComplianceReport(
            total_requirements=total,
            satisfied_requirements=satisfied,
            partially_satisfied=partial,
            unsatisfied_requirements=unsatisfied,
            compliance_score=score,
            critical_gaps=critical_gaps
        )

    async def get_requirement_edges(self, requirement_id: str) -> list[ObligationEdge]:
        """Get all edges from a requirement.

        Args:
            requirement_id: Requirement node ID

        Returns:
            List of edges
        """
        # Query edges with this requirement as source
        entries = await self.blackboard.query(
            namespace=self.edge_namespace,
            tags={requirement_id},  # Edges are tagged with source_id
            limit=1000
        )

        edges = []
        for entry in entries:
            try:
                edge = ObligationEdge(**entry.value)
                if edge.source_id == requirement_id:
                    edges.append(edge)
                    self._edge_cache[edge.edge_id] = edge
            except Exception:
                continue

        return edges

    async def get_artifact_edges(self, artifact_id: str) -> list[ObligationEdge]:
        """Get all edges to an artifact.

        Args:
            artifact_id: Artifact node ID

        Returns:
            List of edges
        """
        # Query edges with this artifact as target
        entries = await self.blackboard.query(
            namespace=self.edge_namespace,
            tags={artifact_id},  # Edges are tagged with target_id
            limit=1000
        )

        edges = []
        for entry in entries:
            try:
                edge = ObligationEdge(**entry.value)
                if edge.target_id == artifact_id:
                    edges.append(edge)
                    self._edge_cache[edge.edge_id] = edge
            except Exception:
                continue

        return edges

    async def _store_node(self, node: ObligationNode) -> None:
        """Store node in blackboard.

        Args:
            node: Node to store
        """
        from ..scopes import ScopeUtils
        key = ScopeUtils.format_key(node_namespace=self.node_namespace, node_type=node.node_type.value, node_id=node.node_id)

        await self.blackboard.write(
            key=key,
            value=node.model_dump(),
            tags={"obligation_node", node.node_type.value, *node.tags},
            created_by=node.created_by
        )

    async def _store_edge(self, edge: ObligationEdge) -> None:
        """Store edge in blackboard.

        Args:
            edge: Edge to store
        """
        from ..scopes import ScopeUtils
        key = ScopeUtils.format_key(edge_namespace=self.edge_namespace, source_id=edge.source_id, target_id=edge.target_id)

        await self.blackboard.write(
            key=key,
            value=edge.model_dump(),
            tags={
                "obligation_edge",
                edge.relationship.value,
                edge.source_id,
                edge.target_id
            },
            created_by=edge.derived_by
        )

    async def _get_all_nodes(self, node_type: NodeType | None = None) -> list[ObligationNode]:
        """Get all nodes, optionally filtered by type.

        Args:
            node_type: Optional node type filter

        Returns:
            List of nodes
        """
        # Build namespace pattern
        if node_type:
            from ..scopes import ScopeUtils
            namespace = ScopeUtils.format_key(node_namespace=self.node_namespace, node_type=node_type.value)
        else:
            namespace = self.node_namespace

        # Query blackboard
        entries = await self.blackboard.query(
            namespace=namespace,
            tags={"obligation_node"},
            limit=10000  # Large limit to get all nodes
        )

        nodes = []
        for entry in entries:
            try:
                node = ObligationNode(**entry.value)
                nodes.append(node)
                self._node_cache[node.node_id] = node
            except Exception:
                continue

        return nodes

    async def get_impact_analysis(
        self,
        node_id: str,
        change_type: str = "modify"
    ) -> dict[str, Any]:
        """Analyze impact of changing a node.

        Args:
            node_id: Node to analyze
            change_type: Type of change: 'modify', 'delete'

        Returns:
            Impact analysis results
        """
        node = self._node_cache.get(node_id)
        if not node:
            # Try fetching from blackboard
            key = f"{self.node_namespace}:{node_id}"
            data = await self.blackboard.read(key)
            if data:
                node = ObligationNode(**data)
                self._node_cache[node_id] = node

        if not node:
            return {"error": f"Node {node_id} not found"}

        impact = {
            "node_id": node_id,
            "node_type": node.node_type.value,
            "change_type": change_type,
            "directly_affected": [],
            "indirectly_affected": [],
            "compliance_impact": None,
            "risk_level": "low"
        }

        if node.node_type == NodeType.REQUIREMENT:
            # Changing requirement affects all linked artifacts
            edges = await self.get_requirement_edges(node_id)
            for edge in edges:
                impact["directly_affected"].append({
                    "artifact_id": edge.target_id,
                    "relationship": edge.relationship.value,
                    "confidence": edge.confidence
                })

            # If requirement is deleted/modified, compliance may be affected
            if change_type == "delete":
                impact["compliance_impact"] = "Requirement removal - ensure not legally required"
                impact["risk_level"] = "high"
            else:
                impact["compliance_impact"] = "Requirement modification - verify artifacts still comply"
                impact["risk_level"] = "medium"

        else:  # ARTIFACT
            # Changing artifact affects requirements it satisfies
            edges = await self.get_artifact_edges(node_id)
            for edge in edges:
                impact["directly_affected"].append({
                    "requirement_id": edge.source_id,
                    "relationship": edge.relationship.value,
                    "confidence": edge.confidence
                })

                # If artifact satisfies requirement, modification is risky
                if edge.relationship == ComplianceRelationship.SATISFIES:
                    impact["risk_level"] = "high"
                    impact["compliance_impact"] = "May break compliance with satisfied requirements"

        return impact

    async def get_coverage_metrics(self) -> dict[str, Any]:
        """Calculate requirement coverage metrics.

        Returns:
            Coverage metrics
        """
        all_requirements = await self._get_all_nodes(NodeType.REQUIREMENT)
        all_artifacts = await self._get_all_nodes(NodeType.ARTIFACT)

        metrics = {
            "total_requirements": len(all_requirements),
            "total_artifacts": len(all_artifacts),
            "fully_covered": 0,
            "partially_covered": 0,
            "uncovered": 0,
            "coverage_percentage": 0.0,
            "artifacts_per_requirement": 0.0,
            "high_confidence_links": 0,
            "low_confidence_links": 0,
            "unvalidated_links": 0
        }

        if not all_requirements:
            return metrics

        total_edges = 0
        for req in all_requirements:
            edges = await self.get_requirement_edges(req.node_id)

            satisfies_count = sum(
                1 for e in edges
                if e.relationship == ComplianceRelationship.SATISFIES
            )
            partial_count = sum(
                1 for e in edges
                if e.relationship == ComplianceRelationship.PARTIAL
            )

            if satisfies_count > 0:
                metrics["fully_covered"] += 1
            elif partial_count > 0:
                metrics["partially_covered"] += 1
            else:
                metrics["uncovered"] += 1

            # Count edge confidence levels
            for edge in edges:
                total_edges += 1
                if edge.confidence >= 0.8:
                    metrics["high_confidence_links"] += 1
                elif edge.confidence < 0.5:
                    metrics["low_confidence_links"] += 1
                if not edge.validated:
                    metrics["unvalidated_links"] += 1

        # Calculate percentages
        metrics["coverage_percentage"] = (
            (metrics["fully_covered"] + 0.5 * metrics["partially_covered"]) /
            metrics["total_requirements"] * 100
        )

        if metrics["total_requirements"] > 0:
            metrics["artifacts_per_requirement"] = (
                total_edges / metrics["total_requirements"]
            )

        return metrics

    async def trace_requirement(
        self,
        requirement_id: str
    ) -> dict[str, Any]:
        """Full traceability report for a requirement.

        Args:
            requirement_id: Requirement to trace

        Returns:
            Traceability information
        """
        from ..scopes import ScopeUtils

        req_node = self._node_cache.get(requirement_id)
        if not req_node:
            key = ScopeUtils.format_key(node_namespace=self.node_namespace, node_type=NodeType.REQUIREMENT.value, node_id=requirement_id)

            data = await self.blackboard.read(key)
            if data:
                req_node = ObligationNode(**data)

        if not req_node:
            return {"error": f"Requirement {requirement_id} not found"}

        trace = {
            "requirement": {
                "id": requirement_id,
                "title": req_node.title,
                "description": req_node.description,
                "source": req_node.source_reference,
                "version": req_node.version,
                "created": req_node.created_at,
                "updated": req_node.updated_at
            },
            "implementations": [],
            "tests": [],
            "documentation": [],
            "compliance_status": "unknown",
            "validation_status": "unvalidated",
            "change_history": req_node.change_history
        }

        # Get all linked artifacts
        edges = await self.get_requirement_edges(requirement_id)

        for edge in edges:
            artifact_key = ScopeUtils.format_key(node_namespace=self.node_namespace, node_type=NodeType.ARTIFACT.value, node_id=edge.target_id)

            artifact_data = await self.blackboard.read(artifact_key)

            if artifact_data:
                artifact = ObligationNode(**artifact_data)
                artifact_info = {
                    "id": edge.target_id,
                    "type": artifact.content.get("artifact_type", "unknown"),
                    "location": artifact.location,
                    "relationship": edge.relationship.value,
                    "confidence": edge.confidence,
                    "validated": edge.validated,
                    "evidence": edge.evidence,
                    "gaps": edge.gaps
                }

                # Categorize by artifact type
                artifact_type = artifact.content.get("artifact_type", "").lower()
                if "test" in artifact_type:
                    trace["tests"].append(artifact_info)
                elif "doc" in artifact_type:
                    trace["documentation"].append(artifact_info)
                else:
                    trace["implementations"].append(artifact_info)

                # Update compliance status
                if edge.relationship == ComplianceRelationship.SATISFIES:
                    trace["compliance_status"] = "satisfied"
                elif edge.relationship == ComplianceRelationship.PARTIAL and trace["compliance_status"] != "satisfied":
                    trace["compliance_status"] = "partial"
                elif edge.relationship == ComplianceRelationship.VIOLATES:
                    trace["compliance_status"] = "violated"

                if edge.validated:
                    trace["validation_status"] = "validated"

        if not edges and trace["compliance_status"] == "unknown":
            trace["compliance_status"] = "uncovered"

        return trace

    async def update_node_version(
        self,
        node_id: str,
        node_type: NodeType,
        changes: dict[str, Any],
        agent_id: str,
        change_reason: str | None = None
    ) -> ObligationNode:
        """Update node with versioning and audit trail.

        Args:
            node_id: Node to update
            changes: Changes to apply
            agent_id: Agent making changes
            change_reason: Optional reason for change

        Returns:
            Updated node
        """
        from ..scopes import ScopeUtils
        # Get current node
        node = self._node_cache.get(node_id)
        if not node:
            key = ScopeUtils.format_key(node_namespace=self.node_namespace, node_type=node_type.value, node_id=node_id)
            data = await self.blackboard.read(key)
            if data:
                node = ObligationNode(**data)

        if not node:
            raise ValueError(f"Node {node_id} not found")

        # Create new version
        old_version_id = f"{node_id}_v{node.version}"

        trace_key = ScopeUtils.format_key(node_namespace=self.trace_namespace, old_version_id=old_version_id)

        # Archive old version
        await self.blackboard.write(
            key=trace_key,
            value=node.model_dump(),
            tags={"obligation_version", node.node_type.value, "archived"},
            created_by=agent_id
        )

        # Update node
        node.version += 1
        node.previous_version_id = old_version_id

        # Apply changes
        for key, value in changes.items():
            if hasattr(node, key):
                setattr(node, key, value)

        # Record change
        node.record_change(
            change_type="update",
            details={
                "changes": changes,
                "reason": change_reason
            },
            agent_id=agent_id
        )

        # Store updated node
        await self._store_node(node)

        # Update cache
        self._node_cache[node_id] = node

        return node


# Utility functions

async def create_requirement_from_spec(
    spec_text: str,
    spec_id: str,
    blackboard: EnhancedBlackboard,
    llm_client: Any
) -> ObligationNode:
    """Create requirement node from specification text.

    This is a utility function that would use an LLM to extract structured
    requirements. For now, creates a basic requirement node.

    DO NOT USE THIS FUNCTION. IT IS ONLY FOR DEMONSTRATION PURPOSES.

    Args:
        spec_text: Specification text
        spec_id: Specification identifier
        blackboard: Blackboard instance
        llm_client: LLM for extracting requirements (not used in basic version)

    Returns:
        Created requirement node
    """
    # Basic implementation: create requirement from text
    # A full implementation would use LLM to parse and structure the spec

    graph = ObligationGraph(blackboard)
    return await graph.add_requirement(
        content={
            "spec_id": spec_id,
            "text": spec_text,
            "source": "manual_spec"
        },
        title=f"Requirement from {spec_id}",
        description=spec_text[:200] if len(spec_text) > 200 else spec_text,
        tags=["spec_derived", spec_id]
    )


async def analyze_compliance(
    requirement_id: str,
    codebase_artifacts: list[dict[str, Any]],
    blackboard: EnhancedBlackboard,
    llm_client: Any
) -> list[ObligationEdge]:
    """Analyze which artifacts satisfy a requirement.

    This is a utility function that would use an LLM to determine compliance.
    For now, it creates basic edges based on simple matching.

    DO NOT USE THIS FUNCTION. IT IS ONLY FOR DEMONSTRATION PURPOSES.

    Args:
        requirement_id: Requirement to analyze
        codebase_artifacts: List of code artifacts with metadata
        blackboard: Blackboard instance
        llm_client: LLM for compliance analysis (not used in basic version)

    Returns:
        List of compliance edges
    """
    # Basic implementation: create edges for artifacts
    # A full implementation would use LLM to analyze actual compliance

    graph = ObligationGraph(blackboard)
    edges = []

    # For each artifact, create a basic relationship
    for artifact in codebase_artifacts:
        # First add the artifact to the graph if not already there
        artifact_node = await graph.add_artifact(
            content=artifact,
            location=artifact.get("location"),
            title=artifact.get("title", "Artifact"),
            tags=artifact.get("tags", [])
        )

        # Create a basic edge (would use LLM analysis in full version)
        # For now, assume partial satisfaction with low confidence
        edge = await graph.link(
            requirement_id=requirement_id,
            artifact_id=artifact_node.node_id,
            relationship=ComplianceRelationship.PARTIAL,
            confidence=0.5,  # Low confidence since not using LLM
            evidence=["Automated basic matching"],
            gaps=["Requires manual verification"],
            derived_by="automated_analysis"
        )
        edges.append(edge)

    return edges

