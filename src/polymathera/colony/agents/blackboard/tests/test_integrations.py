"""Quick integration test for fixed blackboard data structures.

This tests that the fixed implementations properly integrate with
the rest of the framework.
"""

from __future__ import annotations

import asyncio
from typing import Any

# Import the fixed blackboard structures
from polymathera.colony.agents.blackboard.workspace import (
    Workspace,
    WorkspaceBranch,
    WorkspaceManager,
    create_project_workspace,
    create_experiment_branch,
)
from polymathera.colony.agents.blackboard.task_graph import (
    Task,
    TaskGraph,
    TaskStatus,
    TaskPriority,
    create_analysis_task,
)
from polymathera.colony.agents.blackboard.obligation_graph import (
    ObligationNode,
    ObligationEdge,
    ObligationGraph,
    NodeType,
    ComplianceRelationship,
)
from polymathera.colony.agents.blackboard.causality_timeline import (
    CausalEvent,
    CausalityTimeline,
    EventType,
    VectorClock,
)

# Import patterns that integrate with blackboard
from polymathera.colony.agents.patterns.scope import (
    AnalysisScope,
    ScopeAwareResult,
)


class MockBlackboard:
    """Mock blackboard for testing."""

    def __init__(self):
        self.storage: dict[str, Any] = {}
        self.event_bus = MockEventBus()

    async def read(self, key: str) -> Any:
        return self.storage.get(key)

    async def write(self, key: str, value: Any, **kwargs) -> None:
        self.storage[key] = value

    def transaction(self):
        return MockTransaction(self)


class MockEventBus:
    """Mock event bus."""

    async def publish(self, **kwargs) -> None:
        pass


class MockTransaction:
    """Mock transaction context."""

    def __init__(self, blackboard: MockBlackboard):
        self.blackboard = blackboard
        self.temp_storage = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # Commit
            self.blackboard.storage.update(self.temp_storage)

    async def read(self, key: str) -> Any:
        return self.temp_storage.get(key) or self.blackboard.storage.get(key)

    async def write(self, key: str, value: Any) -> None:
        self.temp_storage[key] = value


async def test_workspace_branching():
    """Test the fixed Workspace as a container of branches."""
    print("\n=== Testing Workspace (Container of Branches) ===")

    blackboard = MockBlackboard()
    manager = WorkspaceManager(blackboard)

    # Create workspace
    workspace = await manager.create_workspace(
        name="TestProject",
        owner_agent_id="agent_001",
        description="Test workspace with branches"
    )

    print(f"✓ Created workspace '{workspace.name}' with main branch")
    assert "main" in workspace.branches
    assert workspace.current_branch == "main"

    # Create experimental branch
    exp_branch = workspace.create_branch(
        branch_name="experiment/new_algo",
        parent_branch="main",
        owner_agent_id="agent_002",
        description="Testing new algorithm"
    )

    print(f"✓ Created branch '{exp_branch.branch_name}' from main")
    assert exp_branch.parent_branch == "main"

    # Commit to branch
    commit = await manager.commit(
        workspace_id=workspace.workspace_id,
        branch_name="experiment/new_algo",
        changes={"algorithm": "neural_net", "accuracy": 0.95},
        deletions=[],
        message="Implemented neural network approach",
        author_agent_id="agent_002"
    )

    print(f"✓ Committed to branch: {commit.message}")
    assert commit.branch_name == "experiment/new_algo"

    # Test branch listing
    branches = workspace.list_branches()
    print(f"✓ Workspace has {len(branches)} branches: {branches}")
    assert len(branches) == 2
    assert "main" in branches
    assert "experiment/new_algo" in branches

    print("✅ Workspace branching test passed!")
    return True


async def test_task_graph_dag():
    """Test the improved TaskGraph with DAG operations."""
    print("\n=== Testing TaskGraph (DAG Operations) ===")

    blackboard = MockBlackboard()
    graph = TaskGraph(blackboard)

    # Create root task
    root_task = Task(
        task_id="root",
        task_type="analysis",
        goal="Analyze system",
        description="Root analysis task",
        requesting_agent_id="coordinator"
    )
    await graph.add_task(root_task)
    print(f"✓ Added root task: {root_task.task_id}")

    # Create child tasks
    subtask1 = Task(
        task_id="subtask1",
        task_type="code_analysis",
        goal="Analyze module A",
        description="Analyze module A",
        requesting_agent_id="coordinator",
        parent_task_id="root",
        depends_on=[]
    )
    await graph.add_task(subtask1)

    subtask2 = Task(
        task_id="subtask2",
        task_type="code_analysis",
        goal="Analyze module B",
        description="Analyze module B",
        requesting_agent_id="coordinator",
        parent_task_id="root",
        depends_on=["subtask1"]  # Depends on subtask1
    )
    await graph.add_task(subtask2)

    print(f"✓ Added subtasks with dependencies")

    # Test cycle detection (should fail)
    try:
        cyclic_task = Task(
            task_id="cyclic",
            task_type="test",
            goal="Cyclic task",
            description="This would create a cycle",
            requesting_agent_id="coordinator",
            depends_on=["subtask2", "cyclic"]  # Self-dependency
        )
        await graph.add_task(cyclic_task)
        print("✗ Cycle detection failed!")
        assert False
    except ValueError as e:
        print(f"✓ Cycle detection worked: {e}")

    # Test progress tracking
    progress = await graph.get_task_progress("root")
    print(f"✓ Task progress: {progress['total']} tasks, {progress['progress_percentage']:.1f}% complete")
    assert progress["total"] == 3  # root + 2 subtasks

    # Test critical path
    critical_path = await graph.get_critical_path("root")
    print(f"✓ Critical path: {critical_path}")

    print("✅ TaskGraph DAG test passed!")
    return True


async def test_obligation_graph_traceability():
    """Test the enhanced ObligationGraph with traceability."""
    print("\n=== Testing ObligationGraph (Traceability) ===")

    blackboard = MockBlackboard()
    graph = ObligationGraph(blackboard)

    # Add requirement with traceability
    req = await graph.add_requirement(
        content={
            "requirement_id": "REQ_001",
            "title": "User Authentication",
            "description": "System must authenticate users securely"
        },
        title="User Authentication",
        description="Security requirement for user auth",
        tags=["security", "auth"],
        created_by="analyst_001"
    )

    print(f"✓ Added requirement: {req.title} (v{req.version})")
    assert req.version == 1

    # Add artifact
    artifact = await graph.add_artifact(
        content={
            "artifact_type": "code",
            "file": "auth.py",
            "function": "authenticate_user"
        },
        location={"file": "auth.py", "line": 42},
        title="Authentication Implementation",
        created_by="developer_001"
    )

    print(f"✓ Added artifact: {artifact.title}")

    # Link with compliance
    edge = await graph.link(
        requirement_id=req.node_id,
        artifact_id=artifact.node_id,
        relationship=ComplianceRelationship.SATISFIES,
        confidence=0.9,
        evidence=["Uses bcrypt for password hashing", "Implements 2FA"],
        derived_by="analyzer_001"
    )

    print(f"✓ Linked requirement to artifact: {edge.relationship.value}")

    # Test impact analysis
    impact = await graph.get_impact_analysis(
        node_id=req.node_id,
        change_type="modify"
    )
    print(f"✓ Impact analysis: {len(impact['directly_affected'])} artifacts affected")
    assert len(impact["directly_affected"]) == 1

    # Test coverage metrics
    metrics = await graph.get_coverage_metrics()
    print(f"✓ Coverage: {metrics['coverage_percentage']:.1f}% requirements covered")

    # Test requirement tracing
    trace = await graph.trace_requirement(req.node_id)
    print(f"✓ Traced requirement: {trace['compliance_status']}")
    assert trace["compliance_status"] == "satisfied"

    # Test versioning
    updated_req = await graph.update_node_version(
        node_id=req.node_id,
        changes={"description": "Updated: System must authenticate users with MFA"},
        agent_id="analyst_002",
        change_reason="Added MFA requirement"
    )

    print(f"✓ Updated requirement to v{updated_req.version}")
    assert updated_req.version == 2
    assert len(updated_req.change_history) > 0

    print("✅ ObligationGraph traceability test passed!")
    return True


async def test_causality_timeline_vector_clocks():
    """Test the improved CausalityTimeline with vector clocks."""
    print("\n=== Testing CausalityTimeline (Vector Clocks) ===")

    blackboard = MockBlackboard()
    timeline = CausalityTimeline(blackboard)

    # Create events with vector clocks
    event1 = CausalEvent(
        event_id="e1",
        event_type=EventType.LOCK_ACQUIRE,
        thread_id="thread1",
        resource_id="mutex_a",
        vector_clock=VectorClock(clock={"thread1": 1, "thread2": 0})
    )
    await timeline.add_event(event1, auto_infer_relations=False)
    print(f"✓ Added event e1: {event1.vector_clock.to_string()}")

    event2 = CausalEvent(
        event_id="e2",
        event_type=EventType.SHARED_WRITE,
        thread_id="thread2",
        resource_id="shared_var",
        vector_clock=VectorClock(clock={"thread1": 0, "thread2": 1})
    )
    await timeline.add_event(event2, auto_infer_relations=False)
    print(f"✓ Added event e2: {event2.vector_clock.to_string()}")

    # Test concurrent detection
    are_concurrent = event1.vector_clock.concurrent_with(event2.vector_clock)
    print(f"✓ Events e1 and e2 are concurrent: {are_concurrent}")
    assert are_concurrent

    # Create causally related event
    event3 = CausalEvent(
        event_id="e3",
        event_type=EventType.LOCK_RELEASE,
        thread_id="thread1",
        resource_id="mutex_a",
        vector_clock=VectorClock(clock={"thread1": 2, "thread2": 1})
    )
    # Simulate message passing (thread1 received message from thread2)
    event3.update_vector_clock("thread1", event2.vector_clock)
    await timeline.add_event(event3, auto_infer_relations=False)
    print(f"✓ Added event e3: {event3.vector_clock.to_string()}")

    # Test happens-before
    happens_before = event2.vector_clock.happens_before(event3.vector_clock)
    print(f"✓ Event e2 happens-before e3: {happens_before}")
    assert happens_before

    # Test concurrent sets
    events = [event1, event2, event3]
    concurrent_sets = await timeline.find_concurrent_sets(events)
    print(f"✓ Found {len(concurrent_sets)} concurrent sets")

    # Test critical section analysis
    critical_sections = await timeline.analyze_critical_sections("mutex_a")
    print(f"✓ Analyzed critical sections for mutex_a: {len(critical_sections)} sections")

    print("✅ CausalityTimeline vector clock test passed!")
    return True


async def test_integration_with_patterns():
    """Test integration with domain-agnostic patterns."""
    print("\n=== Testing Integration with Patterns ===")

    blackboard = MockBlackboard()

    # Create a task that returns ScopeAwareResult
    task_graph = TaskGraph(blackboard)

    task = Task(
        task_id="analysis_task",
        task_type="code_analysis",
        goal="Analyze authentication module",
        description="Deep analysis of auth implementation",
        requesting_agent_id="analyzer_001"
    )

    await task_graph.add_task(task)
    print(f"✓ Created analysis task: {task.task_id}")

    # Simulate task completion with ScopeAwareResult
    result = ScopeAwareResult(
        scope=AnalysisScope(
            is_complete=False,
            missing_context=["database_schema.sql", "config.yml"],
            confidence=0.75,
            reasoning=["Analyzed main auth flow", "Missing DB interaction details"]
        ),
        content={
            "vulnerabilities": [],
            "recommendations": ["Add rate limiting", "Implement session timeout"],
            "coverage": 0.8
        }
    )

    # Complete task with scope-aware result
    await task_graph.complete_task(
        task_id=task.task_id,
        result=result,
        agent_id="analyzer_001"
    )

    print(f"✓ Completed task with ScopeAwareResult:")
    print(f"  - Complete: {result.is_complete()}")
    print(f"  - Confidence: {result.scope.confidence}")
    print(f"  - Missing context: {len(result.get_missing_context())} items")

    # Verify task was updated with scope
    updated_task = await task_graph.get_task(task.task_id)
    assert updated_task is not None
    assert updated_task.status == TaskStatus.COMPLETED
    print(f"✓ Task scope updated from result")

    print("✅ Pattern integration test passed!")
    return True


async def main():
    """Run all integration tests."""
    print("=" * 60)
    print("BLACKBOARD INTEGRATION TESTS")
    print("=" * 60)

    tests = [
        test_workspace_branching,
        test_task_graph_dag,
        test_obligation_graph_traceability,
        test_causality_timeline_vector_clocks,
        test_integration_with_patterns,
    ]

    results = []
    for test in tests:
        try:
            result = await test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"\n❌ {test.__name__} failed with error: {e}")
            results.append((test.__name__, False))

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All integration tests passed!")
    else:
        print(f"\n⚠️ {total - passed} tests failed")

    return passed == total


if __name__ == "__main__":
    asyncio.run(main())
