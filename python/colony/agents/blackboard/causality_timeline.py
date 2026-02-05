"""CausalityTimeline for concurrency analysis.

As mentioned in AGENT_FRAMEWORK.md:
"The CausalityTimeline is a data structure that converts interleavings
(trace snippets, lock sequences, async spans) into Causality Timelines
summarizing dependencies and potential conflicts."

This enables qualitative concurrency analysis:
- Temporal event ordering
- Happens-before relationships
- Concurrent execution detection
- Race condition identification
- Deadlock pattern recognition

The CausalityTimeline is generalizable beyond code to:
- Distributed system event correlation
- Workflow execution tracking
- Multi-agent action coordination
- Temporal reasoning about causality
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class VectorClock(BaseModel):
    """Vector clock for distributed causality tracking.

    Vector clocks provide a partial ordering of events in a distributed system,
    allowing us to determine happens-before relationships without synchronized clocks.

    Each process/thread maintains its own counter in the vector.
    """

    clock: dict[str, int] = Field(
        default_factory=dict,
        description="Process ID -> logical timestamp"
    )

    def increment(self, process_id: str) -> None:
        """Increment clock for a process.

        Args:
            process_id: Process/thread ID
        """
        self.clock[process_id] = self.clock.get(process_id, 0) + 1

    def update(self, other: VectorClock) -> None:
        """Update clock with max of self and other.

        Used when receiving a message - take max of each component.

        Args:
            other: Other vector clock
        """
        for pid, timestamp in other.clock.items():
            self.clock[pid] = max(self.clock.get(pid, 0), timestamp)

    def happens_before(self, other: VectorClock) -> bool:
        """Check if self happens-before other.

        A happens-before B if:
        - All components of A <= corresponding components of B
        - At least one component of A < corresponding component of B

        Args:
            other: Other vector clock

        Returns:
            True if self happens-before other
        """
        all_leq = True  # All components <=
        exists_lt = False  # At least one component <

        # Check all processes in both clocks
        all_processes = set(self.clock.keys()) | set(other.clock.keys())

        for pid in all_processes:
            self_val = self.clock.get(pid, 0)
            other_val = other.clock.get(pid, 0)

            if self_val > other_val:
                all_leq = False
                break
            if self_val < other_val:
                exists_lt = True

        return all_leq and exists_lt

    def concurrent_with(self, other: VectorClock) -> bool:
        """Check if events are concurrent.

        Events are concurrent if neither happens-before the other.

        Args:
            other: Other vector clock

        Returns:
            True if concurrent
        """
        return not self.happens_before(other) and not other.happens_before(self)

    def to_string(self) -> str:
        """String representation of vector clock.

        Returns:
            String representation
        """
        items = sorted(self.clock.items())
        return f"VC({', '.join(f'{k}:{v}' for k, v in items)})"


class EventType(str, Enum):
    """Type of concurrent event."""

    LOCK_ACQUIRE = "lock_acquire"
    LOCK_RELEASE = "lock_release"
    SHARED_READ = "shared_read"
    SHARED_WRITE = "shared_write"
    MESSAGE_SEND = "message_send"
    MESSAGE_RECEIVE = "message_receive"
    SPAWN_THREAD = "spawn_thread"
    JOIN_THREAD = "join_thread"
    ASYNC_START = "async_start"
    ASYNC_COMPLETE = "async_complete"
    BARRIER_WAIT = "barrier_wait"
    CONDITION_WAIT = "condition_wait"
    CONDITION_SIGNAL = "condition_signal"


class CausalRelation(str, Enum):
    """Causal relationship between events."""

    BEFORE = "before"  # Event 1 definitely happens before event 2
    AFTER = "after"  # Event 1 definitely happens after event 2
    CONCURRENT = "concurrent"  # Events may happen concurrently
    UNKNOWN = "unknown"  # Relationship unknown
    CONFLICT = "conflict"  # Events conflict (potential race)


class CausalEvent(BaseModel):
    """An event in the causality timeline with vector clock.

    Examples:
        Lock acquisition:
        ```python
        event = CausalEvent(
            event_type=EventType.LOCK_ACQUIRE,
            thread_id="thread_1",
            resource_id="config_lock",
            location=CodeLocation(file="app.py", line=42),
            timestamp=1234567890.123,
            vector_clock=VectorClock(clock={"thread_1": 5, "thread_2": 3})
        )
        ```

        Shared memory write:
        ```python
        event = CausalEvent(
            event_type=EventType.SHARED_WRITE,
            thread_id="thread_2",
            resource_id="shared_counter",
            location=CodeLocation(file="worker.py", line=15),
            timestamp=1234567890.456,
            vector_clock=VectorClock(clock={"thread_1": 4, "thread_2": 7}),
            metadata={"variable": "counter", "value": 42}
        )
        ```
    """

    event_id: str = Field(
        default_factory=lambda: f"event_{uuid.uuid4().hex}",
        description="Unique event identifier"
    )

    event_type: EventType = Field(
        description="Type of event"
    )

    # Context
    thread_id: str = Field(
        description="Thread/goroutine/async task ID"
    )

    resource_id: str | None = Field(
        default=None,
        description="Resource involved (lock name, shared variable, etc.)"
    )

    # Location
    location: dict[str, Any] | None = Field(
        default=None,
        description="Code location where event occurs"
    )

    # Timing
    timestamp: float = Field(
        default_factory=time.time,
        description="Event timestamp (absolute or relative)"
    )

    duration: float | None = Field(
        default=None,
        description="Event duration (for spans)"
    )

    # Vector clock for distributed causality
    vector_clock: VectorClock = Field(
        default_factory=VectorClock,
        description="Vector clock at time of event"
    )

    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional event metadata (variable names, values, etc.)"
    )

    # Relationship tracking (computed from vector clocks)
    happens_after: list[str] = Field(
        default_factory=list,
        description="Event IDs that happen before this event"
    )

    happens_before: list[str] = Field(
        default_factory=list,
        description="Event IDs that happen after this event"
    )

    concurrent_with: list[str] = Field(
        default_factory=list,
        description="Event IDs potentially concurrent with this"
    )

    def update_vector_clock(self, thread_id: str, received_clock: VectorClock | None = None) -> None:
        """Update vector clock for this event.

        Args:
            thread_id: Thread performing the event
            received_clock: Optional received vector clock (for message events)
        """
        if received_clock:
            # Receiving a message: update with max, then increment
            self.vector_clock.update(received_clock)

        # Increment own component
        self.vector_clock.increment(thread_id)


class CausalityMatrix(BaseModel):
    """Adjacency matrix for causal relationships.

    Stores happens-before relationships as specified in the framework:
    "TemporalCausalityMatrix: adjacency matrix stored on blackboard linking
    events with relation labels (before, after, unknown, conflict)."
    """

    matrix_id: str = Field(
        default_factory=lambda: f"matrix_{uuid.uuid4().hex}",
        description="Matrix identifier"
    )

    # Event IDs in the matrix
    events: list[str] = Field(
        default_factory=list,
        description="Event IDs in order"
    )

    # Relations: (event1_id, event2_id) -> relation
    relations: dict[tuple[str, str], CausalRelation] = Field(
        default_factory=dict,
        description="Causal relations between event pairs"
    )

    def add_event(self, event_id: str) -> None:
        """Add event to matrix.

        Args:
            event_id: Event ID
        """
        if event_id not in self.events:
            self.events.append(event_id)

    def set_relation(
        self,
        event1_id: str,
        event2_id: str,
        relation: CausalRelation
    ) -> None:
        """Set causal relation between events.

        Args:
            event1_id: First event ID
            event2_id: Second event ID
            relation: Causal relation
        """
        self.relations[(event1_id, event2_id)] = relation

        # Set symmetric relation
        if relation == CausalRelation.BEFORE:
            self.relations[(event2_id, event1_id)] = CausalRelation.AFTER
        elif relation == CausalRelation.AFTER:
            self.relations[(event2_id, event1_id)] = CausalRelation.BEFORE
        elif relation == CausalRelation.CONCURRENT:
            self.relations[(event2_id, event1_id)] = CausalRelation.CONCURRENT

    def get_relation(self, event1_id: str, event2_id: str) -> CausalRelation:
        """Get causal relation between events.

        Args:
            event1_id: First event ID
            event2_id: Second event ID

        Returns:
            Causal relation
        """
        return self.relations.get((event1_id, event2_id), CausalRelation.UNKNOWN)


class ConcurrencyConflict(BaseModel):
    """A potential concurrency bug."""

    conflict_id: str = Field(
        default_factory=lambda: f"conflict_{uuid.uuid4().hex}",
        description="Conflict identifier"
    )

    conflict_type: str = Field(
        description="Type: 'data_race', 'deadlock', 'atomicity_violation', 'order_violation'"
    )

    resource_id: str = Field(
        description="Shared resource involved"
    )

    events: list[str] = Field(
        default_factory=list,
        description="Event IDs involved in conflict"
    )

    threads: list[str] = Field(
        default_factory=list,
        description="Thread IDs involved"
    )

    severity: str = Field(
        default="medium",
        description="Severity: 'critical', 'high', 'medium', 'low'"
    )

    confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence this is a real bug"
    )

    description: str = Field(
        description="Description of the conflict"
    )

    interleaving: str | None = Field(
        default=None,
        description="Problematic interleaving description"
    )

    suggestion: str | None = Field(
        default=None,
        description="Suggested fix"
    )


class CausalityTimeline:
    """Manages causality timeline for concurrency analysis with vector clocks.

    Provides:
    - Event tracking with vector clocks
    - Precise causal relationship inference
    - Race condition and deadlock detection
    - Lamport's happens-before relation
    - Timeline queries and visualization
    """

    def __init__(self, blackboard: Any):
        """Initialize timeline.

        Args:
            blackboard: EnhancedBlackboard instance
        """
        self.blackboard = blackboard
        self.event_namespace = "causality_event"
        self.matrix_namespace = "causality_matrix"
        self.conflict_namespace = "causality_conflict"
        self._event_cache: dict[str, CausalEvent] = {}  # Cache for performance
        self._thread_clocks: dict[str, VectorClock] = defaultdict(VectorClock)  # Per-thread vector clocks

    async def add_event(
        self,
        event: CausalEvent,
        auto_infer_relations: bool = True
    ) -> None:
        """Add event to timeline.

        Args:
            event: Event to add
            auto_infer_relations: Whether to auto-infer causal relations
        """
        # Store event
        await self._store_event(event)

        # Infer causal relations if requested
        if auto_infer_relations:
            await self._infer_causal_relations(event)

    async def add_relation(
        self,
        event1_id: str,
        event2_id: str,
        relation: CausalRelation,
        confidence: float = 1.0
    ) -> None:
        """Add causal relation between events.

        Args:
            event1_id: First event ID
            event2_id: Second event ID
            relation: Causal relation
            confidence: Confidence in relation
        """
        key = f"causality_relation:{event1_id}:{event2_id}"

        await self.blackboard.write(
            key=key,
            value={
                "event1_id": event1_id,
                "event2_id": event2_id,
                "relation": relation.value,
                "confidence": confidence
            },
            tags={"causality_relation", relation.value, event1_id, event2_id}
        )

    async def detect_conflicts(
        self,
        resource_id: str | None = None
    ) -> list[ConcurrencyConflict]:
        """Detect concurrency conflicts.

        Args:
            resource_id: Optional filter by resource

        Returns:
            List of detected conflicts
        """
        conflicts = []

        # Get all events (or filtered by resource)
        events = await self._get_events(resource_id=resource_id)

        # Detect data races (concurrent writes or write-read to same resource)
        race_conflicts = await self._detect_data_races(events)
        conflicts.extend(race_conflicts)

        # Detect deadlocks (circular lock dependencies)
        deadlock_conflicts = await self._detect_deadlocks(events)
        conflicts.extend(deadlock_conflicts)

        # Store conflicts
        for conflict in conflicts:
            await self._store_conflict(conflict)

        return conflicts

    async def get_event_sequence(
        self,
        thread_id: str
    ) -> list[CausalEvent]:
        """Get chronological sequence of events for a thread.

        Args:
            thread_id: Thread ID

        Returns:
            List of events in chronological order
        """
        events = await self._get_events(thread_id=thread_id)
        return sorted(events, key=lambda e: e.timestamp)

    async def _infer_causal_relations(self, event: CausalEvent) -> None:
        """Infer causal relations for a new event.

        Args:
            event: New event
        """
        # Get events in same thread (they have causal order)
        thread_events = await self.get_event_sequence(event.thread_id)

        for prev_event in thread_events:
            if prev_event.timestamp < event.timestamp:
                # Previous event happens before this one
                await self.add_relation(
                    prev_event.event_id,
                    event.event_id,
                    CausalRelation.BEFORE
                )

        # Infer from synchronization primitives
        if event.event_type == EventType.MESSAGE_RECEIVE:
            # Find corresponding send event
            # That send happens-before this receive
            pass

        if event.event_type == EventType.LOCK_RELEASE:
            # Next lock acquire on same lock happens-after this
            pass

    async def _detect_data_races(
        self,
        events: list[CausalEvent]
    ) -> list[ConcurrencyConflict]:
        """Detect potential data races.

        Args:
            events: Events to analyze

        Returns:
            List of race conflicts
        """
        races = []

        # Group events by resource
        by_resource: dict[str, list[CausalEvent]] = {}
        for event in events:
            if event.resource_id:
                if event.resource_id not in by_resource:
                    by_resource[event.resource_id] = []
                by_resource[event.resource_id].append(event)

        # Check each resource for races
        for resource_id, resource_events in by_resource.items():
            # Find concurrent writes or concurrent write-read
            writes = [e for e in resource_events if e.event_type == EventType.SHARED_WRITE]
            reads = [e for e in resource_events if e.event_type == EventType.SHARED_READ]

            # Check for concurrent writes
            for i, write1 in enumerate(writes):
                for write2 in writes[i+1:]:
                    if await self._are_concurrent(write1, write2):
                        races.append(ConcurrencyConflict(
                            conflict_type="data_race",
                            resource_id=resource_id,
                            events=[write1.event_id, write2.event_id],
                            threads=[write1.thread_id, write2.thread_id],
                            severity="high",
                            confidence=0.8,
                            description=f"Concurrent writes to {resource_id} from {write1.thread_id} and {write2.thread_id}",
                            suggestion="Add synchronization (lock, mutex, etc.)"
                        ))

            # Check for concurrent write-read
            for write in writes:
                for read in reads:
                    if write.thread_id != read.thread_id and await self._are_concurrent(write, read):
                        races.append(ConcurrencyConflict(
                            conflict_type="data_race",
                            resource_id=resource_id,
                            events=[write.event_id, read.event_id],
                            threads=[write.thread_id, read.thread_id],
                            severity="medium",
                            confidence=0.7,
                            description=f"Concurrent write-read to {resource_id}",
                            suggestion="Ensure proper synchronization"
                        ))

        return races

    async def _detect_deadlocks(
        self,
        events: list[CausalEvent]
    ) -> list[ConcurrencyConflict]:
        """Detect potential deadlocks.

        Args:
            events: Events to analyze

        Returns:
            List of deadlock conflicts
        """
        deadlocks = []

        # Build lock acquisition graph
        # Thread -> locks held in order
        thread_lock_sequences: dict[str, list[tuple[float, str]]] = {}

        for event in events:
            if event.event_type == EventType.LOCK_ACQUIRE and event.resource_id:
                if event.thread_id not in thread_lock_sequences:
                    thread_lock_sequences[event.thread_id] = []
                thread_lock_sequences[event.thread_id].append((event.timestamp, event.resource_id))

        # Look for circular dependencies
        # Thread A: lock1 -> lock2
        # Thread B: lock2 -> lock1
        # = potential deadlock

        for thread1, seq1 in thread_lock_sequences.items():
            for thread2, seq2 in thread_lock_sequences.items():
                if thread1 >= thread2:
                    continue

                # Check if lock orderings are reversed
                if len(seq1) >= 2 and len(seq2) >= 2:
                    # Simplistic check for now
                    locks1 = [lock for _, lock in seq1]
                    locks2 = [lock for _, lock in seq2]

                    # Check for any reversed pair
                    for i in range(len(locks1) - 1):
                        lock_a = locks1[i]
                        lock_b = locks1[i + 1]

                        # Check if thread2 acquires in reverse order
                        if lock_b in locks2 and lock_a in locks2:
                            idx_a = locks2.index(lock_a)
                            idx_b = locks2.index(lock_b)

                            if idx_b < idx_a:  # Reversed order
                                deadlocks.append(ConcurrencyConflict(
                                    conflict_type="deadlock",
                                    resource_id=f"{lock_a},{lock_b}",
                                    events=[],  # Would include specific event IDs
                                    threads=[thread1, thread2],
                                    severity="critical",
                                    confidence=0.6,
                                    description=f"Potential deadlock: {thread1} acquires {lock_a} then {lock_b}, "
                                                f"{thread2} acquires {lock_b} then {lock_a}",
                                    suggestion="Enforce consistent lock ordering"
                                ))

        return deadlocks

    async def _are_concurrent(
        self,
        event1: CausalEvent,
        event2: CausalEvent
    ) -> bool:
        """Check if two events are concurrent using vector clocks.

        Args:
            event1: First event
            event2: Second event

        Returns:
            True if concurrent
        """
        # Same thread events are always ordered
        if event1.thread_id == event2.thread_id:
            return False

        # Use vector clocks for precise causality
        return event1.vector_clock.concurrent_with(event2.vector_clock)

    async def _get_relation(
        self,
        event1_id: str,
        event2_id: str
    ) -> CausalRelation:
        """Get causal relation between events.

        Args:
            event1_id: First event ID
            event2_id: Second event ID

        Returns:
            Causal relation
        """
        key = f"causality_relation:{event1_id}:{event2_id}"
        data = await self.blackboard.read(key)

        if data:
            return CausalRelation(data.get("relation", "unknown"))

        return CausalRelation.UNKNOWN

    async def _store_event(self, event: CausalEvent) -> None:
        """Store event in blackboard.

        Args:
            event: Event to store
        """
        key = f"{self.event_namespace}:{event.event_id}"

        await self.blackboard.write(
            key=key,
            value=event.model_dump(),
            tags={
                "causality_event",
                event.event_type.value,
                event.thread_id,
                event.resource_id or "no_resource"
            }
        )

    async def _store_conflict(self, conflict: ConcurrencyConflict) -> None:
        """Store conflict in blackboard.

        Args:
            conflict: Conflict to store
        """
        key = f"{self.conflict_namespace}:{conflict.conflict_id}"

        await self.blackboard.write(
            key=key,
            value=conflict.model_dump(),
            tags={
                "concurrency_conflict",
                conflict.conflict_type,
                conflict.severity,
                conflict.resource_id
            }
        )

    async def _get_events(
        self,
        thread_id: str | None = None,
        resource_id: str | None = None,
        event_type: EventType | None = None
    ) -> list[CausalEvent]:
        """Get events, optionally filtered.

        Args:
            thread_id: Optional thread filter
            resource_id: Optional resource filter
            event_type: Optional event type filter

        Returns:
            List of events
        """
        # Build tags for filtering
        tags = {"causality_event"}
        if thread_id:
            tags.add(thread_id)
        if resource_id:
            tags.add(resource_id)
        if event_type:
            tags.add(event_type.value)

        # Query blackboard
        entries = await self.blackboard.query(
            namespace=self.event_namespace,
            tags=tags,
            limit=10000  # Large limit to get all matching events
        )

        events = []
        for entry in entries:
            try:
                event = CausalEvent(**entry.value)
                # Double-check filters (in case tags are stale)
                if thread_id and event.thread_id != thread_id:
                    continue
                if resource_id and event.resource_id != resource_id:
                    continue
                if event_type and event.event_type != event_type:
                    continue

                events.append(event)
                self._event_cache[event.event_id] = event
            except Exception:
                continue

        return events

    async def compute_happens_before_graph(
        self,
        events: list[CausalEvent] | None = None
    ) -> dict[str, list[str]]:
        """Compute the happens-before graph using vector clocks.

        Args:
            events: Events to analyze (None = all events)

        Returns:
            Adjacency list of happens-before relations
        """
        if events is None:
            events = await self._get_events()

        # Build happens-before graph
        happens_before = defaultdict(list)

        for i, event1 in enumerate(events):
            for j, event2 in enumerate(events):
                if i != j:
                    if event1.vector_clock.happens_before(event2.vector_clock):
                        happens_before[event1.event_id].append(event2.event_id)

        return dict(happens_before)

    async def find_concurrent_sets(
        self,
        events: list[CausalEvent] | None = None
    ) -> list[set[str]]:
        """Find sets of potentially concurrent events.

        Args:
            events: Events to analyze (None = all events)

        Returns:
            List of concurrent event sets
        """
        if events is None:
            events = await self._get_events()

        # Group events by resource
        by_resource = defaultdict(list)
        for event in events:
            if event.resource_id:
                by_resource[event.resource_id].append(event)

        concurrent_sets = []

        # Find concurrent events per resource
        for resource_id, resource_events in by_resource.items():
            # Build concurrent sets for this resource
            visited = set()

            for i, event1 in enumerate(resource_events):
                if event1.event_id in visited:
                    continue

                concurrent_set = {event1.event_id}
                visited.add(event1.event_id)

                for j, event2 in enumerate(resource_events[i+1:], i+1):
                    if event2.event_id not in visited:
                        if event1.vector_clock.concurrent_with(event2.vector_clock):
                            concurrent_set.add(event2.event_id)
                            visited.add(event2.event_id)

                if len(concurrent_set) > 1:
                    concurrent_sets.append(concurrent_set)

        return concurrent_sets

    async def detect_happened_before_violations(
        self,
        expected_orderings: list[tuple[str, str]]
    ) -> list[dict[str, Any]]:
        """Detect violations of expected happens-before orderings.

        Useful for verifying synchronization correctness.

        Args:
            expected_orderings: List of (event1_id, event2_id) that should be ordered

        Returns:
            List of violations
        """
        violations = []

        for event1_id, event2_id in expected_orderings:
            event1 = await self.get_event(event1_id)
            event2 = await self.get_event(event2_id)

            if not event1 or not event2:
                violations.append({
                    "type": "missing_event",
                    "event1": event1_id,
                    "event2": event2_id,
                    "message": "One or both events not found"
                })
                continue

            if not event1.vector_clock.happens_before(event2.vector_clock):
                if event1.vector_clock.concurrent_with(event2.vector_clock):
                    violations.append({
                        "type": "concurrent_execution",
                        "event1": event1_id,
                        "event2": event2_id,
                        "message": f"Events are concurrent but should be ordered",
                        "event1_clock": event1.vector_clock.to_string(),
                        "event2_clock": event2.vector_clock.to_string()
                    })
                elif event2.vector_clock.happens_before(event1.vector_clock):
                    violations.append({
                        "type": "reversed_order",
                        "event1": event1_id,
                        "event2": event2_id,
                        "message": f"Events are in reversed order",
                        "event1_clock": event1.vector_clock.to_string(),
                        "event2_clock": event2.vector_clock.to_string()
                    })

        return violations

    async def get_event(self, event_id: str) -> CausalEvent | None:
        """Get event by ID.

        Args:
            event_id: Event ID

        Returns:
            Event or None
        """
        # Check cache first
        if event_id in self._event_cache:
            return self._event_cache[event_id]

        # Fetch from blackboard
        key = f"{self.event_namespace}:{event_id}"
        data = await self.blackboard.read(key)

        if data:
            event = CausalEvent(**data)
            self._event_cache[event_id] = event
            return event

        return None

    async def analyze_critical_sections(
        self,
        lock_resource_id: str
    ) -> list[dict[str, Any]]:
        """Analyze critical sections protected by a lock.

        Args:
            lock_resource_id: Lock resource identifier

        Returns:
            Critical section analysis
        """
        # Get all lock events for this resource
        lock_events = await self._get_events(resource_id=lock_resource_id)

        # Sort by timestamp
        lock_events.sort(key=lambda e: e.timestamp)

        critical_sections = []
        current_holder = None
        acquire_event = None

        for event in lock_events:
            if event.event_type == EventType.LOCK_ACQUIRE:
                if current_holder:
                    # Nested or error
                    critical_sections.append({
                        "type": "nested_acquire",
                        "thread": event.thread_id,
                        "timestamp": event.timestamp,
                        "current_holder": current_holder,
                        "warning": "Possible nested lock or missing release"
                    })
                else:
                    current_holder = event.thread_id
                    acquire_event = event

            elif event.event_type == EventType.LOCK_RELEASE:
                if current_holder == event.thread_id:
                    # Valid critical section
                    if acquire_event:
                        duration = event.timestamp - acquire_event.timestamp
                        critical_sections.append({
                            "type": "critical_section",
                            "thread": current_holder,
                            "acquire_time": acquire_event.timestamp,
                            "release_time": event.timestamp,
                            "duration": duration,
                            "acquire_clock": acquire_event.vector_clock.to_string(),
                            "release_clock": event.vector_clock.to_string()
                        })
                    current_holder = None
                    acquire_event = None
                else:
                    # Release without acquire
                    critical_sections.append({
                        "type": "unmatched_release",
                        "thread": event.thread_id,
                        "timestamp": event.timestamp,
                        "warning": "Release without matching acquire"
                    })

        # Check for unreleased lock
        if current_holder:
            critical_sections.append({
                "type": "unreleased_lock",
                "thread": current_holder,
                "acquire_time": acquire_event.timestamp if acquire_event else None,
                "warning": "Lock acquired but never released"
            })

        return critical_sections


# Utility functions

async def build_causality_matrix_from_events(
    events: list[CausalEvent]
) -> CausalityMatrix:
    """Build causality matrix from events.

    Args:
        events: Events to analyze

    Returns:
        Causality matrix
    """
    matrix = CausalityMatrix()

    # Add all events
    for event in events:
        matrix.add_event(event.event_id)

    # Infer causal relations
    # Events in same thread have program order
    by_thread: dict[str, list[CausalEvent]] = {}
    for event in events:
        if event.thread_id not in by_thread:
            by_thread[event.thread_id] = []
        by_thread[event.thread_id].append(event)

    # Add happens-before for events in same thread
    for thread_id, thread_events in by_thread.items():
        sorted_events = sorted(thread_events, key=lambda e: e.timestamp)
        for i in range(len(sorted_events) - 1):
            matrix.set_relation(
                sorted_events[i].event_id,
                sorted_events[i+1].event_id,
                CausalRelation.BEFORE
            )

    # Add happens-before from synchronization
    # Message send happens-before message receive
    # Lock release happens-before next lock acquire
    # etc.

    return matrix


async def analyze_concurrent_execution(
    trace_snippets: list[dict[str, Any]],
    blackboard: Any,
    llm_client: Any
) -> list[ConcurrencyConflict]:
    """Analyze concurrent execution from trace snippets.

    This is a utility function that would use an LLM to build causality narrative.
    For now, it creates basic events from trace data and detects conflicts.

    DO NOT USE THIS FUNCTION. IT IS ONLY FOR DEMONSTRATION PURPOSES.

    Args:
        trace_snippets: Execution trace snippets with event data
        blackboard: Blackboard instance
        llm_client: LLM for analysis (not used in basic version)

    Returns:
        List of conflicts
    """
    timeline = CausalityTimeline(blackboard)

    # Convert trace snippets to events
    # Basic implementation: assume traces have event data
    for snippet in trace_snippets:
        try:
            # Create event from snippet
            event = CausalEvent(
                event_type=EventType(snippet.get("event_type", "shared_write")),
                thread_id=snippet.get("thread_id", "unknown"),
                resource_id=snippet.get("resource_id"),
                timestamp=snippet.get("timestamp", time.time()),
                metadata=snippet.get("metadata", {})
            )
            await timeline.add_event(event, auto_infer_relations=True)
        except Exception:
            # Skip invalid traces
            continue

    # Detect conflicts
    conflicts = await timeline.detect_conflicts()

    return conflicts

