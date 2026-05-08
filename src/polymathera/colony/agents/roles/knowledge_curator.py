"""``KnowledgeCuratorAgent`` + ``KnowledgeCuratorCapability``.

Per master §3.5 the KnowledgeCuratorAgent is the colony-generic role
that owns the knowledge corpus: source ingestion, KG maintenance,
citation-integrity CI, and the sampled-human-review queue (master
§3.2). The capability wraps Phase C1a's ``Ingestor`` as
``@action_executor`` methods so an LLM-planned action policy can
ingest content end-to-end without colony-specific glue, and surfaces
the review queue to whatever consumer (the SessionAgent's `/review`
command, an ops dashboard, a backfill script) wants to drain it.

When a Phase C5 design monorepo is configured, the curator can also
mirror ingested literature into the monorepo's ``corpora/`` (master
§6.6.1). The mirror path is opt-in: the constructor takes a
``DesignMonorepoClient`` (or any object with ``commit_with_identity``
+ ``working_dir``); when absent the mirror methods return a typed
"not configured" result instead of failing.

When a Phase C4 convergence runtime is configured, every successful
ingestion fires a ``PageChangeEvent`` so subscribers re-fire on the
new content (master §5.6 transport). The runtime handle is again
optional — tests pass a list-collecting fake.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from overrides import override
from pydantic import BaseModel, ConfigDict, Field

from ...knowledge import (
    Chunk,
    CorpusTier,
    Ingestor,
    IngestionRecord,
    IngestionStatus,
    KnowledgeFormat,
)
from ...vcm.page_events import PageChangeEvent
from ..base import Agent, AgentCapability
from ..blueprint import Blueprint
from ..models import AgentSuspensionState
from ..patterns.actions import action_executor


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Review-queue items + callbacks
# ---------------------------------------------------------------------------


class ReviewItem(BaseModel):
    """One item awaiting sampled human review."""

    model_config = ConfigDict(frozen=False)

    record: IngestionRecord
    chunk_summaries: tuple[str, ...] = Field(default_factory=tuple)
    """Truncated text of each chunk produced (≤200 chars). Avoids
    storing the full chunk corpus inline."""

    queued_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    resolved: bool = False
    resolution: str = ""
    """Free-form: ``"approved"`` / ``"rejected"`` / ``"needs_redo"``,
    or any deployment-defined label."""

    resolved_by: str = ""
    resolved_at: datetime | None = None


PageEventEmitter = Callable[[PageChangeEvent], Awaitable[None]]
"""Callback that emits a ``PageChangeEvent`` into the convergence
runtime. Wired to ``ConvergenceCapability.dispatch_change`` (or, for
non-capability deployments, directly to
``ConvergenceRuntimeDeployment.feed_page_event``) by the deployment."""


class _DesignMonorepoMirror:
    """Adapter shape the curator expects when mirroring corpora into
    a design monorepo. Defined as a Protocol-style typed shim rather
    than importing ``DesignMonorepoClient`` directly so the knowledge
    layer doesn't depend on ``polymathera.colony.design_monorepo``."""

    working_dir: Path

    def commit_with_identity(
        self,
        identity: Any,
        message: str,
        *,
        paths: Sequence[Path] | None = None,
        all_changes: bool = False,
    ) -> str:
        ...


# ---------------------------------------------------------------------------
# KnowledgeCuratorCapability
# ---------------------------------------------------------------------------


class KnowledgeCuratorCapability(AgentCapability):
    """Wraps the C1a ``Ingestor`` + review queue + KG-maintenance
    actions as ``@action_executor`` methods.

    The capability owns:

    - the ``Ingestor`` instance (one per capability — the underlying
      stores / embedder are shared across capability instances when
      they share the same ``Ingestor`` reference),
    - the in-memory review queue (events for the SessionAgent are
      published when an item is queued or resolved),
    - the optional design-monorepo mirror,
    - the optional page-event emitter.
    """

    DEFAULT_REVIEW_QUEUE_PATTERN = "knowledge_curator:review_queue"

    def __init__(
        self,
        agent: Agent | None = None,
        scope_id: str | None = None,
        *,
        ingestor: Ingestor | Blueprint,
        page_event_emitter: PageEventEmitter | None = None,
        design_monorepo: _DesignMonorepoMirror | None = None,
        design_monorepo_identity: Any | None = None,
        capability_key: str | None = None,
        app_name: str | None = None,
    ) -> None:
        # ``ingestor`` accepts either a real :class:`Ingestor` (tests
        # / in-process) or a :class:`Blueprint` for it (cross-Ray
        # construction via ``default_ingestor_blueprint()``); same
        # pattern as ``BulkAcquisitionCapability``.
        super().__init__(
            agent=agent,
            scope_id=scope_id,
            input_patterns=[],
            capability_key=capability_key,
            app_name=app_name,
        )
        self._ingestor = (
            ingestor.local_instance() if isinstance(ingestor, Blueprint) else ingestor
        )
        self._page_event_emitter = page_event_emitter
        self._design_monorepo = design_monorepo
        self._design_monorepo_identity = design_monorepo_identity
        self._review_queue: dict[str, ReviewItem] = {}

        # The ingestor's review-queue callback writes into the
        # capability's own queue + fires a page event.
        # We attach this here so callers don't have to remember.
        self._wire_review_callback()

    def get_capability_tags(self) -> frozenset[str]:
        return frozenset({"knowledge", "ingestion", "curation", "review"})

    # ---- Ingestion -----------------------------------------------------

    @action_executor(
        planning_summary=(
            "Ingest a file into the knowledge corpus (auto-detect format)."
        ),
    )
    async def ingest_file(
        self,
        path: str,
        *,
        tier: CorpusTier = CorpusTier.UNTIERED,
        data_type_override: str | None = None,
        source_uri: str | None = None,
    ) -> IngestionRecord:
        record = await self._ingestor.ingest_file(
            path,
            tier=tier,
            data_type_override=data_type_override,
            source_uri=source_uri,
        )
        await self._maybe_emit_event(record)
        return record

    @action_executor(
        planning_summary="Ingest in-memory text into the knowledge corpus.",
    )
    async def ingest_text(
        self,
        text: str,
        *,
        source_uri: str | None = None,
        fmt: KnowledgeFormat = KnowledgeFormat.PLAIN_TEXT,
        tier: CorpusTier = CorpusTier.UNTIERED,
        data_type_override: str | None = None,
    ) -> IngestionRecord:
        record = await self._ingestor.ingest_text(
            text,
            source_uri=source_uri,
            fmt=fmt,
            tier=tier,
            data_type_override=data_type_override,
        )
        await self._maybe_emit_event(record)
        return record

    # ---- Review queue --------------------------------------------------

    @action_executor(
        planning_summary="List unresolved review items (oldest first).",
    )
    async def list_review_queue(
        self, *, include_resolved: bool = False, limit: int = 50,
    ) -> list[ReviewItem]:
        items = sorted(
            self._review_queue.values(),
            key=lambda it: it.queued_at,
        )
        if not include_resolved:
            items = [it for it in items if not it.resolved]
        return items[:limit]

    @action_executor(
        planning_summary="Resolve a queued review item with a verdict.",
    )
    async def resolve_review_item(
        self,
        record_id: str,
        resolution: str,
        *,
        resolved_by: str = "",
    ) -> ReviewItem:
        item = self._review_queue.get(record_id)
        if item is None:
            raise KeyError(
                f"No review item with record_id {record_id!r}.",
            )
        item.resolved = True
        item.resolution = resolution
        item.resolved_by = resolved_by
        item.resolved_at = datetime.now(timezone.utc)
        return item

    # ---- Design-monorepo mirror (master §6.6.1) -----------------------

    @action_executor(
        planning_summary=(
            "Mirror an ingested source into the design monorepo's corpora/. "
            "Returns the commit SHA, or a typed 'not configured' result."
        ),
    )
    async def mirror_to_design_monorepo(
        self,
        source_path: str,
        *,
        sub_path: str = "papers",
        message: str | None = None,
    ) -> dict[str, Any]:
        if self._design_monorepo is None or self._design_monorepo_identity is None:
            return {
                "ok": False,
                "reason": "no design_monorepo configured for this capability",
            }
        src = Path(source_path)
        if not src.is_file():
            return {"ok": False, "reason": f"source file does not exist: {src}"}
        target_dir = (
            self._design_monorepo.working_dir
            / "corpora"
            / sub_path
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / src.name
        try:
            target.write_bytes(src.read_bytes())
        except OSError as exc:
            return {"ok": False, "reason": f"copy failed: {exc}"}
        commit_message = message or f"corpora: add {src.name}"
        try:
            sha = await asyncio.to_thread(
                self._design_monorepo.commit_with_identity,
                self._design_monorepo_identity,
                commit_message,
                paths=[target.relative_to(self._design_monorepo.working_dir)],
            )
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": f"commit failed: {exc}"}
        return {"ok": True, "sha": sha, "path": str(target)}

    # ---- Suspension hooks ----------------------------------------------

    @override
    async def serialize_suspension_state(
        self, state: AgentSuspensionState
    ) -> AgentSuspensionState:
        if self._review_queue:
            state.custom_data["knowledge_curator_capability"] = {
                "review_queue": [
                    item.model_dump(mode="json")
                    for item in self._review_queue.values()
                ],
            }
        return state

    @override
    async def deserialize_suspension_state(
        self, state: AgentSuspensionState
    ) -> None:
        payload = state.custom_data.get("knowledge_curator_capability") or {}
        for raw in payload.get("review_queue") or ():
            try:
                item = ReviewItem.model_validate(raw)
            except Exception:  # noqa: BLE001
                continue
            self._review_queue[item.record.record_id] = item

    # ---- Internals -----------------------------------------------------

    def _wire_review_callback(self) -> None:
        original = self._ingestor._review

        async def wrapped(record: IngestionRecord, chunks: Sequence[Chunk]) -> None:
            summaries = tuple(c.text[:200] for c in chunks[:5])
            self._review_queue[record.record_id] = ReviewItem(
                record=record, chunk_summaries=summaries,
            )
            if original is not None:
                try:
                    await original(record, chunks)
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "KnowledgeCuratorCapability: chained review callback raised",
                    )

        # Replace the ingestor's callback. Sample rate stays as
        # configured on the ingestor; this wraps the queue.
        self._ingestor._review = wrapped

    async def _maybe_emit_event(self, record: IngestionRecord) -> None:
        if self._page_event_emitter is None:
            return
        if record.status is IngestionStatus.FAILED:
            return
        # Each chunk would in principle yield its own PageChangeEvent;
        # for simplicity we publish one ``page_added`` event per source.
        # Subscribers that want chunk-grain refire query the vector
        # store via ``retrieve``.
        try:
            await self._page_event_emitter(
                PageChangeEvent.page_added(
                    page_id=record.source_uri,
                    source=record.source_uri,
                    data_type="ingested_source",
                    scope_id=self.scope_id,
                    extra={
                        "ingestion_record_id": record.record_id,
                        "chunks": record.chunks_produced,
                        "claims": record.claims_extracted,
                    },
                )
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "KnowledgeCuratorCapability: page_event_emitter failed",
            )


# ---------------------------------------------------------------------------
# KnowledgeCuratorAgent
# ---------------------------------------------------------------------------


class KnowledgeCuratorAgent(Agent):
    """Generic knowledge-curator role (master §3.5).

    Owns the corpus + review queue. Subclassed by CPS-shared
    ``RegulatoryAgent`` / per-domain corpus agents (Phase P1).
    """

    agent_type: str = (
        "polymathera.colony.agents.roles.knowledge_curator.KnowledgeCuratorAgent"
    )


__all__ = (
    "KnowledgeCuratorAgent",
    "KnowledgeCuratorCapability",
    "ReviewItem",
    "PageEventEmitter",
)
