"""Generic agent roles in colony.

Two colony-generic roles:

- ``KnowledgeCuratorAgent`` — corpus ingestion, KG maintenance,
  sampled-human-review queue. Bridges the C1a knowledge layer with
  the C4 convergence runtime and the C5 design monorepo.
- ``DataCurationAgent`` — dataset registration + content-hash
  versioning + lineage.

The earlier ``SupervisorAgent`` / ``OptimizationAgent`` /
``GeneralPurposeAgent`` and the ``workflows`` runtime were deleted:
speculative wrappers with no real consumers (a SciPy passthrough; a
routing/summarisation surface that duplicated existing primitives;
a parallel orchestration runtime competing with action policies).
The human-approval gate primitives that lived on
``SupervisorCapability`` are now a proper event-driven
``HumanApprovalCapability`` under
``agents/patterns/capabilities/`` — see PR #2 for the rationale.

CPS-shared agents layer on top in ``cps/agents/``.
"""

from __future__ import annotations

from .data_curation import (
    ContentHashMismatchError,
    DataCurationAgent,
    DataCurationCapability,
    DataCurationError,
    DatasetLineageEdge,
    DatasetVersion,
    UnknownDatasetVersionError,
)
from .knowledge_curator import (
    KnowledgeCuratorAgent,
    KnowledgeCuratorCapability,
    PageEventEmitter,
    ReviewItem,
)


__all__ = (
    # Knowledge curator
    "KnowledgeCuratorAgent",
    "KnowledgeCuratorCapability",
    "ReviewItem",
    "PageEventEmitter",
    # Data curation
    "DataCurationAgent",
    "DataCurationCapability",
    "DataCurationError",
    "UnknownDatasetVersionError",
    "ContentHashMismatchError",
    "DatasetVersion",
    "DatasetLineageEdge",
)
