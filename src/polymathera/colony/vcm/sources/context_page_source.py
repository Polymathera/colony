"""Context page source: Interface for page clustering and retrieval.

This module provides abstractions for grouping virtual context pages
and finding relevant pages based on queries (key-query-value attention).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any
from pydantic import BaseModel, Field
from enum import Enum


from ..models import ContextPageId
from ..page_events import PageChangeEvent
from ...vcm.models import MmapConfig
from ...distributed.ray_utils import serving

logger = logging.getLogger(__name__)


class PageCluster(BaseModel):
    """A cluster of related pages."""
    cluster_id: str
    page_ids: list[str]
    syscontext: serving.ExecutionContext = Field(
        default_factory=serving.require_execution_context,
        description="For identifying related pages (e.g., from same git repo)."
    )
    relationship_score: float = Field(ge=0.0, le=1.0, description="Average relationship strength")
    cluster_type: str  # "file_group", "semantic", "hybrid", etc.
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextPageSource(ABC):
    """Abstract interface for page clustering and retrieval.
    This provides a mapping of application-level records to VCM pages.
    Application-level records include files, blackboard entries, knowledge graphs, etc.

    Implementations can use different strategies:
    - File-based: Uses FileGrouper from sharding infrastructure
    - Semantic: Uses embedding-based clustering
    - Hybrid: Combines multiple signals
    - LLM-learned: Uses LLM to determine clusters

    **Watcher contract** (master §5.6 item 1).

    Sources that can detect upstream changes and translate them into
    page-graph mutations are constructed with ``static=False`` and
    override ``watch()``. Sources that cannot — a one-shot static
    dump, an archived corpus snapshot, a git repo pinned at a frozen
    commit — pass ``static=True``; the convergence runtime refuses to
    subscribe to a static instance as a live page-graph input and
    instead relies on whatever bulk re-ingestion path the source
    provides. The same source class can produce both kinds of
    instance — see ``FileGrouperContextPageSource`` for the canonical
    mixed-mode use case.
    """

    def __init__(
        self,
        scope_id: str,
        mmap_config: MmapConfig,
        *,
        static: bool = True,
    ):
        """Initialize the context page source.
        Args:
            scope_id: Unique identifier for the scope of this source (e.g., file system ID, blackboard scope ID)
            mmap_config: Configuration for memory-mapped storage (if needed)
            static: Whether this source's backing store is static. The same
                source class can produce frozen-commit instances
                (``static=True``) and live-update instances
                (``static=False``) side by side; subclasses that implement
                ``watch()`` typically default this to ``False`` in their own
                ``__init__``. The VCM convergence runtime reads
                ``self.static`` to decide whether to attach the source as a
                live input.
        """
        self.scope_id = scope_id
        self.syscontext: serving.ExecutionContext = serving.require_execution_context()
        self.mmap_config = mmap_config
        self.static = static

    @classmethod
    def get_source_metadata(cls, scope_id: str) -> str:
        """Get metadata for the page source."""
        return f"{cls.__qualname__}:{scope_id}"

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the page source (load or build page graph)."""
        pass

    @abstractmethod
    async def claim_orphaned_events(self) -> None:
        """Claim any orphaned events for this source (e.g., from previous instance)."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Clean up resources and gracefully shut down the source."""
        pass

    @abstractmethod
    async def get_page_id_for_record(self, record_id: str) -> ContextPageId | None:
        """Get the page ID associated with a specific record ID, if any."""
        pass

    @abstractmethod
    async def get_record_ids_for_page(self, page_id: ContextPageId) -> list[str]:
        """Get all record IDs associated with a specific page ID."""
        pass

    @abstractmethod
    async def get_all_mapped_records(self) -> dict[str, ContextPageId]:
        """Get a mapping of all record IDs to their associated page IDs."""
        pass

    @abstractmethod
    async def get_all_mapped_pages(self) -> dict[ContextPageId, list[str]]:
        """Get a mapping of all page IDs to their associated record IDs."""
        pass

    async def watch(self) -> AsyncIterator[PageChangeEvent]:
        """Yield ``PageChangeEvent``s as the backing store mutates.

        The default raises ``NotImplementedError``; sources that can
        detect changes override and are constructed with
        ``static=False``. The convergence runtime checks ``self.static``
        before attaching a source as a live input.

        The implementation contract:

        - The iterator must be cooperatively cancellable. The runtime
          drives it from a long-running task; cancellation should
          unblock cleanly within ``shutdown``'s grace window.
        - The iterator must not emit duplicate events for the same
          underlying mutation; debouncing is the source's
          responsibility (the convergence runtime layer applies a
          rate limiter on top, but that is a safety net, not a
          de-duplicator).
        - The iterator must propagate ``effective_at`` and ``data_type``
          on the event when it can, so subscribers can filter without
          re-resolving the affected page.
        """

        raise NotImplementedError(
            f"{type(self).__name__} is a static source (static = "
            f"{self.static}); override watch() and construct with "
            "static=False to make it live."
        )
        # Make the function an async generator at the bytecode level
        # so subclasses' overrides match this signature without the
        # type checker complaining about the abstract default.
        if False:  # pragma: no cover
            yield  # type: ignore[unreachable]


class BuilInContextPageSourceType(str, Enum):
    """Built-in context page source types for easy reference.
    Users can also register custom types via ContextPageSourceFactory.
    """
    FILE_GROUPER = "file_grouper"
    BLACKBOARD = "blackboard"


class ContextPageSourceFactory:
    """Factory for creating ContextPageSource instances.

    Page source classes register via the ``@register_new_source_type``
    decorator, which records both the class and its module path.  On the
    driver node (``polymath.py``), all necessary modules are imported
    (built-in + user-defined), then ``publish_to_env()`` serializes the
    module paths to ``POLYMATH_PAGE_SOURCE_MODULES``.  That env var is
    propagated to Ray workers via ``runtime_env``.

    On worker nodes, ``create()`` calls ``_load_sources_from_env()``
    which reads the env var, imports each module (triggering their
    ``@register`` decorators), and populates the registry.  This handles
    both built-in and user-defined page sources without hardcoding.
    """

    _registry: dict[str, type[ContextPageSource]] = {}
    _module_paths: dict[str, str] = {}
    _env_loaded: bool = False

    # Env var used to propagate registered module paths to Ray workers.
    # Starts with POLYMATH_ so it's included in worker_env_vars automatically.
    _ENV_VAR = "POLYMATH_PAGE_SOURCE_MODULES"

    @staticmethod
    def register_new_source_type(source_type: str):
        """Decorator to register a new ContextPageSource type."""
        def decorator(source_class: type[ContextPageSource]):
            ContextPageSourceFactory._registry[source_type] = source_class
            ContextPageSourceFactory._module_paths[source_type] = source_class.__module__
            return source_class
        return decorator

    @staticmethod
    def publish_to_env() -> None:
        """Serialize registered module paths to env var for propagation to workers.

        Call this on the driver node after all page source modules have been
        imported and before ``ray.init()`` so the env var is included in
        ``runtime_env["env_vars"]``.
        """
        import json
        import os
        os.environ[ContextPageSourceFactory._ENV_VAR] = json.dumps(
            ContextPageSourceFactory._module_paths
        )
        logger.info(
            "Published %d page source module paths: %s",
            len(ContextPageSourceFactory._module_paths),
            list(ContextPageSourceFactory._module_paths.keys()),
        )

    @staticmethod
    def _load_sources_from_env() -> None:
        """Import page source modules from env var (propagated from driver to workers).

        On worker nodes, the registry starts empty. This reads the module
        paths serialized by ``publish_to_env()`` on the driver and imports
        each one, which triggers their ``@register_new_source_type``
        decorators and populates the registry.
        """
        if ContextPageSourceFactory._env_loaded:
            return
        ContextPageSourceFactory._env_loaded = True

        import json
        import importlib
        import os

        module_paths_json = os.environ.get(ContextPageSourceFactory._ENV_VAR)
        if not module_paths_json:
            return

        try:
            module_paths: dict[str, str] = json.loads(module_paths_json)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Invalid JSON in %s env var", ContextPageSourceFactory._ENV_VAR)
            return

        for source_type, mod_path in module_paths.items():
            if source_type in ContextPageSourceFactory._registry:
                continue
            try:
                importlib.import_module(mod_path)
            except ImportError:
                logger.warning(
                    "Failed to import page source module %s for type '%s'",
                    mod_path, source_type,
                )

    @staticmethod
    def list_registered_source_types() -> list[type[ContextPageSource]]:
        """List all registered context page source types."""
        ContextPageSourceFactory._load_sources_from_env()
        return list(ContextPageSourceFactory._registry.values())

    @staticmethod
    def create(
        source_type: str,
        scope_id: str,
        mmap_config: MmapConfig,
        *args: Any,
        **kwargs: Any
    ) -> ContextPageSource:
        """Create and initialize a ContextPageSource.

        Args:
            source_type: Type of source to create ("file_grouper", etc.)
            scope_id: Unique identifier for the scope (e.g., file system ID, blackboard scope ID)
            mmap_config: Configuration for memory-mapped storage (if needed)
            *args: Positional arguments for source constructor
            **kwargs: Keyword arguments for source constructor

        Returns:
            Initialized ContextPageSource instance
        """
        ContextPageSourceFactory._load_sources_from_env()
        if source_type in ContextPageSourceFactory._registry:
            return ContextPageSourceFactory._registry[source_type](
                scope_id=scope_id,
                mmap_config=mmap_config,
                *args,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown ContextPageSource type: {source_type}, not found in registry: {list(ContextPageSourceFactory._registry.keys())}")


