"""Persistent storage for VirtualContextPages.

This module provides durable storage for virtual context pages using
Polymathera's distributed storage system. Pages can be stored either on:
- EFS (Elastic File System) - default, fast access, shared across cluster
- S3 (Object Storage) - optional, cheaper for cold storage

The storage uses a hybrid approach:
- Page metadata stored in relational database (PostgreSQL) for efficient queries
- Page tokens stored as binary blobs in file system or S3
"""

from __future__ import annotations

import json
import asyncio
import logging
import msgpack
import pickle
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Literal, TYPE_CHECKING
from overrides import override
import sqlmodel as sqlm
from sqlalchemy import func
from pydantic import BaseModel, Field

import networkx as nx

from .models import VirtualContextPageMetadata, VirtualContextPage, ContextPageId
from .sources import PageCluster
from ..distributed.ray_utils.serving import get_colony_id, get_tenant_id, require_colony_id, require_tenant_id

if TYPE_CHECKING:
    from ..distributed.storage import Storage
    from ..distributed.stores.databases import RelationalStorage


logger = logging.getLogger(__name__)



class PageMetadataStore(ABC):

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the metadata storage."""
        pass

    @abstractmethod
    async def store_page_metadata(self, page: VirtualContextPage, storage_location: str, backend_type: str) -> None:
        """Store page metadata.

        Args:
            page: VirtualContextPage object
            storage_location: Location where tokens are stored
            backend_type: Storage backend type ("efs" or "s3")
        """
        pass

    @abstractmethod
    async def get_page_metadata(self, page_id: str) -> VirtualContextPageMetadata | None:
        """Retrieve page metadata by page ID.

        Args:
            page_id: Unique page identifier

        Returns:
            VirtualContextPageMetadata if found, None otherwise
        """
        pass

    @abstractmethod
    async def delete_page_metadata(self, page_id: str) -> VirtualContextPageMetadata | None:
        """Delete page metadata by page ID.

        Args:
            page_id: Unique page identifier

        Returns:
            Deleted VirtualContextPageMetadata if found, None otherwise
        """
        pass

    @abstractmethod
    async def list_pages(
        self,
        source_pattern: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ContextPageId]:
        """List page IDs with optional filtering.
        Args:
            source_pattern: Optional source pattern to filter pages
            limit: Maximum number of pages to return
            offset: Number of pages to skip for pagination

        Returns:
            List of ContextPageId objects matching the filters
        """
        pass

    @abstractmethod
    async def list_page_summaries(
        self,
        source_pattern: str | None = None,
        limit: int = 20000,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List page summaries (id, source, size) without loading blobs.

        Args:
            source_pattern: Filter by source pattern (SQL LIKE pattern)
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of dicts with page_id, source, size, group_id, colony_id, tenant_id
        """
        pass

    @abstractmethod
    async def query_pages_by_metadata(
        self,
        filters: dict[str, Any],
        limit: int = 1000,
    ) -> list[ContextPageId]:
        """Query pages by metadata filters.

        Args:
            filters: Dictionary of metadata filters
            limit: Maximum number of pages to return

        Returns:
            List of ContextPageId objects matching the filters
        """
        pass

    @abstractmethod
    async def get_storage_stats(self) -> dict:
        """Get storage statistics.

        Returns:
            Dictionary with storage statistics
        """
        pass



class RelationalPageMetadataStore(PageMetadataStore):
    def __init__(self, relational_storage: RelationalStorage):
        self.relational_storage = relational_storage

    @override
    async def initialize(self) -> None:
        """Initialize the metadata storage (create tables)."""
        # Create metadata table schema in PostgreSQL (needed for both backends)
        engine = self.relational_storage._get_current_loop_engine()
        async with engine.begin() as conn:
            await conn.run_sync(VirtualContextPageMetadata.metadata.create_all)

    @override
    async def store_page_metadata(self, page: VirtualContextPage, storage_location: str, backend_type: str) -> None:
        """Store page metadata in the relational database."""
        import json
        session_maker = self.relational_storage._get_current_loop_session_maker()
        async with session_maker() as session:
            async with session.begin():
                # Check if page already exists
                stmt = sqlm.select(VirtualContextPageMetadata).where(
                    VirtualContextPageMetadata.page_id == page.page_id
                )
                result = await session.execute(stmt)
                existing_page: VirtualContextPageMetadata | None = result.scalar_one_or_none()

                source = page.metadata.get("source", "unknown")

                if existing_page:
                    # Update existing page
                    existing_page.tenant_id = page.tenant_id
                    existing_page.colony_id = page.colony_id
                    existing_page.source = source
                    existing_page.updated_at = datetime.now(timezone.utc)
                    existing_page.size = page.size
                    existing_page.metadata_json = json.dumps(page.metadata)
                    existing_page.storage_location = storage_location
                    existing_page.storage_backend = backend_type
                    existing_page.group_id = page.group_id
                    existing_page.created_by = page.created_by
                    existing_page.expires_at = (
                        datetime.fromtimestamp(page.expires_at, tz=timezone.utc)
                        if page.expires_at
                        else None
                    )
                    session.add(existing_page)
                else:
                    # Create new page metadata
                    page_metadata = VirtualContextPageMetadata(
                        page_id=page.page_id,
                        tenant_id=page.tenant_id,
                        colony_id=page.colony_id,
                        source=source,
                        created_at=datetime.fromtimestamp(page.created_at, tz=timezone.utc),
                        size=page.size,
                        metadata_json=json.dumps(page.metadata),
                        storage_location=storage_location,
                        storage_backend=backend_type,
                        group_id=page.group_id,
                        created_by=page.created_by,
                        expires_at=(
                            datetime.fromtimestamp(page.expires_at, tz=timezone.utc)
                            if page.expires_at
                            else None
                        ),
                    )
                    session.add(page_metadata)

    @override
    async def get_page_metadata(self, page_id: str) -> VirtualContextPageMetadata | None:
        """Retrieve page metadata by page ID."""
        tenant_id = require_tenant_id()
        colony_id = require_colony_id()
        session_maker = self.relational_storage._get_current_loop_session_maker()
        async with session_maker() as session:
            stmt = sqlm.select(VirtualContextPageMetadata).where(
                VirtualContextPageMetadata.page_id == page_id,
                VirtualContextPageMetadata.tenant_id == tenant_id,
                VirtualContextPageMetadata.colony_id == colony_id,
            )
            result = await session.execute(stmt)
            page_metadata: VirtualContextPageMetadata | None = result.scalar_one_or_none()
            return page_metadata

    @override
    async def delete_page_metadata(self, page_id: str) -> VirtualContextPageMetadata | None:
        """Delete page metadata by page ID."""
        session_maker = self.relational_storage._get_current_loop_session_maker()
        async with session_maker() as session:
            async with session.begin():
                tenant_id = require_tenant_id()
                colony_id = require_colony_id()
                stmt = sqlm.select(VirtualContextPageMetadata).where(
                    VirtualContextPageMetadata.page_id == page_id,
                    VirtualContextPageMetadata.tenant_id == tenant_id,
                    VirtualContextPageMetadata.colony_id == colony_id,
                )
                result = await session.execute(stmt)
                page_metadata: VirtualContextPageMetadata | None = result.scalar_one_or_none()
                if not page_metadata:
                    logger.debug(f"Page {page_id} not found for deletion")
                    return None

                await session.delete(page_metadata)
                await session.commit()
                return page_metadata

    @override
    async def list_pages(
        self,
        source_pattern: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ContextPageId]:
        """Query page IDs by filters.

        Args:
            source_pattern: Filter by source pattern (SQL LIKE pattern)
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of page IDs matching filters

        Raises:
            Exception: If query fails
        """
        tenant_id = get_tenant_id()
        colony_id = get_colony_id()

        logger.debug(
            f"Listing pages (tenant={tenant_id}, colony={colony_id}, source_pattern={source_pattern}, "
            f"limit={limit}, offset={offset})"
        )

        try:
            session_maker = self.relational_storage._get_current_loop_session_maker()
            async with session_maker() as session:
                # Build query with filters
                stmt = sqlm.select(VirtualContextPageMetadata.page_id)

                if tenant_id:
                    stmt = stmt.where(VirtualContextPageMetadata.tenant_id == tenant_id)

                if colony_id:
                    stmt = stmt.where(VirtualContextPageMetadata.colony_id == colony_id)

                if source_pattern:
                    stmt = stmt.where(VirtualContextPageMetadata.source.like(source_pattern))

                stmt = stmt.order_by(VirtualContextPageMetadata.created_at.desc())
                stmt = stmt.limit(limit).offset(offset)

                result = await session.execute(stmt)
                page_ids = [row[0] for row in result.all()]

            logger.debug(f"Found {len(page_ids)} pages matching filters")
            return page_ids

        except Exception as e:
            logger.error(f"Failed to list pages: {e}", exc_info=True)
            raise

    @override
    async def list_page_summaries(
        self,
        source_pattern: str | None = None,
        limit: int = 20000,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List page summaries (id, source, size) without loading blobs.

        Args:
            source_pattern: Filter by source pattern (SQL LIKE pattern)
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of dicts with page_id, source, size, group_id, tenant_id
        """
        try:
            tenant_id = get_tenant_id()
            colony_id = get_colony_id()

            session_maker = self.relational_storage._get_current_loop_session_maker()
            async with session_maker() as session:
                stmt = sqlm.select(
                    VirtualContextPageMetadata.page_id,
                    VirtualContextPageMetadata.source,
                    VirtualContextPageMetadata.size,
                    VirtualContextPageMetadata.group_id,
                    VirtualContextPageMetadata.tenant_id,
                    VirtualContextPageMetadata.colony_id,
                    VirtualContextPageMetadata.metadata_json,
                )

                if tenant_id:
                    stmt = stmt.where(VirtualContextPageMetadata.tenant_id == tenant_id)
                if colony_id:
                    stmt = stmt.where(VirtualContextPageMetadata.colony_id == colony_id)
                if source_pattern:
                    stmt = stmt.where(VirtualContextPageMetadata.source.like(source_pattern))

                stmt = stmt.order_by(VirtualContextPageMetadata.created_at.desc())
                stmt = stmt.limit(limit).offset(offset)

                result = await session.execute(stmt)
                summaries = []
                for row in result.all():
                    files: list[str] = []
                    if row.metadata_json:
                        try:
                            import json
                            meta = json.loads(row.metadata_json) if isinstance(row.metadata_json, str) else row.metadata_json
                            files = meta.get("files", [])
                        except Exception:
                            pass
                    summaries.append({
                        "page_id": row.page_id,
                        "source": row.source or "",
                        "size": row.size or 0,
                        "group_id": row.group_id or "",
                        "tenant_id": row.tenant_id or "",
                        "colony_id": row.colony_id or "",
                        "files": files,
                    })
                return summaries
        except Exception as e:
            logger.error(f"Failed to list page summaries: {e}", exc_info=True)
            raise

    @override
    async def query_pages_by_metadata(
        self,
        filters: dict[str, Any],
        limit: int = 1000,
    ) -> list[ContextPageId]:
        """Query pages by metadata filters.

        Uses indexed columns (``source``, ``group_id``, ``tenant_id``, ``colony_id``,
        ``created_by``) when available for efficient queries. Falls back to
        JSON metadata search for non-indexed fields.

        Convention: ``XYZContextPageSource`` sets
        ``metadata["source"] = f"{XYZContextPageSource.get_source_metadata(scope_id)}"`` on all pages it creates.
        Queries by ``scope_id`` use the indexed ``source`` column.

        Supported filter keys:
        - ``scope_id``: Maps to ``source = f"{XYZContextPageSource.get_source_metadata(scope_id)}"``.
        - ``group_id``: Direct indexed column match.
        - ``tenant_id``: Direct indexed column match.
        - ``colony_id``: Direct indexed column match.
        - ``created_by``: Direct indexed column match.

        Args:
            filters: Filter criteria (see above for supported keys).
            limit: Maximum number of pages to return.

        Returns:
            List of matching ``ContextPageId`` objects.
        """
        logger.debug(f"Querying pages by metadata: filters={filters}, limit={limit}")

        try:
            session_maker = self.relational_storage._get_current_loop_session_maker()
            async with session_maker() as session:
                stmt = sqlm.select(VirtualContextPageMetadata)

                # Apply indexed column filters
                if "scope_id" in filters:
                    from .sources.context_page_source import ContextPageSourceFactory
                    source_prefixes = [
                        f"{cls.get_source_metadata(filters['scope_id'])}"
                        for cls in ContextPageSourceFactory.list_registered_source_types()
                    ]
                    for source_prefix in source_prefixes:
                        stmt = stmt.where(
                            VirtualContextPageMetadata.source.like(f"{source_prefix}%")
                        )

                if "group_id" in filters:
                    stmt = stmt.where(
                        VirtualContextPageMetadata.group_id == filters["group_id"]
                    )

                if "tenant_id" in filters:
                    stmt = stmt.where(
                        VirtualContextPageMetadata.tenant_id == filters["tenant_id"]
                    )

                if "colony_id" in filters:
                    stmt = stmt.where(
                        VirtualContextPageMetadata.colony_id == filters["colony_id"]
                    )

                if "created_by" in filters:
                    stmt = stmt.where(
                        VirtualContextPageMetadata.created_by == filters["created_by"]
                    )

                stmt = stmt.order_by(VirtualContextPageMetadata.created_at.desc())
                stmt = stmt.limit(limit)

                result = await session.execute(stmt)
                rows = result.all()

            # Load full pages for each matching metadata row
            pages: list[ContextPageId] = []
            for (row,) in rows:
                pages.append(row.page_id)

            logger.debug(f"Found {len(pages)} pages matching metadata filters")
            return pages

        except Exception as e:
            logger.error(f"Failed to query pages by metadata: {e}", exc_info=True)
            raise

    @override
    async def get_storage_stats(self) -> dict:
        """Get storage statistics.

        Returns:
            Dictionary with storage stats

        Raises:
            Exception: If stats query fails
        """
        try:
            session_maker = self.relational_storage._get_current_loop_session_maker()
            async with session_maker() as session:
                # Get total pages and size
                stmt = sqlm.select(
                    func.count(VirtualContextPageMetadata.page_id).label("total_pages"),
                    func.sum(VirtualContextPageMetadata.size).label("total_tokens"),
                    func.count(func.distinct(VirtualContextPageMetadata.tenant_id)).label("unique_tenants"),
                    func.count(func.distinct(VirtualContextPageMetadata.colony_id)).label("unique_colonies"),
                    func.count(func.distinct(VirtualContextPageMetadata.source)).label("unique_sources"),
                )
                result = await session.execute(stmt)
                row = result.one()

                # Get stats by backend
                stmt_backend = sqlm.select(
                    VirtualContextPageMetadata.storage_backend,
                    func.count(VirtualContextPageMetadata.page_id).label("page_count"),
                    func.sum(VirtualContextPageMetadata.size).label("total_tokens"),
                ).group_by(VirtualContextPageMetadata.storage_backend)

                result_backend = await session.execute(stmt_backend)
                backend_rows = result_backend.all()

                return {
                    "total_pages": row.total_pages or 0,
                    "total_tokens": row.total_tokens or 0,
                    "unique_tenants": row.unique_tenants or 0,
                    "unique_colonies": row.unique_colonies or 0,
                    "unique_sources": row.unique_sources or 0,
                    "by_backend": {
                        backend_row.storage_backend: {
                            "page_count": backend_row.page_count,
                            "total_tokens": backend_row.total_tokens,
                        }
                        for backend_row in backend_rows
                    },
                }

        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}", exc_info=True)
            raise


class PageBlobStorage(ABC):

    @abstractmethod
    async def store_blob(self, blob_key: str, blob_data: bytes) -> str:
        pass

    @abstractmethod
    async def retrieve_blob(self, storage_location: str) -> bytes | None:
        pass

    @abstractmethod
    async def delete_blob(self, storage_location: str) -> bool:
        pass


class EfsPageBlobStorage(PageBlobStorage):

    def __init__(self, storage_backend: Storage, storage_path: str = "colony/context_pages"):
        self.storage_backend = storage_backend
        self.storage_path = storage_path

        self.file_storage = self.storage_backend.distributed_file_system
        logger.info(f"Using EFS for page token storage at {self.storage_path}")

    @override
    async def store_blob(self, blob_key: str, blob_data: bytes) -> str:
        """Persist a binary blob to storage.

        Args:
            blob_key: Key or path for the blob
            blob_data: Binary data to store

        Returns:
            Storage location string

        Raises:
            Exception: If storage operation fails
        """
        # Write to distributed file system (async)
        blob_path = await self.file_storage.join(blob_key)
        await self.file_storage.write_binary_file(blob_path, blob_data)
        return f"efs://{blob_key}"

    @override
    async def retrieve_blob(self, storage_location: str) -> bytes | None:
        """Load a blob from storage.

        Args:
            storage_location: Storage location string

        Returns:
            Bytes data if found, None otherwise

        Raises:
            Exception: If retrieval operation fails
        """
        # Read from EFS (async)
        blob_key = storage_location.replace("efs://", "")
        blob_path = await self.file_storage.join(blob_key)
        if not await self.file_storage.exists(blob_path):
            return None
        return await self.file_storage.read_binary_file(blob_path)

    @override
    async def delete_blob(self, storage_location: str) -> bool:
        """Delete a page from storage.

        Args:
            storage_location: Storage location string

        Returns:
            True if page was deleted, False if not found

        Raises:
            Exception: If deletion operation fails
        """
        # Delete from EFS (async)
        blob_key = storage_location.replace("efs://", "")
        blob_path = await self.file_storage.join(blob_key)
        if not await self.file_storage.exists(blob_path):
            return False

        await self.file_storage.delete(blob_path)
        return True


class S3PageBlobStorage(PageBlobStorage):

    def __init__(self, storage_backend: Storage, storage_path: str = "colony/context_pages", s3_bucket: str = "polymathera-context-pages"):
        self.storage_backend = storage_backend
        self.storage_path = storage_path
        self.s3_bucket = s3_bucket
        self.object_storage = self.storage_backend.object_storage

    @override
    async def store_blob(self, blob_key: str, blob_data: bytes) -> str:
        """Persist a binary blob to storage.

        Args:
            blob_key: Key or path for the blob
            blob_data: Binary data to store

        Returns:
            Storage location string

        Raises:
            Exception: If storage operation fails
        """
        # Store to S3 object storage (sync - run in executor)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            self.object_storage.store_object,
            self.s3_bucket,
            blob_key,
            blob_data
        )
        return f"s3://{self.s3_bucket}/{blob_key}"

    @override
    async def retrieve_blob(self, storage_location: str) -> bytes | None:
        """Load a blob from storage.

        Args:
            storage_location: Storage location string

        Returns:
            Bytes data if found, None otherwise

        Raises:
            Exception: If retrieval operation fails
        """
        # Read from S3 (sync - run in executor)
        s3_path = storage_location.replace("s3://", "")
        bucket, key = s3_path.split("/", 1)
        assert bucket == self.s3_bucket, f"Bucket mismatch: expected {self.s3_bucket}, got {bucket}"
        loop = asyncio.get_running_loop()

        try:
            return await loop.run_in_executor(
                None,
                self.blob_storage.get_object,
                self.s3_bucket,
                key
            )
        except Exception as e:
            # S3 raises exception if object doesn't exist
            return None

    @override
    async def delete_blob(self, storage_location: str) -> bool:
        """Delete a page from storage.

        Args:
            storage_location: Storage location string

        Returns:
            True if page was deleted, False if not found

        Raises:
            Exception: If deletion operation fails
        """
        # Delete from S3 (sync - run in executor)
        s3_path = storage_location.replace("s3://", "")
        bucket, key = s3_path.split("/", 1)
        assert bucket == self.s3_bucket, f"Bucket mismatch: expected {self.s3_bucket}, got {bucket}"
        loop = asyncio.get_running_loop()

        try:
            await loop.run_in_executor(
                None,
                self.object_storage.delete_object,
                self.s3_bucket,
                key
            )
            return True
        except Exception:
            return False


class PageStorageConfig(BaseModel):
    backend_type: Literal["efs", "s3"] = Field(default="efs")
    storage_path: str = Field(default="colony/context_pages")
    s3_bucket: str = Field(default="polymathera-context-pages")


class PageStorage:
    """Persistent storage for VirtualContextPages.

    Uses Polymathera's distributed storage system:
    - Metadata in PostgreSQL (for queries and stats)
    - Tokens in EFS or S3 (configurable)

    Example:
        ```python
        page_storage = PageStorage(
            backend_type="efs",  # or "s3"
            storage_path=self.page_storage_path,
            s3_bucket=self.page_storage_s3_bucket,
        )
        await page_storage.initialize()

        # Store a page
        await page_storage.store_page(page)

        # Retrieve a page
        page = await page_storage.retrieve_page("page-id")
        ```
    """

    def __init__(
        self,
        backend_type: Literal["efs", "s3"] = "efs",
        storage_path: str = "colony/context_pages",
        s3_bucket: str = "polymathera-context-pages",
    ):
        """Initialize page storage.

        Args:
            backend_type: Storage backend to use ("efs" or "s3")
            storage_path: Path prefix for storing pages
            s3_bucket: S3 bucket name (used only if backend_type="s3")
        """
        self.backend_type = backend_type
        self.storage_path = storage_path
        self.s3_bucket = s3_bucket
        self.storage_backend: Storage | None = None

        # Storage components (initialized in initialize())
        self.page_metadata_store: PageMetadataStore | None = None
        self.blob_storage: PageBlobStorage | None = None  # ScalableDistributedFileSystem or ObjectStorage
        self._page_graphs: dict[str, nx.DiGraph] = {}  # TODO: This should be loaded from PageStorage dynamically? Or is this just a local cache?
        self._page_graph_data: dict[str, Any] = {}  # For storing additional graph-level data (e.g., page-to-file mapping)

    async def initialize(self) -> None:
        """Initialize storage components."""
        logger.info(f"Initializing PageStorage with backend={self.backend_type}")

        self.storage_backend = await self._get_storage_backend()

        # Get relational storage for metadata
        self.page_metadata_store = RelationalPageMetadataStore(self.storage_backend.relational_storage)
        await self.page_metadata_store.initialize()

        # Get blob storage based on backend type
        if self.backend_type == "efs":
            self.blob_storage = EfsPageBlobStorage(storage_backend=self.storage_backend, storage_path=self.storage_path)
            logger.info(f"Using EFS for page token storage at {self.storage_path}")
        elif self.backend_type == "s3":
            self.blob_storage = S3PageBlobStorage(
                storage_backend=self.storage_backend,
                s3_bucket=self.s3_bucket,
                storage_path=self.storage_path
            )
            logger.info(f"Using S3 for page token storage at {self.s3_bucket}/{self.storage_path}")
        else:
            raise ValueError(f"Invalid backend_type: {self.backend_type}. Must be 'efs' or 's3'")

        logger.info("PageStorage initialized successfully")

    async def _get_storage_backend(self) -> Storage:
        """Helper to get the Storage instance from Polymathera."""
        from ..distributed import get_polymathera
        polymathera = get_polymathera()
        storage_backend = await polymathera.get_storage()
        return storage_backend

    async def store_page(self, page: VirtualContextPage) -> None:
        """Persist a virtual page to storage.

        Args:
            page: VirtualContextPage to store

        Raises:
            Exception: If storage operation fails
        """
        logger.debug(f"Storing page {page.page_id} ({page.size} tokens)")

        try:
            # 1. Serialize tokens to binary using msgpack (efficient)
            tokens_data = msgpack.packb(page.tokens, use_bin_type=True)

            # 2. Store tokens to blob storage
            blob_key = self._get_blob_key(page.page_id, page.colony_id, page.tenant_id)

            storage_location = await self.blob_storage.store_blob(blob_key, tokens_data)

            # 2b. Store text blob if available (for remote LLM deployments)
            if page.text is not None:
                text_blob_key = f"{blob_key}.text"
                text_data = page.text.encode("utf-8")
                await self.blob_storage.store_blob(text_blob_key, text_data)

            # 3. Store metadata to PostgreSQL
            await self.page_metadata_store.store_page_metadata(
                page,
                storage_location,
                self.backend_type
            )

            logger.debug(
                f"Stored page {page.page_id} "
                f"(backend={self.backend_type}, size={page.size} tokens, "
                f"has_text={page.text is not None}, location={storage_location})"
            )

        except Exception as e:
            logger.error(f"Failed to store page {page.page_id}: {e}", exc_info=True)
            raise

    async def retrieve_page(self, page_id: ContextPageId) -> VirtualContextPage | None:
        """Load a virtual page from storage.

        Args:
            page_id: Page identifier

        Returns:
            VirtualContextPage if found, None otherwise

        Raises:
            Exception: If retrieval operation fails
        """
        # TODO: This assumes page_id is globally unique across tenants and colonies. We may want to enforce this or change the API to include tenant_id and colony_id for disambiguation.
        logger.debug(f"Retrieving page {page_id}")

        try:
            # 1. Get metadata from PostgreSQL
            page_metadata = await self.page_metadata_store.get_page_metadata(page_id)

            if not page_metadata:
                logger.debug(f"Page {page_id} not found in metadata")
                return None

            # 2. Get tokens from blob storage
            storage_location = page_metadata.storage_location

            tokens_data = await self.blob_storage.retrieve_blob(storage_location)

            # 3. Deserialize tokens
            tokens = msgpack.unpackb(tokens_data, raw=False)

            # 3b. Try to retrieve text blob (may not exist for older pages)
            text = None
            try:
                blob_key = self._get_blob_key(page_id, page_metadata.colony_id, page_metadata.tenant_id)
                text_blob_key = f"{blob_key}.text"
                text_data = await self.blob_storage.retrieve_blob(text_blob_key)
                if text_data:
                    text = text_data.decode("utf-8")
            except Exception:
                # Text blob doesn't exist — this is expected for pages created
                # before the text field was added
                pass

            # 4. Reconstruct VirtualContextPage
            metadata = json.loads(page_metadata.metadata_json)

            page = VirtualContextPage(
                page_id=page_metadata.page_id,
                tenant_id=page_metadata.tenant_id,
                colony_id=page_metadata.colony_id,
                tokens=tokens,
                text=text,
                size=page_metadata.size,
                created_at=page_metadata.created_at.timestamp(),
                metadata=metadata,
                group_id=page_metadata.group_id,
                created_by=page_metadata.created_by,
                expires_at=(
                    page_metadata.expires_at.timestamp()
                    if page_metadata.expires_at
                    else None
                ),
            )

            logger.debug(
                f"Retrieved page {page_id} "
                f"(backend={self.backend_type}, size={page.size} tokens)"
            )

            return page

        except Exception as e:
            logger.error(f"Failed to retrieve page {page_id}: {e}", exc_info=True)
            raise

    async def delete_page(self, page_id: ContextPageId) -> bool:
        """Delete a page from storage.

        Args:
            page_id: Page identifier

        Returns:
            True if page was deleted, False if not found

        Raises:
            Exception: If deletion operation fails
        """
        logger.debug(f"Deleting page {page_id}")

        try:
            # 1. Delete metadata from PostgreSQL
            page_metadata = await self.page_metadata_store.delete_page_metadata(page_id)
            if not page_metadata:
                logger.debug(f"Page {page_id} not found for deletion")
                return False

            # 2. Get storage location from metadata
            storage_location = page_metadata.storage_location

            # 3. Delete blob from storage
            await self.blob_storage.delete_blob(storage_location)

            # 3b. Delete text blob if it exists
            try:
                blob_key = self._get_blob_key(page_id, page_metadata.colony_id, page_metadata.tenant_id)
                text_blob_key = f"{blob_key}.text"
                await self.blob_storage.delete_blob(text_blob_key)
            except Exception:
                pass  # Text blob may not exist

            logger.info(f"Deleted page {page_id} (backend={self.backend_type})")
            return True

        except Exception as e:
            logger.error(f"Failed to delete page {page_id}: {e}", exc_info=True)
            raise

    async def list_pages(
        self,
        source_pattern: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[str]:
        """Query page IDs by filters.

        Args:
            source_pattern: Filter by source pattern (SQL LIKE pattern)
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of page IDs matching filters

        Raises:
            Exception: If query fails
        """
        return await self.page_metadata_store.list_pages(
            source_pattern=source_pattern,
            limit=limit,
            offset=offset
        )

    async def list_page_summaries(
        self,
        source_pattern: str | None = None,
        limit: int = 20000,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List page summaries (id, source, size) without loading blobs.

        Delegates to the metadata store for efficient column-only queries.
        """
        return await self.page_metadata_store.list_page_summaries(
            source_pattern=source_pattern,
            limit=limit,
            offset=offset,
        )

    async def query_pages_by_metadata(
        self,
        filters: dict[str, Any],
        limit: int = 1000,
    ) -> list[VirtualContextPage]:
        """Query pages by metadata filters.

        Uses indexed columns (``source``, ``group_id``, ``tenant_id``, ``colony_id``,
        ``created_by``) when available for efficient queries. Falls back to
        JSON metadata search for non-indexed fields.

        Convention: ``XYZContextPageSource`` sets
        ``metadata["source"] = f"{XYZContextPageSource.get_source_metadata(scope_id)}"`` on all pages it creates.
        Queries by ``scope_id`` use the indexed ``source`` column.

        Supported filter keys:
        - ``scope_id``: Maps to ``source = f"{XYZContextPageSource.get_source_metadata(scope_id)}"``.
        - ``group_id``: Direct indexed column match.
        - ``tenant_id``: Direct indexed column match.
        - ``colony_id``: Direct indexed column match.
        - ``created_by``: Direct indexed column match.

        Args:
            filters: Filter criteria (see above for supported keys).
            limit: Maximum number of pages to return.

        Returns:
            List of matching ``VirtualContextPage`` objects.
        """
        page_ids = await self.page_metadata_store.query_pages_by_metadata(
            filters=filters,
            limit=limit
        )

        # Load full pages for each matching metadata row
        pages: list[VirtualContextPage] = []
        for page_id in page_ids:
            page = await self.retrieve_page(page_id)
            if page is not None:
                pages.append(page)

        logger.debug(f"Found {len(pages)} pages matching metadata filters")
        return pages

    async def get_storage_stats(self) -> dict:
        """Get storage statistics.

        Returns:
            Dictionary with storage stats

        Raises:
            Exception: If stats query fails
        """
        return await self.page_metadata_store.get_storage_stats()

    def _get_blob_key(self, page_id: str, colony_id: str, tenant_id: str) -> str:
        """Generate blob storage key for a page.

        Args:
            page_id: Page identifier
            colony_id: Colony identifier
            tenant_id: Tenant identifier

        Returns:
            Storage key/path
        """
        # Sanitize page_id for filesystem use (replace : / with -)
        safe_page_id = page_id.replace(":", "-").replace("/", "-")

        # Structure: {storage_path}/{tenant_id}/{colony_id}/{safe_page_id}.bin
        return f"{self.storage_path}/{tenant_id}/{colony_id}/{safe_page_id}.bin"

    # === Page Graph Persistence ===

    async def store_page_graph_level_data(
        self,
        data_key: str,
        graph_data: Any,
    ) -> None:
        """Store a page relationship graph to persistent storage.

        This supports ContextPageSource implementations that maintain
        relationship graphs between pages.

        Args:
            `data_key`: Key identifying the specific graph-level data
            `graph_data`: Arbitrary graph data (e.g., `PageKey`)

        Raises:
            `Exception`: If storage operation fails
        """
        tenant_id = require_tenant_id()
        colony_id = require_colony_id()
        logger.debug(f"Storing page graph data {data_key} for colony {colony_id}, tenant {tenant_id}")

        try:
            # Generate storage key for graph
            graph_key = f"{self.storage_path}/graphs/{tenant_id}/{colony_id}.{data_key}"
            await self.blob_storage.store_blob(graph_key, pickle.dumps(graph_data))

        except Exception as e:
            logger.error(f"Failed to store page graph data {data_key} for colony {colony_id}, tenant {tenant_id}: {e}", exc_info=True)
            raise

    async def store_page_graph(
        self,
        graph_data: nx.DiGraph) -> None:
        """Store a page relationship graph to persistent storage.

        This supports ContextPageSource implementations that maintain
        relationship graphs between pages.

        Args:
            graph_data: Page graph (`networkx.DiGraph`)

        Raises:
            `Exception`: If storage operation fails
        """
        try:
            return await self.store_page_graph_level_data(
                data_key="graph",
                graph_data=graph_data
            )
        except Exception as e:
            logger.error(f"Failed to store page graph: {e}", exc_info=True)
            raise

    async def retrieve_page_graph_level_data(
        self,
        data_key: str,
        cached: bool = True
    ) -> Any | None:
        """This allows retrieving additional graph-level data associated with a page graph, identified by `data_key`.
        This allows separating data that is associated with a page graph
        (e.g., page-to-file mapping) in storage so that it does not have to be
        updated each time the graph changes. Moreover, some graph-level data
        is computed once by one entity (e.g., sharding strategy) and used by all agents."""
        try:
            tenant_id = require_tenant_id()
            colony_id = require_colony_id()
            # Check local cache first
            effective_data_key = f"{tenant_id}:{colony_id}:{data_key}"
            if cached and effective_data_key in self._page_graph_data:
                return self._page_graph_data[effective_data_key]

            # Generate storage key for graph
            graph_key = f"{self.storage_path}/graphs/{tenant_id}/{colony_id}.{data_key}"
            graph_data = await self.blob_storage.retrieve_blob(graph_key)
            if not graph_data:
                logger.debug(f"Page graph data {data_key} for colony {colony_id}, tenant {tenant_id} not found in EFS")
                return None

            data_obj = pickle.loads(graph_data)
            self._page_graph_data[effective_data_key] = data_obj
            return data_obj

        except Exception as e:
            logger.error(f"Failed to retrieve page graph data {data_key} for colony {colony_id}, tenant {tenant_id}: {e}", exc_info=True)
            raise

    async def retrieve_page_graph(self) -> nx.DiGraph | None:
        """Retrieve a page relationship graph from persistent storage.
        This allows ContextPageSource implementations to load/store
        auxiliary data with their relationship graphs (e.g., file-to-page mappings).

        Returns:
            Graph data if found, None otherwise

        Raises:
            `Exception`: If retrieval operation fails
        """
        return await self.retrieve_page_graph_level_data(
            data_key="graph",
            cached=False
        )

    async def delete_page_graph(self) -> bool:
        """Delete a page relationship graph from storage.

        Returns:
            True if graph was deleted, False if not found

        Raises:
            Exception: If deletion operation fails
        """
        tenant_id = require_tenant_id()
        colony_id = require_colony_id()

        logger.debug(f"Deleting page graph for colony {colony_id}")

        try:
            # Generate storage key for graph
            graph_key = f"{self.storage_path}/graphs/{tenant_id}/{colony_id}.graph"

            deleted = await self.blob_storage.delete_blob(graph_key)
            if deleted:
                logger.info(f"Deleted page graph for colony {colony_id} from storage")
            else:
                logger.debug(f"Page graph for colony {colony_id} not found for deletion")
            return deleted

        except Exception as e:
            logger.error(f"Failed to delete page graph for colony {colony_id}: {e}", exc_info=True)
            raise

    async def load_page_graph(self, cached: bool = True) -> nx.DiGraph:
        """Load page graph dynamically from PageStorage.

        This allows the agent and its components to load
        the page graph when needed, rather than
        passing the entire graph in metadata.

        Args:
            cached: If True, return the locally cached graph if available. If False, always load from storage.
        Returns:
            The page graph as a networkx DiGraph. If loading from storage fails, returns an empty graph.
        """
        tenant_id = require_tenant_id()
        colony_id = require_colony_id()
        graph_key = f"{tenant_id}:{colony_id}:graph"
        if cached and graph_key in self._page_graphs and len(self._page_graphs[graph_key].nodes) > 0:
            return self._page_graphs[graph_key]

        try:
            if not colony_id or not tenant_id:
                raise RuntimeError("Missing colony_id or tenant_id, creating empty page graph")

            await self.initialize()

            graph = await self.retrieve_page_graph()

            if graph:
                self._page_graphs[graph_key] = graph
                logger.info(
                    f"Loaded page graph for {colony_id}: "
                    f"{len(graph.nodes)} nodes, {len(graph.edges)} edges, "
                    f"{graph.number_of_edges()} relationships"
                )
            else:
                # Build new graph (requires FileGrouper - deferred to first usage)
                logger.info(f"No existing page graph for {colony_id}, "
                            "will build on first use, creating empty graph")
                logger.warning(f"No existing page graph found in storage for {colony_id}, creating empty graph")
                self._page_graphs[graph_key] = nx.DiGraph()

            return self._page_graphs[graph_key]

        except Exception as e:
            logger.debug(f"Failed to load page graph: {e}")
            self._page_graphs[graph_key] = nx.DiGraph()
            return self._page_graphs[graph_key]

    async def get_page_cluster(
        self,
        cluster_size: int = 10,
        cluster_type: str | None = None
    ) -> PageCluster:
        """Get a cluster of related pages."""
        tenant_id = require_tenant_id()
        colony_id = require_colony_id()

        # TODO: Get page graph dynamically from PageStorage, rather than relying on in-memory graph.
        # This allows the agent to pick up the latest graph state.
        # Simple implementation: use community detection or just take connected component
        graph_key = f"{tenant_id}:{colony_id}:graph"
        if not self._page_graphs.get(graph_key) or len(self._page_graphs[graph_key].nodes) == 0:
            raise RuntimeError(
                f"PageStorage[{colony_id}]: page graph not initialized or empty"
            )

        graph = self._page_graphs[graph_key]

        # Get strongly connected components
        components = list(nx.strongly_connected_components(graph))

        # Find component matching size
        for i, component in enumerate(components):
            if len(component) <= cluster_size:
                page_ids = list(component)
                return PageCluster(
                    cluster_id=f"{colony_id}-cluster-{i}",
                    page_ids=page_ids,
                    relationship_score=0.8,  # TODO: Compute from graph
                    cluster_type=cluster_type or "connected_component",
                    metadata={"component_index": i}
                )

        ### # Group by locality_key from page metadata
        ### groups: dict[str, list[str]] = {}
        ### for node in graph.nodes:
        ###     data = graph.nodes[node]
        ###     locality = data.get("locality_key", "unknown")
        ###     groups.setdefault(locality, []).append(node)
        ### # Return the largest cluster that fits
        ### for group_key, page_ids in sorted(
        ###     groups.items(), key=lambda x: len(x[1]), reverse=True
        ### ):
        ###     if len(page_ids) <= cluster_size:
        ###         return PageCluster(
        ###             cluster_id=f"bb:{tenant_id}:{colony_id}:{group_key}",
        ###             page_ids=page_ids,
        ###             relationship_score=0.8,
        ###             cluster_type=cluster_type or "locality_group",
        ###             metadata={"locality_key": group_key},
        ###         )

        # Fallback: take first N pages
        all_pages = list(graph.nodes)[:cluster_size]
        return PageCluster(
            cluster_id=f"bb:{tenant_id}:{colony_id}:cluster-fallback",
            page_ids=all_pages,
            relationship_score=0.5,
            cluster_type="fallback",
            metadata={}
        )

    async def get_all_clusters(
        self,
        max_cluster_size: int = 10,
        min_cluster_size: int = 2
    ) -> AsyncIterator[PageCluster]:
        """Iterate over all page clusters."""
        tenant_id = require_tenant_id()
        colony_id = require_colony_id()
        # TODO: Get page graph dynamically from PageStorage, rather than relying on in-memory graph.
        # This allows the agent to pick up the latest graph state.
        graph_key = f"{tenant_id}:{colony_id}:graph"
        if not self._page_graphs.get(graph_key) or len(self._page_graphs[graph_key].nodes) == 0:
            return

        graph = self._page_graphs[graph_key]

        # Get strongly connected components
        components = list(nx.strongly_connected_components(graph))

        for i, component in enumerate(components):
            if min_cluster_size <= len(component) <= max_cluster_size:
                page_ids = list(component)
                yield PageCluster(
                    cluster_id=f"{tenant_id}:{colony_id}-cluster-{i}",
                    page_ids=page_ids,
                    relationship_score=0.8,
                    cluster_type="connected_component",
                    metadata={"component_index": i}
                )
        ### groups: dict[str, list[str]] = {}
        ### for node in graph.nodes:
        ###     data = graph.nodes[node]
        ###     locality = data.get("locality_key", "unknown")
        ###     groups.setdefault(locality, []).append(node)
        ### for group_key, page_ids in groups.items():
        ###     if min_cluster_size <= len(page_ids) <= max_cluster_size:
        ###         yield PageCluster(
        ###             cluster_id=f"bb:{tenant_id}:{colony_id}:{group_key}",
        ###             page_ids=page_ids,
        ###             relationship_score=0.8,
        ###             cluster_type="locality_group",
        ###             metadata={"locality_key": group_key},
        ###         )

    async def update_page_graph(
        self,
        page_relationships: dict[tuple[str, str], dict[str, Any]],
    ) -> None:
        """Update page graph with new relationships and persist to storage."""
        # FIXME - TODO: This currently updates the in-memory graph and then persists the entire graph to storage.
        # This may be susceptible to race conditions if multiple agents are updating the graph concurrently.
        # We should consider implementing a more robust graph update mechanism that can handle concurrent updates,
        # such as using graph databases or implementing locking mechanisms.
        tenant_id = require_tenant_id()
        colony_id = require_colony_id()
        graph_key = f"{tenant_id}:{colony_id}:graph"
        if not self._page_graphs.get(graph_key) or len(self._page_graphs[graph_key].nodes) == 0:
            return

        graph = self._page_graphs[graph_key]

        for (src, tgt), rel_info in page_relationships.items():
            if graph.has_edge(src, tgt):
                edge_data = graph.get_edge_data(src, tgt)
                edge_data.update(rel_info)
            else:
                graph.add_edge(src, tgt, **rel_info)

        # Persist to storage
        await self.store_page_graph(
            graph_data=graph
        )
        logger.info(f"Persisted page graph for {tenant_id}:{colony_id}")

    async def get_page_neighbors(
        self,
        page_id: str,
        max_neighbors: int = 5,
        relationship_types: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Get nearest neighbor pages."""
        tenant_id = require_tenant_id()
        colony_id = require_colony_id()
        graph_key = f"{tenant_id}:{colony_id}:graph"
        if not self._page_graphs.get(graph_key) or len(self._page_graphs[graph_key].nodes) == 0:
            return []

        graph = self._page_graphs[graph_key]

        if not graph or page_id not in graph:
            return []

        neighbors = []
        for neighbor in graph.successors(page_id):
            edge_data = graph.get_edge_data(page_id, neighbor)
            weight = edge_data.get("weight", 0.5)
            neighbors.append((neighbor, weight))

        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors[:max_neighbors]


