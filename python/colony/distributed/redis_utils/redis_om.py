from __future__ import annotations

import asyncio
import os
import logging
import operator
import pickle
import time
import uuid
import zlib
import json
import aiofiles
from collections.abc import Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import reduce, wraps
from typing import (
    Any,
    TypeVar,
    get_args,
    AsyncIterator,
    AsyncGenerator,
    Literal,
    Callable,
    Awaitable,
)

from pydantic import BaseModel, Field, field_validator
from redis.asyncio import Redis
from redis.asyncio.client import Pipeline, PubSub

from .client import RedisClient
from ..metrics.redis_om import (
    QUERY_PLANNING_DURATION,
    QUERY_EXECUTION_DURATION,
    QUERY_CACHE_HITS,
    QUERY_CACHE_MISSES,
    QUERY_OPTIMIZATION_SAVINGS
)
from ..metrics.common import record_duration
from ..metrics.redis_om import RedisOMMetricsMonitor

logger = logging.getLogger(__name__)

# Some potential future improvements (added as TODOs):
# 1. Add support for more complex optimizations:
# - Optimize range queries on sorted sets
# - Optimize IN queries using sets
# - Support query rewriting for better optimization
# 2. Add more sophisticated caching:
# - Cache intermediate results for common subqueries
# - Add cache warming for frequent queries
# - Add cache eviction strategies
# 3. Add more performance features:
# - Circuit breakers for expensive queries
# - Query timeout support
# - Query cost budgets
# - Query prioritization
# 4. Add more monitoring:
# - Query pattern analysis
# - Cache effectiveness metrics
# - Optimization effectiveness metrics
# - Resource usage tracking


# Additional TODOs for further optimization:
# 1. Implement selective cache invalidation:
# - Track query dependencies on fields
# - Only invalidate affected queries
# - Use Bloom filters to track dependencies
# 2. Add more sophisticated size estimation:
# - Use index statistics
# - Track query patterns
# - Use cardinality estimation algorithms
# 3. Add memory monitoring:
# - Track Redis memory usage
# - Add adaptive thresholds
# - Implement emergency eviction
# 4. Add query optimization:
# - Rewrite queries for better performance
# - Use index statistics for planning
# - Add query cost budgets

T = TypeVar("T", bound=BaseModel)

metrics = RedisOMMetricsMonitor()

def track_operation(operation: str):
    """Decorator to track Redis operation metrics"""
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            start_time = time.time()
            try:
                result = await func(self, *args, **kwargs)
                metrics.REDIS_OP_COUNT.labels(
                    operation=operation,
                    status="success",
                    namespace=self.namespace
                ).inc()
                return result
            except Exception as e:
                metrics.REDIS_OP_COUNT.labels(
                    operation=operation,
                    status="error",
                    namespace=self.namespace
                ).inc()
                raise
            finally:
                duration = time.time() - start_time
                metrics.REDIS_OP_DURATION.labels(
                    operation=operation,
                    namespace=self.namespace
                ).observe(duration)
        return wrapper
    return decorator

"""
The type of the index data structure is determined by the `IndexType` enum:
- `NONE`: no index
- `SORTED`: A set of IDs sorted by the field value. Use this to specify the score field.
- `SET`: A set of IDs corresponding to each possible field value. Use this for enum-like (categorical) fields.

The name of the index (Redis key) depends on the IndexType:
- `NONE`: no index
- `SORTED`: model name + field name + optional prefix
- `SET`: model name + field name + optional prefix + value
"""


class LockAcquisitionError(Exception):
    """Exception raised when a lock cannot be acquired"""
    pass


class IndexType(Enum):
    NONE = "none"
    SORTED = "sorted"  # For numeric comparisons
    SET = "set"  # For membership tests


class IndexMetadata(BaseModel):
    """Metadata for indexed fields, supporting nested paths"""

    field_path: list[str] | str = Field(
        description="Field path as list (of field names forming the path) or dot-notation string"
    )
    index_type: IndexType = Field(description="Type of index to create")
    prefix: str = Field(default="", description="Prefix for the index key")
    exclude_nested: bool = Field(
        default=False, description="Whether to exclude nested fields under this path"
    )
    ttl: int | None = Field(
        default=None,
        description="TTL in seconds, None for no expiry",
        ge=1,  # Must be positive
    )
    is_immutable: bool = Field(  # TODO: Use this to optimize index updates
        default=True,
        description="If true, the indexed field will never change even if "
        "the item is updated. So, there is no need to remove the item from the index.",
    )

    @field_validator("field_path")
    @classmethod
    def validate_field_path(cls, v: list[str] | str) -> list[str]:
        """Convert string path to list and validate"""
        if isinstance(v, str):
            return v.split(".")
        return v

    @field_validator("index_type", mode="before")
    @classmethod
    def validate_index_type(cls, v: Any) -> IndexType:
        """Convert string to IndexType enum"""
        if isinstance(v, str):
            try:
                return IndexType[v.upper()]
            except KeyError:
                raise ValueError(f"Invalid index_type: {v}")
        return v

    @property
    def field_name(self) -> str:
        return ".".join(self.field_path)


class QueryOp(Enum):
    EQ = "eq"
    GT = "gt"
    LT = "lt"
    GTE = "gte"
    LTE = "lte"
    AND = "and"
    OR = "or"
    IN = "in"
    FIELD_ACCESS = "field_access"

    def get_operator(self):
        """Get corresponding operator function"""
        return {
            QueryOp.EQ: operator.eq,
            QueryOp.GT: operator.gt,
            QueryOp.LT: operator.lt,
            QueryOp.GTE: operator.ge,
            QueryOp.LTE: operator.le,
        }.get(self)


class QueryExpr(BaseModel):
    """
    Query expression for building and representing queries.
    Also serves as a descriptor for field access when building queries.
    """

    op: QueryOp
    field: str | None = None
    value: Any = None
    left: QueryExpr | None = None
    right: QueryExpr | None = None
    model_cls: type[BaseModel] | None = None  # Used only during query building

    def __get__(self, obj: Any, objtype: type[BaseModel]) -> QueryExpr:
        """Support field access for building queries from model class"""
        return QueryExpr(op=QueryOp.FIELD_ACCESS, model_cls=objtype)

    def __getattr__(self, name: str) -> QueryExpr:
        """Support field access for building queries: q.field.nested"""
        if self.op != QueryOp.FIELD_ACCESS:
            raise AttributeError(
                f"Cannot access attribute {name} on non-field-access expression"
            )

        # Build field path
        if self.field is None:
            new_path = name
        else:
            new_path = f"{self.field}.{name}"

        return QueryExpr(
            op=QueryOp.FIELD_ACCESS, field=new_path, model_cls=self.model_cls
        )

    def __eq__(self, other: Any) -> QueryExpr:
        if self.op != QueryOp.FIELD_ACCESS:
            return super().__eq__(other)
        return QueryExpr(op=QueryOp.EQ, field=self.field, value=other)

    def __gt__(self, other: Any) -> QueryExpr:
        if self.op != QueryOp.FIELD_ACCESS:
            raise TypeError("'>' not supported between instances of 'QueryExpr'")
        return QueryExpr(op=QueryOp.GT, field=self.field, value=other)

    def __lt__(self, other: Any) -> QueryExpr:
        if self.op != QueryOp.FIELD_ACCESS:
            raise TypeError("'<' not supported between instances of 'QueryExpr'")
        return QueryExpr(op=QueryOp.LT, field=self.field, value=other)

    def __ge__(self, other: Any) -> QueryExpr:
        if self.op != QueryOp.FIELD_ACCESS:
            raise TypeError("'>=' not supported between instances of 'QueryExpr'")
        return QueryExpr(op=QueryOp.GTE, field=self.field, value=other)

    def __le__(self, other: Any) -> QueryExpr:
        if self.op != QueryOp.FIELD_ACCESS:
            raise TypeError("'<=' not supported between instances of 'QueryExpr'")
        return QueryExpr(op=QueryOp.LTE, field=self.field, value=other)

    def __and__(self, other: QueryExpr) -> QueryExpr:
        return QueryExpr(op=QueryOp.AND, left=self, right=other)

    def __or__(self, other: QueryExpr) -> QueryExpr:
        return QueryExpr(op=QueryOp.OR, left=self, right=other)


class QueryPlanStep(BaseModel):
    """A step in the query execution plan"""
    op: QueryOp
    keys: list[str] = Field(default_factory=list)  # Redis keys to operate on
    temp_key: str | None = None  # Temporary key for storing intermediate results
    cost: float = 0.0  # Estimated cost of operation

class QueryPlan(BaseModel):
    """Query execution plan with optimization metadata"""
    steps: list[QueryPlanStep] = Field(default_factory=list)
    temp_keys: set[str] = Field(default_factory=set)
    estimated_cost: float = 0.0

    def add_step(self, step: QueryPlanStep):
        self.steps.append(step)
        self.estimated_cost += step.cost
        if step.temp_key:
            self.temp_keys.add(step.temp_key)

class RedisIndex:
    """Decorator to mark fields for indexing"""

    def __init__(
        self,
        prefix: str = "",
        exclude_nested: bool = False,  # Option to exclude nested model fields from indexing - Default for fields without explicit setting
        ttl: int | None = None,  # Default TTL in seconds for model and indices
        indices: Sequence[dict[str, Any] | IndexMetadata]
        | None = None,  # External index definitions
        query_builder_name: str = "q",  # Name of the query builder attribute
    ):
        self.prefix = prefix
        self.default_exclude_nested = exclude_nested
        self.default_ttl = ttl
        self.query_builder_name = query_builder_name
        # Convert and validate indices
        self.external_indices = []
        if indices:
            for idx in indices:
                if isinstance(idx, dict):
                    self.external_indices.append(IndexMetadata.model_validate(idx))
                elif isinstance(idx, IndexMetadata):
                    self.external_indices.append(idx)
                else:
                    raise ValueError(f"Invalid index specification: {idx}")

    def __call__(self, cls: type[BaseModel]) -> type[BaseModel]:
        # Store index metadata on the model class
        if not hasattr(cls, "__redis_indices__"):
            cls.__redis_indices__ = []
            cls.__redis_ttl__ = self.default_ttl

            # Check for name collision before adding query builder
            if hasattr(cls, self.query_builder_name):
                raise ValueError(
                    f"Cannot add query builder: class {cls.__name__} already has an attribute "
                    f"named '{self.query_builder_name}'. Use a different name via query_builder_name parameter."
                )
            # Add query builder as class variable with the specified name
            setattr(cls, self.query_builder_name, QueryExpr(op=QueryOp.FIELD_ACCESS))

        # Process external index definitions first
        for idx in self.external_indices:
            # Apply decorator defaults if not specified
            if self.prefix and not idx.prefix:
                idx.prefix = self.prefix
            # Use decorator defaults if not specified
            if idx.ttl is None:
                idx.ttl = self.default_ttl
            if not idx.exclude_nested:
                idx.exclude_nested = self.default_exclude_nested
            cls.__redis_indices__.append(idx)

        # Then recursively process model fields for any additional indices
        self._process_model_fields(cls, [], cls.model_fields)
        return cls

    def _is_collection_of_models(
        self, field_type: Any
    ) -> tuple[bool, type[BaseModel] | None]:
        """Check if field is a collection of BaseModel"""
        try:
            # Get collection item type
            args = get_args(field_type)
            if not args:
                return False, None

            item_type = args[0]
            if isinstance(item_type, type) and issubclass(item_type, BaseModel):
                return True, item_type
        except Exception:
            pass
        return False, None

    def _process_model_fields(
        self, cls: type[BaseModel], path: list[str], fields: dict
    ) -> None:
        """Recursively process fields to find indexed fields"""
        for field_name, field in fields.items():
            current_path = path + [field_name]

            # Check if field is indexed
            if (
                field.json_schema_extra is not None
                and "redis_index" in field.json_schema_extra
            ):
                cls.__redis_indices__.append(
                    IndexMetadata(
                        field_path=current_path,
                        index_type=field.json_schema_extra["redis_index"],
                        prefix=field.json_schema_extra.get("prefix", ""),
                        exclude_nested=field.json_schema_extra.get(
                            "exclude_nested", self.default_exclude_nested
                        ),
                        ttl=field.json_schema_extra.get("ttl", self.default_ttl),
                    )
                )

            # Process nested fields if not excluded at this level
            exclude_nested = (field.json_schema_extra or {}).get(
                "exclude_nested", self.default_exclude_nested
            )
            if not exclude_nested:
                # Check if field is a nested model
                if isinstance(field.annotation, type) and issubclass(
                    field.annotation, BaseModel
                ):
                    self._process_model_fields(
                        cls,  # The __redis_indices__ will always be on the top-level class
                        current_path,
                        field.annotation.model_fields,
                    )

                # Check if field is a collection of models
                is_collection, model_type = self._is_collection_of_models(
                    field.annotation
                )
                if is_collection and model_type:
                    self._process_model_fields(
                        cls, current_path, model_type.model_fields
                    )


class QueryCacheInvalidationStrategy(Enum):
    """Strategy for invalidating query cache entries"""
    NONE = "none"  # No invalidation
    IMMEDIATE = "immediate"  # Invalidate immediately on write
    PERIODIC = "periodic"  # Invalidate periodically based on TTL
    SELECTIVE = "selective"  # Invalidate only affected queries


class QueryPlannerConfig(BaseModel):
    query_cache_ttl: int | None = None  # TTL for query cache in seconds
    max_chain_length: int = 10  # Max length of AND/OR chains to optimize
    enable_query_cache: bool = True  # Whether to enable query result caching
    enable_query_optimization: bool = True  # Whether to enable query optimization
    invalidation_strategy: QueryCacheInvalidationStrategy = QueryCacheInvalidationStrategy.IMMEDIATE
    max_cached_queries: int = 10000  # Maximum number of cached queries per model
    max_result_size: int = 10000  # Maximum number of IDs in cached results
    compression_threshold: int = 1000  # Compress cached results above this size
    enable_compression: bool = True  # Whether to enable compression for large results
    circuit_breaker_threshold: int = 100000  # Maximum number of IDs to process in memory


class QueryCache:
    def __init__(
        self,
        redis_client: RedisClient,
        namespace: str = "",
        config: QueryPlannerConfig | None = None,
    ):
        self.redis_client = redis_client
        self.namespace = namespace
        self.config = config or QueryPlannerConfig()
        self.compression = zlib if self.config.enable_compression else None

    def _build_query_cache_key(self, model_cls: type[BaseModel], query: QueryExpr) -> str:
        """Build cache key for query results"""
        return f"{self.namespace}:qcache:{model_cls.__name__}:{hash(str(query))}"

    def _build_model_cache_key(self, model_cls: type[BaseModel]) -> str:
        """Build key for tracking cached queries for a model"""
        return f"{self.namespace}:qcache:{model_cls.__name__}:queries"

    async def get_results(
        self, model_cls: type[BaseModel], query: QueryExpr
    ) -> set[str] | None:
        """Get cached query results if available"""
        if not self.config.enable_query_cache:
            return None

        cache_key = self._build_query_cache_key(model_cls, query)
        async with self.redis_client.get_redis_connection() as redis:
            cached = await redis.get(cache_key)
            if cached:
                QUERY_CACHE_HITS.add(1)
                if self.compression and len(cached) > self.config.compression_threshold:
                    cached = self.compression.decompress(cached)
                return pickle.loads(cached)
            QUERY_CACHE_MISSES.add(1)
            return None

    async def cache_query_results(
        self, model_cls: type[BaseModel], query: QueryExpr, results: set[str]
    ):
        """Cache query results with TTL and size limits"""
        if not self.config.enable_query_cache:
            return

        # Skip caching if result set is too large
        if len(results) > self.config.max_result_size:
            return

        cache_key = self._build_query_cache_key(model_cls, query)
        model_cache_key = self._build_model_cache_key(model_cls)

        async with self.redis_client.get_pipeline() as (pipe, redis):
            # Track this query in the model's cached queries set
            pipe.sadd(model_cache_key, cache_key)

            # Enforce max cached queries limit with LRU-like eviction
            pipe.scard(model_cache_key)
            num_cached_queries = await pipe.execute()

            if num_cached_queries > self.config.max_cached_queries:
                # Evict oldest query (approximated by taking first from set)
                if old_key := await redis.spop(model_cache_key):
                    await redis.delete(old_key)

            # Serialize and optionally compress results
            data = pickle.dumps(results)
            if (
                self.compression
                and len(data) > self.config.compression_threshold
            ):
                data = self.compression.compress(data)

            if self.config.query_cache_ttl:
                await redis.setex(
                    cache_key,
                    self.config.query_cache_ttl,
                    data
                )
            else:
                await redis.set(cache_key, data)

    async def invalidate_model_cache(
        self,
        model_cls: type[BaseModel],
        field_paths: list[str] | None = None
    ):
        """Invalidate all cached queries for a model"""
        if not self.config.enable_query_cache:
            return

        model_cache_key = self._build_model_cache_key(model_cls)

        async with self.redis_client.get_pipeline() as (pipe, redis):
            # Get all cached query keys for this model
            cached_keys = await redis.smembers(model_cache_key)

            if cached_keys:
                # Delete all cached queries and the tracking set
                for key in cached_keys:
                    pipe.delete(key.decode())
                pipe.delete(model_cache_key)
                await pipe.execute()



class ItemCheckinResult(BaseModel):
    """
    Detailed result of a checkin operation.
    """
    item_id: str = Field(description="The ID of the item that was updated")
    exists: bool = Field(description="Whether the item exists in the database")
    success: bool = Field(description="Whether the item was successfully updated")
    version_token: str | None = Field(description="The new version token for the item")
    error_type: str | None = Field(default=None, description="The type of error that occurred, if any")
    error_message: str | None = Field(default=None, description="The message of the error that occurred, if any")

class SyncStats(BaseModel):
    """Statistics for a sync operation"""
    processed: int = 0  # Total items processed
    successful: int = 0 # Items successfully updated
    failed: int = 0     # Items that failed to update
    added: int = 0      # New items added to set
    removed: int = 0    # Items removed from set
    updated: int = 0    # Items whose values changed

class SyncResult(BaseModel):
    """Result of a sync operation"""
    stats: SyncStats
    duration: float  # Total time taken
    results: list[ItemCheckinResult]  # Raw results from checkin_items

class BatchError(Exception):
    """Raised when a batch operation fails in all-or-nothing mode"""
    pass

@RedisIndex(prefix="time_series", query_builder_name="db")
class TimeSeriesItem(BaseModel):
    """An item in a time series"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="The ID of the item")
    name: str = Field(description="The name of the time series", json_schema_extra={"redis_index": IndexType.SET})
    value: Any = Field(description="The value of the item")
    timestamp: datetime = Field(description="The timestamp of the item", json_schema_extra={"redis_index": IndexType.SORTED})

@RedisIndex(prefix="bounded_list", query_builder_name="db")
class BoundedList(BaseModel):
    """A bounded list of items"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="The ID of the list")
    name: str = Field(description="The name of the list", json_schema_extra={"redis_index": IndexType.SET})
    values: list[Any] = Field(description="The values of the list")





class DistributedStateUpdate(BaseModel):
    """
    Update to an inference job state.
    """

    topic: str = Field(..., description="The topic to update: metadata, results, etc.")
    timestamp: float
    type: Literal["update", "initialization"]
    replace_all: bool = False
    data: dict[str, Any]


class DistributedStateSubscriber:
    """
    A subscriber for state updates of an inference job.
    """

    def __init__(self, redis_client: RedisClient, channel_key: str):
        self.redis_client = redis_client
        self.channel_key = channel_key
        self._subscriber: PubSub | None = None
        self._listener_task: asyncio.Task | None = None
        self._callback: Callable[
            [DistributedStateUpdate, Exception | None], bool
        ] | None = None

    async def start(
        self,
        callback: Callable[[DistributedStateUpdate, Exception | None], bool],
    ):
        """Start listening for state updates."""
        if self._subscriber is None:
            # Create Redis connection for Pub/Sub
            async with self.redis_client.get_redis_connection() as conn:
                self._subscriber = conn.pubsub()

            # Start listener in background
            await self._subscriber.subscribe(self.channel_key)
            self._listener_task = asyncio.create_task(self._listen_for_updates())
        self._callback = callback

    async def cancel(self):
        """Cancel subscriber and cleanup resources."""
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        if self._subscriber:
            await self._subscriber.unsubscribe()
            await self._subscriber.close()
        if self.redis_client:
            await self.redis_client.cleanup()

    async def _listen_for_updates(self):
        """Listen for state updates from Redis Pub/Sub."""
        try:
            while True:
                message = await self._subscriber.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
                if message is None:
                    continue

                update = DistributedStateUpdate.model_validate_json(message)
                if not self._callback(update, ex=None):
                    break

        except Exception as e:
            self._callback(update=None, ex=e)
        finally:
            # Cleanup
            if self._subscriber:
                await self._subscriber.unsubscribe()
                await self._subscriber.close()





class RedisOM:
    """Redis Object Mapper"""

    def __init__(
        self,
        redis_client: RedisClient,
        namespace: str = "",
        query_planner_config: QueryPlannerConfig | dict[str, Any] = QueryPlannerConfig(),
        enable_metrics: bool = True,
    ):
        self.redis_client = redis_client
        self.namespace = namespace
        self.query_planner_config = (
            query_planner_config
            if isinstance(query_planner_config, QueryPlannerConfig)
            else QueryPlannerConfig.model_validate(query_planner_config)
        )
        self.max_chain_length = self.query_planner_config.max_chain_length
        self.enable_query_optimization = self.query_planner_config.enable_query_optimization
        self.query_cache = QueryCache(
            redis_client=self.redis_client,
            namespace=self.namespace,
            config=self.query_planner_config,
        )
        self.enable_metrics = enable_metrics
        self.metrics = RedisOMMetricsMonitor()

    def _get_nested_values(self, obj: Any, path: list[str]) -> list[Any]:
        """Get value from nested object using path"""
        if isinstance(obj, (list, tuple, set)):
            # If we hit a list, map the rest of the path over each item
            return [
                val for val in (self._get_nested_values(item, path) for item in obj)
            ]
        else:
            value = getattr(obj, path[0])
            return self._get_nested_values(value, path[1:]) if path[1:] else [value]

    def _set_nested_value(self, obj: Any, path: list[str], value: Any) -> None:
        """Set value in nested object using path"""
        # TODO: Handle lists along the path
        *parent_path, last = path
        parent = reduce(lambda o, key: getattr(o, key), parent_path, obj)
        setattr(parent, last, value)

    def _build_key(self, model_cls: type[BaseModel], id: str) -> str:
        """Build Redis key with namespace"""
        model_name = model_cls.__name__.lower()
        return f"{self.namespace}:{model_name}:{id}"

    def _build_version_key(self, model_cls: type[BaseModel], id: str) -> str:
        """Build version key"""
        return f"{self._build_key(model_cls, id)}:version"

    def _build_lock_key(self, model_cls: type[BaseModel], id: str) -> str:
        """Build Redis key for item lock."""
        return f"{self.namespace}:{model_cls.__name__}:lock:{id}"

    def _build_index_key(self, model_cls: type[BaseModel], index: IndexMetadata) -> str:
        """Build index key"""
        model_name = model_cls.__name__.lower() # TODO: Will this lead to collisions?
        return f"{self.namespace}:{model_name}:idx:{index.prefix}{index.field_name}"

    def _build_index_set_key(self, model_cls: type[BaseModel], index: IndexMetadata, value: Any) -> str:
        """Build index set key"""
        idx_key = self._build_index_key(model_cls, index)
        return f"{idx_key}:{value}"

    def _build_index_all_sets_pattern(self, model_cls: type[BaseModel], index: IndexMetadata) -> str:
        """Build index all-sets pattern"""
        idx_key = self._build_index_key(model_cls, index)
        return f"{idx_key}:*"

    def _build_state_topic_key(self, topic_name: str) -> str:
        return f"{self.namespace}:state_topic:{topic_name}"

    def _build_state_update_channel(self, topic_name: str) -> str:
        return f"{self.namespace}:state_updates:{topic_name}"

    def _get_index(self, model_cls: type[BaseModel], field_path: list[str]) -> IndexMetadata:
        try:
            return next(
                idx for idx in model_cls.__redis_indices__
                if idx.field_path == field_path
            )
        except StopIteration:
            raise ValueError(f"No index found for field path: {'.'.join(field_path)}")

    def _convert_to_sortable_value(self, value: Any) -> float:
        """Convert value to a sortable float for Redis sorted sets"""
        if isinstance(value, datetime):
            # Convert to Unix timestamp in milliseconds for better precision
            return value.timestamp() * 1000
        return float(value)

    async def _update_index_size(self, model_cls: type[BaseModel], index: IndexMetadata):
        """Update index size metric"""
        async with self.redis_client.get_redis_connection() as redis:
            size = 0
            if index.index_type == IndexType.SORTED:
                idx_key = self._build_index_key(model_cls, index)
                size = await redis.zcard(idx_key)
            elif index.index_type == IndexType.SET:
                # For SET indices, we need to sum up the sizes of all value-specific sets
                # This is expensive, so we might want to make it optional or periodic
                all_sets_pattern = self._build_index_all_sets_pattern(model_cls, index)
                async for key in redis.scan_iter(all_sets_pattern):
                    size += await redis.scard(key)

            self.metrics.REDIS_INDEX_SIZE.labels(
                model=model_cls.__name__,
                field=index.field_name,
                index_type=index.index_type.value,
                namespace=self.namespace
            ).set(size)

    async def _execute_with_semaphore(
        self, operation: Callable[[Redis], Awaitable[Any]]
    ) -> Any:
        """Execute operation with proper connection management and circuit breaker."""
        return await self.redis_client.execute_with_semaphore(operation)

    @asynccontextmanager
    async def _get_pipeline(self) -> AsyncGenerator[Pipeline, None]:
        """Get a Redis pipeline with proper connection management."""
        async with self.redis_client.get_pipeline() as (pipe, _):
            yield pipe

    @track_operation("save")
    async def save(
        self,
        obj: BaseModel,
        id: str,
        update_if_exists: bool = True,
        version_token: str | None = None,
        ttl: int | None = None,  # Override default TTL
        model_cls: type[BaseModel] | None = None,  # Optional model class for index definitions
    ) -> tuple[bool, str]:
        """
        Save object and update indices atomically with metrics and query cache invalidation.
        Returns (success, new_version_token).
        If update_if_exists=False and object exists, returns (False, None).
        If version_token provided and doesn't match, returns (False, current_version).
        """
        # Use provided model class or infer from object
        model_cls = model_cls or obj.__class__
        if not hasattr(model_cls, "__redis_indices__"):
            raise ValueError("Model class must be decorated with @RedisIndex")

        # Get model class and indices
        indices = model_cls.__redis_indices__
        model_ttl = ttl or getattr(model_cls, "__redis_ttl__", None)

        # Main object key
        obj_key = self._build_key(model_cls, id)
        version_key = self._build_version_key(model_cls, id)

        async with self.redis_client.get_pipeline() as (pipe, redis):
            # Start watching the object and its version
            await pipe.watch(obj_key, version_key)

            # Check if object exists
            exists = await redis.exists(obj_key)
            if exists:
                if not update_if_exists:
                    return False, None

                # Check version if provided
                if version_token is not None:
                    current_version = await redis.get(version_key)
                    if current_version and current_version.decode() != version_token:
                        return False, current_version.decode()

                # Get old object for index cleanup
                old_data = await redis.get(obj_key)
                old_obj = pickle.loads(old_data) if old_data else None

            # Start transaction
            pipe.multi()
            new_version = await self._save_item(obj, old_obj, id, model_ttl, pipe, model_cls=model_cls)

            try:
                await pipe.execute()
                # Update index size metrics after successful save
                for idx in indices:
                    await self._update_index_size(model_cls, idx)
                await self._invalidate_query_cache(obj, model_cls)
                return True, new_version
            except Exception:  # Includes WatchError
                return False, version_token if version_token else None

    async def _save_item(
        self,
        obj: BaseModel,
        old_obj: BaseModel | None,
        id: str,
        model_ttl: int | None,
        pipe: Pipeline,
        model_cls: type[BaseModel] | None = None,
    ) -> str:
        # Use provided model class or infer from object
        model_cls = model_cls or obj.__class__
        indices = model_cls.__redis_indices__

        # Store main object with new version and TTL
        obj_key = self._build_key(model_cls, id)
        version_key = self._build_version_key(model_cls, id)

        # Generate new version
        new_version = str(uuid.uuid4())

        pipe.set(obj_key, pickle.dumps(obj))
        if model_ttl:
            pipe.expire(obj_key, model_ttl)
        pipe.set(version_key, new_version)
        if model_ttl:
            pipe.expire(version_key, model_ttl)

        # Update indices
        for idx in indices:
            idx_key = self._build_index_key(model_cls, idx)

            # Remove old values from indices if object existed
            if old_obj:
                old_values = self._get_nested_values(old_obj, idx.field_path)
                for old_value in old_values:
                    if idx.index_type == IndexType.SORTED:
                        pipe.zrem(idx_key, id)
                    elif idx.index_type == IndexType.SET:
                        pipe.srem(f"{idx_key}:{old_value}", id)

            # Add new values to indices
            new_values = self._get_nested_values(obj, idx.field_path)
            for value in new_values:
                if idx.index_type == IndexType.SORTED:
                    pipe.zadd(
                        idx_key, {id: self._convert_to_sortable_value(value)}
                    )
                    if idx.ttl:
                        pipe.expire(idx_key, idx.ttl)
                elif idx.index_type == IndexType.SET:
                    set_key = f"{idx_key}:{value}"
                    pipe.sadd(set_key, id)
                    if idx.ttl:
                        pipe.expire(set_key, idx.ttl)
        return new_version

    @track_operation("save_batch")
    async def save_batch(
        self,
        items: list[BaseModel],
        ids: list[str],
        update_if_exists: bool = True,
        version_tokens: list[str] | None = None,
        ttl: int | None = None,  # Override default TTL
        model_cls: type[BaseModel] | None = None,  # Optional model class for index definitions
    ) -> list[ItemCheckinResult]:
        """
        Save a batch of items atomically with optimized existence checks and index updates.
        Returns list of `ItemCheckinResult` objects for each item.
        """
        # Use internal save batch with version check only if tokens provided
        return await self._save_batch_internal(
            items=items,
            ids=ids,
            version_tokens=version_tokens or [None] * len(items),
            ttl=ttl,
            all_or_nothing=False,  # save_batch is greedy (i.e., save items that either didn't exist or exist with matching versions)
            update_if_exists=update_if_exists,
            model_cls=model_cls,
        )

    async def _save_batch_internal(
        self,
        items: list[BaseModel],
        ids: list[str],
        version_tokens: list[str | None],
        ttl: int | None,
        all_or_nothing: bool,
        update_if_exists: bool = True,
        model_cls: type[BaseModel] | None = None,  # Optional model class for index definitions
    ) -> list[ItemCheckinResult]:
        """
        Internal method for batch saving items with version control.
        Returns list of `ItemCheckinResult` objects.

        Parameters:
        - items: List of items to save
        - version_tokens: List of version tokens (None if no version check needed)
        - ttl: Optional TTL override
        - all_or_nothing: If True, either all items are updated or none are
        - update_if_exists: Whether to update items that already exist
        - model_cls: Optional model class for index definitions
        """
        if not items:
            return []

        if len(items) != len(ids) or len(items) != len(version_tokens):
            raise ValueError("Length mismatch between items, ids, and version tokens")

        # Use provided model class or infer from first item
        model_cls = model_cls or items[0].__class__
        if not all(isinstance(item, model_cls) for item in items):
            raise ValueError("All items must be of same type")

        if not hasattr(model_cls, "__redis_indices__"):
            raise ValueError("Model class must be decorated with @RedisIndex")

        # Get model class and indices
        indices = model_cls.__redis_indices__
        model_ttl = ttl or getattr(model_cls, "__redis_ttl__", None)

        # Build keys for all items
        obj_keys = [self._build_key(model_cls, id) for id in ids]
        version_keys = [self._build_version_key(model_cls, id) for id in ids]
        lock_keys = [self._build_lock_key(model_cls, id) for id in ids]

        # Track results: (exists, updated, latest_version)
        results: list[ItemCheckinResult] = [
            ItemCheckinResult(item_id=id, exists=False, updated=False, version_token=None)
            for id in ids
        ]
        items_to_save: list[tuple[int, BaseModel | None]] = []  # [(index, old_obj), ...]

        async with self.redis_client.get_pipeline() as (pipe, _):
            # Start watching all keys
            await pipe.watch(*obj_keys, *version_keys, *lock_keys)

            # Check existence and versions in batch
            pipe.multi()
            for key in obj_keys:
                pipe.exists(key)
            for key in version_keys:
                pipe.get(key)
            for key in lock_keys:
                pipe.get(key)
            exists_versions_owners = await pipe.execute()

            # Process existence checks and version checks
            exists_list = exists_versions_owners[:len(items)]
            current_versions = exists_versions_owners[len(items):len(items) + len(items)]
            current_versions = [current_version.decode() if current_version else None for current_version in current_versions]
            owners = exists_versions_owners[len(items) + len(items):]
            owners = [owner.decode() if owner else None for owner in owners]

            # Get existing objects for index cleanup
            pipe.multi()
            for i, exists in enumerate(exists_list):
                if exists:
                    pipe.get(obj_keys[i])
            old_data_list = await pipe.execute()

            old_data_idx = 0
            for i, (exists, current_version, version_token, owner) in enumerate(zip(exists_list, current_versions, version_tokens, owners)):
                if exists:
                    if not update_if_exists:
                        results[i] = ItemCheckinResult(item_id=ids[i], exists=True, updated=False, version_token=current_version)
                        continue

                    if owner and owner != owner_id:
                        results[i] = ItemCheckinResult(
                            item_id=ids[i],
                            exists=True,
                            updated=False,
                            version_token=current_version,
                            error_type="lock_mismatch",
                            error_message="Lock mismatch",
                        )
                        if all_or_nothing:
                            return results
                        continue

                    # Version check logic
                    if version_token is not None and current_version and current_version != version_token:
                        # Never update if version doesn't match (do not overwrite new data with old data)
                        results[i] = ItemCheckinResult(item_id=ids[i], exists=True, updated=False, version_token=current_version,
                                                       error_type="version_mismatch", error_message="Version mismatch")
                        if all_or_nothing:
                            return results
                        continue

                    # Get old object for index cleanup
                    old_data = old_data_list[old_data_idx]
                    old_data_idx += 1
                    items_to_save.append((i, pickle.loads(old_data) if old_data else None))
                else:
                    items_to_save.append((i, None))

            if not items_to_save:
                return results

            # Start transaction for saves
            pipe.multi()

            # Store objects with versions and TTL
            new_versions = []
            for i, old_obj in items_to_save:
                new_version = await self._save_item(items[i], old_obj, ids[i], model_ttl, pipe, model_cls=model_cls)
                new_versions.append(new_version)

            try:
                await pipe.execute()

                # Update results for saved items
                for (i, _), new_version in zip(items_to_save, new_versions):
                    results[i] = ItemCheckinResult(item_id=ids[i], exists=exists_list[i], updated=True, version_token=new_version)

                # Update index size metrics after successful save
                for idx in indices:
                    await self._update_index_size(model_cls, idx)

                # Batch invalidate query cache
                if self.query_planner_config.invalidation_strategy == QueryCacheInvalidationStrategy.IMMEDIATE:
                    # Get all affected field paths
                    field_paths = set()
                    for _, item, _, _ in items_to_save:
                        field_paths.update(self._get_changed_field_paths(item))
                    # Invalidate cached queries that might be affected
                    await self.query_cache.invalidate_model_cache(
                        model_cls, list(field_paths)
                    )

            except Exception as e:  # Includes WatchError
                # Keep existing results (False, None) or (False, current_version)
                # Handle Redis watch error
                return [
                    ItemCheckinResult(
                        item_id=ids[i],
                        exists=exists_list[i],
                        updated=False,
                        version_token=current_versions[i],
                        error_type="watch_error",
                        error_message="Concurrent modification detected",
                    )
                    for i in range(len(items))
                ]

        return results

    @track_operation("delete")
    async def delete(
        self,
        model_cls: type[BaseModel],
        id: str,
    ) -> bool:
        """
        Delete object and clean up indices atomically.
        Returns True if object was deleted, False if it didn't exist.
        """
        if not hasattr(model_cls, "__redis_indices__"):
            raise ValueError("Model class must be decorated with @RedisIndex")

        # Get indices
        indices = model_cls.__redis_indices__

        # Main object key
        obj_key = self._build_key(model_cls, id)
        version_key = self._build_version_key(model_cls, id)

        async with self.redis_client.get_pipeline() as (pipe, redis):
            # Start watching the object
            await pipe.watch(obj_key)

            # Get object for index cleanup
            obj_data = await redis.get(obj_key)
            if not obj_data:
                return False

            # Get object to clean up indices
            obj = pickle.loads(obj_data)

            # Start transaction
            pipe.multi()

            # Delete object and version
            pipe.delete(obj_key)
            pipe.delete(version_key)

            # Clean up indices
            for idx in indices:
                idx_key = self._build_index_key(model_cls, idx)

                # Get values to remove from indices
                values = self._get_nested_values(obj, idx.field_path)
                for value in values:
                    if idx.index_type == IndexType.SORTED:
                        pipe.zrem(idx_key, id)
                    elif idx.index_type == IndexType.SET:
                        pipe.srem(f"{idx_key}:{value}", id)

            try:
                await pipe.execute()

                # Update index size metrics after successful delete
                for idx in indices:
                    await self._update_index_size(model_cls, idx)

                # Invalidate query cache
                if self.query_planner_config.invalidation_strategy == QueryCacheInvalidationStrategy.IMMEDIATE:
                    # Get all affected field paths
                    field_paths = self._get_changed_field_paths(obj)
                    # Invalidate cached queries that might be affected
                    await self.query_cache.invalidate_model_cache(
                        model_cls, field_paths
                    )

                return True
            except Exception:  # Includes WatchError
                return False

    @track_operation("delete_batch")
    async def delete_batch(
        self,
        model_cls: type[BaseModel],
        ids: list[str],
    ) -> list[bool]:
        """
        Delete multiple objects and clean up their indices atomically.
        Returns list of booleans indicating which objects were deleted.
        """
        if not ids:
            return []

        if not hasattr(model_cls, "__redis_indices__"):
            raise ValueError("Model class must be decorated with @RedisIndex")

        # Get indices
        indices = model_cls.__redis_indices__

        # Build keys
        obj_keys = [self._build_key(model_cls, id) for id in ids]
        # version_keys = [self._build_version_key(model_cls, id) for id in ids]

        # Track which objects were deleted
        results = [False] * len(ids)
        objects_to_delete: list[tuple[int, BaseModel, str]] = []  # [(index, obj, id), ...]

        async with self.redis_client.get_pipeline() as (pipe, redis):
            # Start watching all objects
            await pipe.watch(*obj_keys)

            # Get objects for index cleanup
            pipe.multi()
            for key in obj_keys:
                pipe.get(key)
            obj_data_list = await pipe.execute()

            # Process objects to delete
            for i, (id, obj_data) in enumerate(zip(ids, obj_data_list)):
                if obj_data:
                    obj = pickle.loads(obj_data)
                    objects_to_delete.append((i, obj, id))

            if not objects_to_delete:
                return results

            # Start transaction
            pipe.multi()

            # Delete objects and versions
            for _, _, id in objects_to_delete:
                pipe.delete(self._build_key(model_cls, id))
                pipe.delete(self._build_version_key(model_cls, id))

            # Clean up indices
            for idx in indices:
                idx_key = self._build_index_key(model_cls, idx)

                for _, obj, id in objects_to_delete:
                    # Get values to remove from indices
                    values = self._get_nested_values(obj, idx.field_path)
                    for value in values:
                        if idx.index_type == IndexType.SORTED:
                            pipe.zrem(idx_key, id)
                        elif idx.index_type == IndexType.SET:
                            pipe.srem(f"{idx_key}:{value}", id)

            try:
                await pipe.execute()

                # Update results
                for i, _, _ in objects_to_delete:
                    results[i] = True

                # Update index size metrics after successful deletes
                for idx in indices:
                    await self._update_index_size(model_cls, idx)

                # Batch invalidate query cache
                if self.query_planner_config.invalidation_strategy == QueryCacheInvalidationStrategy.IMMEDIATE:
                    # Get all affected field paths
                    field_paths = set()
                    for _, obj, _ in objects_to_delete:
                        field_paths.update(self._get_changed_field_paths(obj))
                    # Invalidate cached queries that might be affected
                    await self.query_cache.invalidate_model_cache(
                        model_cls, list(field_paths)
                    )

            except Exception:  # Includes WatchError
                pass

        return results

    @track_operation("remove_all")
    async def remove_all(self, model_cls: type[BaseModel]):
        """
        Remove all objects and associated data for a particular model type.
        This includes:
        - All objects
        - All version keys
        - All index entries
        - All query cache entries
        """
        if not hasattr(model_cls, "__redis_indices__"):
            raise ValueError("Model class must be decorated with @RedisIndex")

        async with self.redis_client.get_pipeline() as (_, redis):
            # Delete all keys for this model (objects, versions, indices) in one shot
            model_pattern = f"{self.namespace}:{model_cls.__name__.lower()}:*"
            all_keys = await redis.keys(model_pattern)
            if all_keys:
                await redis.delete(*all_keys)

            # Clean up query cache separately since it uses a different key pattern
            if self.query_planner_config.invalidation_strategy != QueryCacheInvalidationStrategy.NONE:
                await self.query_cache.invalidate_model_cache(model_cls)

            # Reset index size metrics
            for idx in model_cls.__redis_indices__:
                self.metrics.REDIS_INDEX_SIZE.labels(
                    model=model_cls.__name__,
                    field=".".join(idx.field_path),
                    index_type=idx.index_type.value,
                    namespace=self.namespace
                ).set(0)

    async def _invalidate_query_cache(self, obj: BaseModel, model_cls: type[BaseModel] | None = None):
        if self.query_planner_config.invalidation_strategy == QueryCacheInvalidationStrategy.IMMEDIATE:
            # Get affected field paths from the object's changes
            field_paths = self._get_changed_field_paths(obj)
            # Invalidate cached queries that might be affected
            await self.query_cache.invalidate_model_cache(
                model_cls or obj.__class__, field_paths
            )

    def _get_changed_field_paths(self, obj: BaseModel) -> list[str]:
        """Get paths of fields that were changed in the object"""
        # TODO: Implement change tracking in BaseModel to make this more efficient
        # For now, conservatively return all indexed fields
        return [
            ".".join(idx.field_path)
            for idx in obj.__class__.__redis_indices__
        ]

    async def _get(
        self, model_cls: type[T], id: str, pipe: Pipeline
    ) -> tuple[T | None, str | None]:
        """
        Get object by ID with its version token.
        If pipe is provided, queue the get operations in the pipeline instead of executing them.
        """
        obj_key = self._build_key(model_cls, id)
        version_key = self._build_version_key(model_cls, id)
        pipe.get(obj_key)
        pipe.get(version_key)  # Actual results will be processed by caller

    @track_operation("get")
    async def get(self, model_cls: type[T], id: str) -> tuple[T | None, str | None]:
        """Get object by ID with its version token"""
        async with self.redis_client.get_pipeline() as (pipe, _):
            await self._get(model_cls, id, pipe)
            data, version = await pipe.execute()
            if data:
                # data is pickled bytes - do not decode
                # version is a string - needs decode
                return pickle.loads(data), version.decode() if version else None
            return None, None

    @track_operation("get_batch")
    async def get_batch(self, model_cls: type[T], ids: list[str]) -> list[tuple[T, str]]:
        """Get objects by IDs, returns list of (object, version) tuples"""
        async with self.redis_client.get_pipeline() as (pipe, _):
            return await self._get_batch(model_cls, ids, pipe)

    async def _get_batch(self, model_cls: type[T], ids: list[str], pipe: Pipeline) -> list[tuple[T, str]]:
        """Get objects by IDs, returns list of (object, version) tuples"""
        # Queue up all get operations
        for id in ids:
            await self._get(model_cls, id, pipe)
        results = await pipe.execute()
        # return [
        #     (pickle.loads(data), (version.decode() if version else None))
        #     for data, version in zip(results[::2], results[1::2])
        # ]

        # Process results in pairs (object, version)
        objects = []
        for i in range(0, len(results), 2):
            obj_data = results[i]  # Pickled bytes
            version_data = results[i + 1]  # String bytes
            if obj_data:
                # obj_data is pickled bytes - do not decode
                # version_data is a string - needs decode
                obj = pickle.loads(obj_data)
                version = version_data.decode() if version_data else None
                objects.append((obj, version))

        return objects

    async def get_all_item_ids(self, model_cls: type[T]) -> set[str]:
        """Get IDs of all items of a particular model type."""
        async with self.redis_client.get_pipeline() as (_, redis):
            all_keys = await redis.keys(self._build_key(model_cls, "*"))
            return [key.decode().split(":")[-1] for key in all_keys]

    @track_operation("find")
    async def find(self, model_cls: type[T], query: QueryExpr) -> list[tuple[T, str]]:
        """Find objects matching query"""
        async for batch in self.find_batches(model_cls, query, batch_size=0):
            return batch

    @track_operation("find_batches")
    async def find_batches(self, model_cls: type[T], query: QueryExpr, batch_size: int = 1000, use_planner=True) -> AsyncIterator[list[tuple[T, str]]]:
        """Find objects matching query, returns batches of (object, version) tuples"""
        async with self.redis_client.get_pipeline() as (pipe, redis):
            ids = await self._find_ids(model_cls, query, redis, use_planner=use_planner)

            # Fetch all objects and versions in a single pipeline (single roundtrip)
            if not ids:
                return

            if batch_size > 0:
                for i in range(0, len(ids), batch_size):
                    batch = ids[i:i + batch_size]
                    yield await self._get_batch(model_cls, batch, pipe)
            else:
                yield await self._get_batch(model_cls, ids, pipe)

    @track_operation("find_ids")
    async def find_ids(self, model_cls: type[T], query: QueryExpr, use_planner=True) -> set[str]:
        """Find IDs matching query"""
        async with self.redis_client.get_pipeline() as (_, redis):
            return await self._find_ids(model_cls, query, redis, use_planner=use_planner)

    async def _find_ids(
        self,
        model_cls: type[T],
        query: QueryExpr,
        redis: Redis,
        use_query_planner: bool = True
    ) -> set[str]:
        """Find IDs matching query with circuit breaker"""
        # Check if query might return too many results
        if await self._estimate_result_size(model_cls, query, redis) > self.query_planner_config.circuit_breaker_threshold:
            raise ValueError(
                f"Query would return too many results (>{self.query_planner_config.circuit_breaker_threshold}). "
                "Please add more restrictive filters."
            )

        # Check cache first
        if cached := await self.query_cache.get_results(model_cls, query):
            return list(cached)

        if not use_query_planner:
            ids = await self._execute_complex_query(model_cls, query, redis, top_level=True)
            # Cache results
            await self.query_cache.cache_query_results(model_cls, query, ids)
            return ids

        # Create and execute optimized plan
        async with record_duration(QUERY_PLANNING_DURATION):
            plan = self._create_query_plan(model_cls, query)

        # Record optimization savings
        if self.enable_query_optimization:
            naive_cost = self._estimate_naive_cost(query)
            savings = naive_cost - plan.estimated_cost
            QUERY_OPTIMIZATION_SAVINGS.record(savings)

        # Execute plan
        async with record_duration(QUERY_EXECUTION_DURATION):
            results = await self._execute_query_plan(model_cls, plan, redis)

        # Cache results
        await self.query_cache.cache_query_results(model_cls, query, results)
        return list(results)

    async def _estimate_result_size(
        self,
        model_cls: type[BaseModel],
        query: QueryExpr,
        redis: Redis
    ) -> int:
        """Estimate number of results a query might return"""
        # TODO: Implement smarter estimation based on index statistics
        # For now, use a simple heuristic based on index cardinality
        try:
            if query.op == QueryOp.EQ and query.field:
                # For equality, check the specific value's set size
                key = await self._get_set_key_for_eq(model_cls, query)
                if key:
                    return await redis.scard(key)

            # For other operations, check the total index size
            if query.field:
                field_path = query.field.split(".")
                index = self._get_index(model_cls, field_path)

                if index.index_type == IndexType.SET:
                    # Sum up all value-specific set sizes
                    total = 0
                    all_sets_pattern = self._build_index_all_sets_pattern(model_cls, index)
                    async for key in redis.scan_iter(all_sets_pattern):
                        total += await redis.scard(key)
                    return total
                elif index.index_type == IndexType.SORTED:
                    idx_key = self._build_index_key(model_cls, index)
                    return await redis.zcard(idx_key)

        except Exception:
            pass

        # If estimation fails, assume the worst
        return float('inf')

    def _create_query_plan(
        self, model_cls: type[BaseModel], query: QueryExpr
    ) -> QueryPlan:
        """Create optimized query execution plan"""
        if not self.enable_query_optimization:
            # Return simple plan with single step
            return QueryPlan(steps=[
                QueryPlanStep(op=query.op)
            ])

        plan = QueryPlan()

        # Find chains of AND/OR operations
        chains = self._find_operation_chains(query)

        # Create steps for each chain
        for chain_op, chain in chains:
            if len(chain) > 1:
                # Create temporary key for chain result
                temp_key = f"{self.namespace}:temp:{uuid.uuid4()}"

                # Get Redis keys for all equality comparisons
                keys = []
                for expr in chain:
                    if key := self._get_set_key_for_eq(model_cls, expr):
                        keys.append(key)

                if len(keys) > 1:
                    # Add step to execute chain in single Redis operation
                    plan.add_step(QueryPlanStep(
                        op=chain_op,
                        keys=keys,
                        temp_key=temp_key,
                        cost=len(keys) - 1  # Cost is number of operations needed
                    ))
                    continue

            # Fall back to regular execution for this part
            plan.add_step(QueryPlanStep(op=chain_op))

        return plan

    def _find_operation_chains(
        self, query: QueryExpr | None, current_op: QueryOp | None = None
    ) -> list[tuple[QueryOp, list[QueryExpr]]]:
        """Find chains of AND/OR operations in query tree to convert
        chains of AND operations into single SINTER calls and chains
        of OR operations into single SUNION calls"""
        if not query:
            return []

        chains = []
        current_chain = []

        def process_chain():
            if current_chain:
                chains.append((current_op, current_chain.copy()))
                current_chain.clear()

        if query.op in (QueryOp.AND, QueryOp.OR):
            # Start new chain if operation changes
            if current_op and query.op != current_op:
                process_chain()

            # Add left and right to current chain
            if len(current_chain) < self.max_chain_length:
                current_chain.extend(self._collect_chain_nodes(query))

            # Process any remaining nodes
            process_chain()

            # Recursively find chains in remaining parts
            chains.extend(self._find_operation_chains(query.left, query.op))
            chains.extend(self._find_operation_chains(query.right, query.op))
        else:
            # Leaf node
            if current_op:
                current_chain.append(query)
                process_chain()

        return chains

    def _collect_chain_nodes(
        self, query: QueryExpr | None, chain: list[QueryExpr] | None = None
    ) -> list[QueryExpr]:
        """Collect all nodes in an AND/OR chain"""
        if chain is None:
            chain = []

        if not query:
            return chain

        if query.op == QueryOp.EQ:
            chain.append(query)
        elif query.op in (QueryOp.AND, QueryOp.OR):
            self._collect_chain_nodes(query.left, chain)
            self._collect_chain_nodes(query.right, chain)

        return chain

    async def _execute_query_plan(
        self, model_cls: type[BaseModel], plan: QueryPlan, redis: Redis
    ) -> set[str]:
        """Execute optimized query plan"""
        try:
            results = set()

            for step in plan.steps:
                if step.keys and step.temp_key:
                    # Execute optimized chain
                    if step.op == QueryOp.AND:
                        await redis.sinterstore(step.temp_key, *step.keys)
                    else:  # OR
                        await redis.sunionstore(step.temp_key, *step.keys)

                    # Get results
                    ids = await redis.smembers(step.temp_key)
                    step_results = {id.decode() for id in ids}
                else:
                    # Execute regular query
                    step_results = await self._execute_query(model_cls, step.op, redis)

                # Combine results
                if not results:
                    results = step_results
                else:
                    if step.op == QueryOp.AND:
                        results &= step_results
                    else:  # OR
                        results |= step_results

            return results

        finally:
            # Clean up temporary keys
            if plan.temp_keys:
                await redis.delete(*plan.temp_keys)

    async def _execute_query(
        self, model_cls: type[BaseModel], query: QueryExpr, redis: Redis
    ) -> set[str]:
        """Execute query and return matching IDs"""
        # Record query operation
        self.metrics.REDIS_QUERY_COUNT.labels(
            model=model_cls.__name__,
            operation=query.op.value,
            index_type=self._get_query_index_type(model_cls, query),
            namespace=self.namespace
        ).inc()

        if query.op in (QueryOp.AND, QueryOp.OR):
            # Try to optimize if both sides are SET equality queries
            try:
                left_key  = await self._get_set_key_for_eq(model_cls, query.left)
                right_key = await self._get_set_key_for_eq(model_cls, query.right)
                if left_key and right_key:
                    # Both sides are SET equality queries, use Redis set operations
                    if query.op == QueryOp.AND:
                        ids = await redis.sinter(left_key, right_key)
                    else:  # OR
                        ids = await redis.sunion(left_key, right_key)
                    return {id.decode() for id in ids}
            except Exception:
                pass  # Fall back to regular execution if optimization fails

            # Regular execution if optimization not possible
            left_ids  = await self._execute_query(model_cls, query.left, redis)
            right_ids = await self._execute_query(model_cls, query.right, redis)
            return (
                left_ids & right_ids
                if query.op == QueryOp.AND
                else left_ids | right_ids
            )

        # Find index for field
        field_path = query.field.split(".")
        index = self._get_index(model_cls, field_path)

        if index.index_type == IndexType.SET:
            set_key = self._build_index_set_key(model_cls, index, query.value)
            if query.op == QueryOp.EQ:
                ids = await redis.smembers(set_key)
                return {id.decode() for id in ids}

        elif index.index_type == IndexType.SORTED:
            idx_key = self._build_index_key(model_cls, index)
            op_func = query.op.get_operator()
            if op_func:
                if op_func in (operator.gt, operator.ge):
                    # For > use "(value" to exclude the value
                    # For >= use "value" to include the value
                    min_value = (
                        f"({query.value}" if op_func == operator.gt else query.value
                    )
                    ids = await redis.zrangebyscore(
                        idx_key, min=min_value, max=float("inf")
                    )
                    return {id.decode() for id in ids}
                elif op_func in (operator.lt, operator.le):
                    # For < use "(value" to exclude the value
                    # For <= use "value" to include the value
                    max_value = (
                        f"({query.value}" if op_func == operator.lt else query.value
                    )
                    ids = await redis.zrangebyscore(
                        idx_key, min=float("-inf"), max=max_value
                    )
                    return {id.decode() for id in ids}

        raise ValueError(
            f"Unsupported operation {query.op} for index type {index.index_type}"
        )

    async def _execute_complex_query(
        self, model_cls: type[BaseModel], query: QueryExpr, redis: Redis, top_level: bool = True
    ) -> tuple[set[str], set[str]] | set[str]:
        """
        Execute a complex query that might need temporary storage.
        Returns (result_ids, temp_keys) where temp_keys is a set of all temporary
        Redis keys created during query execution that need cleanup.
        """
        temp_keys = set()  # Initialize before try block
        try:
            if query.op in (QueryOp.AND, QueryOp.OR):
                # Execute both sides of the query
                left_ids, left_temp_keys = await self._execute_complex_query(
                    model_cls, query.left, redis, top_level=False
                )
                right_ids, right_temp_keys = await self._execute_complex_query(
                    model_cls, query.right, redis, top_level=False
                )

                # Collect all temporary keys
                temp_keys = left_temp_keys | right_temp_keys

                # Try to optimize using Redis set operations if possible
                try:
                    left_key  = await self._get_set_key_for_eq(model_cls, query.left)
                    right_key = await self._get_set_key_for_eq(model_cls, query.right)
                    if left_key and right_key:
                        # Both sides are SET equality queries, use Redis set operations
                        result_key = f"{self.namespace}:temp:{uuid.uuid4()}"
                        if query.op == QueryOp.AND:
                            await redis.sinterstore(result_key, left_key, right_key)
                        else:  # OR
                            await redis.sunionstore(result_key, left_key, right_key)

                        # Get results but keep the temporary key for potential parent queries
                        ids = await redis.smembers(result_key)
                        temp_keys.add(result_key)
                        if top_level:
                            return {id.decode() for id in ids}
                        else:
                            return {id.decode() for id in ids}, temp_keys
                except Exception:
                    pass  # Fall back to in-memory operations

                # Do set operation in memory if optimization not possible
                result = (
                    left_ids & right_ids
                    if query.op == QueryOp.AND
                    else left_ids | right_ids
                )
                if top_level:
                    return result
                else:
                    return result, temp_keys

            # For leaf nodes, execute normally and return empty set of temp keys
            results = await self._execute_query(model_cls, query, redis)
            if top_level:
                return results
            else:
                return results, set()

        finally:
            # Clean up all temporary keys at the end of the query
            if temp_keys and top_level:
                async with self.redis_client.get_redis_connection() as redis:
                    await redis.delete(*temp_keys)

    async def _get_set_key_for_eq(
        self, model_cls: type[BaseModel], query: QueryExpr
    ) -> str | None:
        """
        If the query is a SET equality query, return its Redis key.
        Otherwise return None.
        """
        if query.op != QueryOp.EQ:
            return None

        try:
            field_path = query.field.split(".")
            index = self._get_index(model_cls, field_path)
            if index.index_type == IndexType.SET:
                return self._build_index_set_key(model_cls, index, query.value)
        except Exception:
            pass
        return None

    def _get_query_index_type(self, model_cls: type[BaseModel], query: QueryExpr) -> str:
        """Get index type used by query for metrics"""
        if query.op in (QueryOp.AND, QueryOp.OR):
            return "composite"
        try:
            field_path = query.field.split(".")
            index = self._get_index(model_cls, field_path)
            return index.index_type.value
        except Exception:
            return "unknown"

    def _estimate_naive_cost(self, query: QueryExpr | None) -> float:
        """Estimate cost of naive query execution"""
        if not query:
            return 0

        if query.op in (QueryOp.AND, QueryOp.OR):
            return (
                1 +  # Cost of combining results
                self._estimate_naive_cost(query.left) +
                self._estimate_naive_cost(query.right)
            )

        return 1  # Cost of leaf node

    @track_operation("checkout")
    async def checkout_items(
        self,
        model_cls: type[T],
        ids: list[str],
    ) -> tuple[list[T], list[str]]:
        """
        Get items by ID with their version tokens for optimistic locking.
        Returns (items, version_tokens) tuple.

        The version tokens should be passed to checkin_items() to ensure
        the items haven't been modified since they were retrieved.
        """
        if not ids:
            return [], []

        async with self.redis_client.get_pipeline() as (pipe, _):
            # Get items and their current versions
            item_version_pairs = await self._get_batch(model_cls, ids, pipe)

            # Process results, generating new version tokens if needed
            items = []
            version_tokens = []
            for item, version in item_version_pairs:
                items.append(item)
                version_tokens.append(version or str(uuid.uuid4()))  # Generate new version if none exists

            return items, version_tokens

    @track_operation("checkin")
    async def checkin_items(
        self,
        items: list[BaseModel],
        version_tokens: list[str],
        ids: list[str],
        update_if_exists: bool = True,
        ttl: int | None = None,  # Override default TTL
        model_cls: type[BaseModel] | None = None,  # Optional model class for index definitions
    ) -> list[ItemCheckinResult]:
        """
        Update items using optimistic locking.
        Returns a list of `ItemCheckinResult` objects.

        If an item was modified since `checkout_items()` was called,
        its update will fail (`False` in return list).

        Parameters:
        - `items`: List of items to update
        - `version_tokens`: List of version tokens from `checkout_items()`
        - `update_if_exists`: Whether to update items that already exist
        - `ttl`: Optional TTL override for items
        - `model_cls`: Optional model class for index definitions
        """
        # Use internal save batch with strict version checking
        return await self._save_batch_internal(
            items=items,
            ids=ids,
            version_tokens=version_tokens,
            ttl=ttl,
            all_or_nothing=True,  # For optimistic locking, all versions must match or none are saved
            update_if_exists=update_if_exists,
            model_cls=model_cls,
        )

    @track_operation("export_knowledge_base")
    async def export_knowledge_base(
        self,
        model_cls: type[T],
        output_dir: str,
        include_metadata: bool = True,
    ) -> str:
        """
        Export all objects of a model type as a knowledge base.
        Returns the path to the exported file.

        Parameters:
        - model_cls: The model class to export
        - output_dir: Directory to save the export file
        - include_metadata: Whether to include model metadata in export
        """
        if not hasattr(model_cls, "__redis_indices__"):
            raise ValueError("Model class must be decorated with @RedisIndex")

        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Get all object IDs
            all_ids = await self.get_all_item_ids(model_cls)

            # Get all objects with their versions
            all_objects = []
            if all_ids:
                async for batch in self.find_batches(model_cls, None, batch_size=1000):
                    all_objects.extend(batch)

            # Prepare export data
            export_data = {
                "model": model_cls.__name__,
                "export_timestamp": datetime.now().isoformat(),
                "objects": [
                    {
                        "data": obj.model_dump(),
                        "version": version,
                    }
                    for obj, version in all_objects
                ],
            }

            # Include metadata if requested
            if include_metadata:
                export_data["metadata"] = {
                    "indices": [
                        {
                            "field_path": idx.field_path,
                            "index_type": idx.index_type.value,
                            "prefix": idx.prefix,
                            "exclude_nested": idx.exclude_nested,
                            "ttl": idx.ttl,
                            "is_immutable": idx.is_immutable,
                        }
                        for idx in model_cls.__redis_indices__
                    ],
                    "ttl": getattr(model_cls, "__redis_ttl__", None),
                }

            # Save to file
            output_file = os.path.join(
                output_dir,
                f"knowledge_base_{model_cls.__name__.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            )

            async with aiofiles.open(output_file, "w") as f:
                await f.write(json.dumps(export_data, indent=2))

            return output_file

        except Exception as e:
            logger.error(f"Failed to export knowledge base for {model_cls.__name__}: {e}")
            raise

    @track_operation("import_knowledge_base")
    async def import_knowledge_base(
        self,
        model_cls: type[T],
        input_file: str,
        merge_strategy: Literal["skip_existing", "overwrite", "merge"] = "skip_existing",
        validate_metadata: bool = True,
    ) -> int:
        """
        Import objects from a knowledge base file.
        Returns number of imported objects.

        Parameters:
        - model_cls: The model class to import into
        - input_file: Path to the knowledge base file
        - merge_strategy: How to handle existing objects:
            - skip_existing: Skip if object exists
            - overwrite: Always overwrite existing objects
            - merge: Update existing objects if version matches
        - validate_metadata: Whether to validate model metadata matches
        """
        if not hasattr(model_cls, "__redis_indices__"):
            raise ValueError("Model class must be decorated with @RedisIndex")

        try:
            # Read and parse knowledge base file
            async with aiofiles.open(input_file, "r") as f:
                content = await f.read()
                data = json.loads(content)

            # Validate model type if metadata included
            if validate_metadata and "metadata" in data:
                # Check indices match
                current_indices = {
                    ".".join(idx.field_path): idx
                    for idx in model_cls.__redis_indices__
                }
                for idx_data in data["metadata"]["indices"]:
                    field_path = ".".join(idx_data["field_path"])
                    if field_path not in current_indices:
                        raise ValueError(
                            f"Model {model_cls.__name__} is missing index for field {field_path}"
                        )
                    current_idx = current_indices[field_path]
                    if current_idx.index_type.value != idx_data["index_type"]:
                        raise ValueError(
                            f"Index type mismatch for field {field_path}: "
                            f"expected {current_idx.index_type.value}, got {idx_data['index_type']}"
                        )

            # Import objects
            imported_count = 0
            for obj_data in data["objects"]:
                try:
                    # Convert to model object
                    obj = model_cls.model_validate(obj_data["data"])
                    version = obj_data.get("version")

                    # Handle based on merge strategy
                    if merge_strategy == "skip_existing":
                        # Check if exists first
                        existing, _ = await self.get(model_cls, obj.id)
                        if existing:
                            continue
                        success, _ = await self.save(model_cls, obj.id, obj)
                        if success:
                            imported_count += 1

                    elif merge_strategy == "overwrite":
                        # Save without version check
                        success, _ = await self.save(model_cls, obj.id, obj)
                        if success:
                            imported_count += 1

                    elif merge_strategy == "merge":
                        # Save with version check
                        success, _ = await self.save(
                            model_cls, obj.id, obj,
                            version_token=version
                        )
                        if success:
                            imported_count += 1

                    else:
                        raise ValueError(f"Invalid merge strategy: {merge_strategy}")

                except Exception as e:
                    logger.error(
                        f"Failed to import object {obj_data.get('data', {}).get('id')}: {e}"
                    )
                    continue

            return imported_count

        except Exception as e:
            logger.error(f"Failed to import knowledge base for {model_cls.__name__}: {e}")
            raise

    @track_operation("update_indexed_property")
    async def update_indexed_property(
        self,
        model_cls: type[BaseModel],
        field_path: str | list[str],
        new_value: Any,
        new_value_members: list[str],
        value_setter: Callable[[BaseModel], None],
        value_resetter: Callable[[BaseModel], None],
    ) -> list[tuple[bool, bool, str | None]]:
        """
        Efficiently update an indexed property in batch with optimistic concurrency.
        Updates both:
        - Items that should have the new value (set value)
        - Items that previously had the value but shouldn't anymore (reset value)

        Args:
            model_cls: The model class with Redis indices
            field_path: Path to the indexed field as string
            new_value: Value to set for the indexed field
            new_value_members: List of item IDs that should have the new value
            value_setter: Function to set the field value
            value_resetter: Function to reset the field value

        Returns:
            List of (exists, updated, latest_version) tuples for each updated item.
            - exists: Whether the item existed
            - updated: Whether the update was successful
            - latest_version: The new version token if updated, or current version if not updated

        Example:
            ```python
            # Update node types in batch
            results = await redis_om.update_indexed_property(
                model_cls=Node,
                field_path="node_type",
                new_value="function",
                new_value_members=function_node_ids,
                value_setter=lambda node: setattr(node, "node_type", "function"),
                value_resetter=lambda node: setattr(node, "node_type", "unknown"),
            )
            ```
        """
        # Convert field path to list if string
        if isinstance(field_path, str):
            field_path = field_path.split(".")

        # Find the index for this field
        index = self._get_index(model_cls, field_path)
        if index.index_type != IndexType.SET:
            raise ValueError(
                f"Field {'.'.join(field_path)} must be indexed as SET, got {index.index_type}"
            )

        # Get IDs of items currently having this value
        set_key = self._build_index_set_key(model_cls, index, new_value)
        async with self.redis_client.get_redis_connection() as redis:
            old_set_members = {
                id.decode()
                for id in await redis.smembers(set_key)
            }

        # Items to update: union of current and desired classifications
        items_to_update = old_set_members | set(new_value_members)
        if not items_to_update:
            return []

        # Fetch all items and versions in one batch
        items, version_tokens = await self.checkout_items(model_cls, list(items_to_update))
        if not items:
            return []

        # Update values in memory
        for item in items:
            if str(item.id) not in new_value_members:
                # Reset value for previously classified items
                value_resetter(item)
            else:
                # Set value for newly classified items
                value_setter(item)

        # Batch checkin all updated items
        return await self.checkin_items(
            items=items,
            version_tokens=version_tokens,
            ids=[str(item.id) for item in items],
            model_cls=model_cls,
            all_or_nothing=True,  # Either all updates succeed or none do
        )

    @track_operation("sync_set_index")
    async def sync_set_index(
        self,
        model_cls: type[BaseModel],
        field_path: str | list[str],
        desired_state: dict[str, Any],
        *,
        on_add: Callable[[BaseModel, Any], None] | None = None,
        on_remove: Callable[[BaseModel, Any], None] | None = None,
        on_update: Callable[[BaseModel, Any, Any], None] | None = None,
        batch_size: int = 1000,  # Control batch size for large syncs
        all_or_nothing: bool = True,  # Transaction semantics
    ) -> SyncResult:
        """
        Synchronize a SET index to match a desired state with optimistic concurrency.

        This is a high-level operation that reconciles the current state of a SET index
        with a desired state, similar to a database MERGE operation or Kubernetes
        reconciliation loop.

        - Any item that is in the `desired_state` but not in the current state will be added.
        - Any item that is in the current state but not in the `desired_state` will be removed.
        - Any item that is in the current state and in the `desired_state` but with a different
        value will be updated.

        Key improvements in this design:
        1. Familiar Pattern: This matches familiar database operations like MERGE/UPSERT and
           Kubernetes-style reconciliation.
        2. `Explicit State`: Instead of separate value/members, it uses a single `desired_state`
           mapping that clearly shows the target state.
        3. `Flexible Callbacks`: The callbacks allow customizing behavior for different operations
           while keeping the core synchronization logic reusable.
        4. `Result Object`: Returns a structured result object with detailed information about
           what changed.
        5. `Bidirectional`: Can handle cases where items move between different sets, not just
           in/out of a single set.

        This could be implemented on top of the existing `update_indexed_property`,
        but provides a more intuitive interface for common use cases.

        Args:
            `model_cls`: The model class with Redis indices
            `field_path`: Path to the indexed field (string or list)
            `desired_state`: Mapping of item IDs to their desired values
            `on_add`: Callback(item, new_value) when item added to set
            `on_remove`: Callback(item, old_value) when item removed from set
            `on_update`: Callback(item, old_value, new_value) when value changes
            `batch_size`: Number of items to process in each batch
            `all_or_nothing`: If True, entire sync fails if any item fails

        Returns:
            `SyncResult` with detailed sync statistics and results

        Example:
            ```python
            # Sync GPU inference groups
            result = await redis_om.sync_set_index(
                model_cls=InferenceGroupState,
                field_path="deployment_node",
                desired_state={
                    "group_state_1_id": "node_1",
                    "group_state_2_id": "node_2",
                    "group_state_3_id": "node_3"
                },
                on_add=lambda group, node_id: setattr(group, "deployment_node", node_id),
                on_remove=lambda group, _: setattr(group, "deployment_node", None)
            )
            ```

        Raises:
            ValueError: If field is not indexed as SET
            ConcurrencyError: If optimistic locking fails
            BatchError: If all_or_nothing=True and any item fails
        """
        # Normalize field path
        if isinstance(field_path, str):
            field_path = field_path.split(".")

        # Validate index type
        index = self._get_index(model_cls, field_path)
        if index.index_type != IndexType.SET:
            raise ValueError(
                f"Field {'.'.join(field_path)} must be indexed as SET, got {index.index_type}"
            )

        # Track metrics
        start_time = time.time()
        sync_stats = SyncStats()

        try:
            # Get current state efficiently
            current_state: dict[str, Any] = {}
            async with self.redis_client.get_redis_connection() as redis:
                # Scan all sets for this index
                pattern = self._build_index_all_sets_pattern(model_cls, index)
                async for key in redis.scan_iter(pattern):
                    value = key.decode().split(":")[-1]  # Extract value from key
                    members = await redis.smembers(key)
                    for member in members:
                        current_state[member.decode()] = value

            # Calculate diffs
            to_add: dict[str, Any] = {}    # id -> new_value
            to_remove: dict[str, Any] = {} # id -> old_value
            to_update: dict[str, tuple[Any, Any]] = {} # id -> (old_value, new_value)

            # Find items to remove (in current but not in desired)
            for id, old_value in current_state.items():
                if id not in desired_state:
                    to_remove[id] = old_value
                elif desired_state[id] != old_value:
                    to_update[id] = (old_value, desired_state[id])

            # Find items to add (in desired but not in current)
            for id, new_value in desired_state.items():
                if id not in current_state:
                    to_add[id] = new_value

            # Process changes in batches
            results = []
            all_changes = {
                id: ("add", value) for id, value in to_add.items()
            } | {
                id: ("remove", value) for id, value in to_remove.items()
            } | {
                id: ("update", values) for id, values in to_update.items()
            }

            # Process in batches
            for i in range(0, len(all_changes), batch_size):
                batch_changes = dict(list(all_changes.items())[i:i + batch_size])
                batch_ids = list(batch_changes.keys())

                # Checkout items
                items, version_tokens = await self.checkout_items(model_cls, batch_ids)
                if not items:
                    continue

                # Apply changes with callbacks
                for item in items:
                    id = str(item.id)
                    change_type, values = batch_changes[id]

                    if change_type == "add":
                        if on_add:
                            on_add(item, values)
                        sync_stats.added += 1
                    elif change_type == "remove":
                        if on_remove:
                            on_remove(item, values)
                        sync_stats.removed += 1
                    else:  # update
                        old_value, new_value = values
                        if on_update:
                            on_update(item, old_value, new_value)
                        sync_stats.updated += 1

                # Checkin batch
                batch_results = await self.checkin_items(
                    items=items,
                    version_tokens=version_tokens,
                    ids=batch_ids,
                    model_cls=model_cls,
                    all_or_nothing=all_or_nothing
                )
                results.extend(batch_results)

                # Handle failures
                if all_or_nothing and not all(r[1] for r in batch_results):
                    raise BatchError("Some items failed to update")

                sync_stats.processed += len(batch_results)
                sync_stats.successful += sum(1 for r in batch_results if r[1])
                sync_stats.failed += sum(1 for r in batch_results if not r[1])

            return SyncResult(
                stats=sync_stats,
                duration=time.time() - start_time,
                results=results
            )

        except Exception as e:
            logger.error(f"Sync failed for {model_cls.__name__}.{'.'.join(field_path)}: {e}")
            raise

    @track_operation("lock_item")
    async def lock_item(
        self,
        model_cls: type[BaseModel],
        id: str,
        *,
        owner_id: str | None = None,
        timeout: int = 30,
    ) -> tuple[bool, str]:
        """
        Acquire a lock on an item with timeout.

        Args:
            model_cls: The model class
            id: Item ID to lock
            owner_id: Optional owner ID (generated if not provided)
            timeout: Lock timeout in seconds

        Returns:
            Tuple of (success, owner_id)
            If successful, returns (True, owner_id)
            If failed, returns (False, current_owner_id)
        """
        owner_id = owner_id or str(uuid.uuid4())
        lock_key = self._build_lock_key(model_cls, id)

        # Try to acquire lock
        acquired = await self._execute_with_semaphore(
            lambda redis: redis.set(
                lock_key,
                owner_id,
                nx=True,  # Only set if not exists
                ex=timeout,  # Set expiry
            )
        )

        if acquired:
            return True, owner_id

        # Get current owner if lock exists
        current_owner = await self._execute_with_semaphore(
            lambda redis: redis.get(lock_key)
        )
        return False, current_owner.decode() if current_owner else ""

    @track_operation("unlock_item")
    async def unlock_item(
        self,
        model_cls: type[BaseModel],
        id: str,
        owner_id: str,
    ) -> bool:
        """
        Release a lock if we own it.

        Args:
            model_cls: The model class
            id: Item ID to unlock
            owner_id: Must match the ID used to acquire the lock

        Returns:
            True if lock was released, False if we don't own it
        """
        lock_key = self._build_lock_key(model_cls, id)

        async with self._get_pipeline() as pipe:
            # Check ownership and delete atomically
            await pipe.watch(lock_key)

            current_owner = await self._execute_with_semaphore(
                lambda redis: redis.get(lock_key)
            )

            if not current_owner or current_owner.decode() != owner_id:
                await pipe.unwatch()
                return False

            pipe.multi()
            await pipe.delete(lock_key)
            await pipe.execute()
            return True

    @track_operation("renew_lock")
    async def renew_lock(
        self,
        model_cls: type[BaseModel],
        id: str,
        owner_id: str,
        timeout: int = 30,
    ) -> bool:
        """
        Renew a lock if we own it.

        Args:
            model_cls: The model class
            id: Item ID to renew lock for
            owner_id: Must match the ID used to acquire the lock
            timeout: New timeout in seconds

        Returns:
            True if lock was renewed, False if we don't own it
        """
        lock_key = self._build_lock_key(model_cls, id)

        async with self._get_pipeline() as pipe:
            await pipe.watch(lock_key)

            current_owner = await self._execute_with_semaphore(
                lambda redis: redis.get(lock_key)
            )

            if not current_owner or current_owner.decode() != owner_id:
                await pipe.unwatch()
                return False

            pipe.multi()
            await pipe.expire(lock_key, timeout)
            await pipe.execute()
            return True

    @track_operation("get_lock_info")
    async def get_lock_info(
        self,
        model_cls: type[BaseModel],
        id: str,
    ) -> tuple[bool, str | None, float | None]:
        """
        Get information about an item's lock without acquiring it.

        Args:
            model_cls: The model class
            id: Item ID to check

        Returns:
            Tuple of (is_locked, owner_id, remaining_ttl)
            If not locked, returns (False, None, None)
        """
        lock_key = self._build_lock_key(model_cls, id)

        async with self._get_pipeline() as pipe:
            await pipe.get(lock_key)
            await pipe.ttl(lock_key)
            owner, ttl = await pipe.execute()

            if not owner or ttl <= 0:
                return False, None, None

            return True, owner.decode(), ttl

    @asynccontextmanager
    async def item_lock(
        self,
        model_cls: type[BaseModel],
        id: str,
        *,
        owner_id: str | None = None,
        timeout: int = 30,
        raise_on_fail: bool = True,
        auto_renew: bool = False,
        auto_update: bool = False,
    ) -> AsyncGenerator[tuple[bool, str, BaseModel | None], None]:
        """
        Context manager for locking an item.

        Args:
            `model_cls`: The model class
            `id`: Item ID to lock
            `owner_id`: Optional owner ID (generated if not provided)
            `timeout`: Lock timeout in seconds
            `raise_on_fail`: If True, raises if lock can't be acquired
            `auto_renew`: If True, auto-renews lock at timeout/2 intervals
            `auto_update`: If True, auto-updates item with lock before exiting the context manager

        Yields:
            Tuple of (`success`, `owner_id`, `item`)

        Raises:
            LockAcquisitionError: If raise_on_fail=True and lock can't be acquired
        """
        renew_task = None
        success, owner = await self.lock_item(
            model_cls, id, owner_id=owner_id, timeout=timeout
        )

        if not success and raise_on_fail:
            raise LockAcquisitionError(f"Failed to acquire lock for {id}, owned by {owner}")

        try:
            if success and auto_renew:
                # Start renewal task
                async def renew_loop():
                    while True:
                        await asyncio.sleep(timeout / 2)
                        await self.renew_lock(model_cls, id, owner, timeout)

                renew_task = asyncio.create_task(renew_loop())

            item = None
            if success:
                item = await self.get(model_cls, id)
                if not item:
                    raise KeyError(f"No item found for item_id={id}")

            yield success, owner, item

            # Auto-update item if requested and we have the lock
            if auto_update and success and item:
                result = await self.save(
                    item,
                    id,
                    model_cls=model_cls,
                    update_if_exists=True,
                    version_token=None,
                    ttl=None,
                )
                if not result[0]:
                    raise KeyError(f"Failed to update item {id}")

        finally:
            if renew_task:
                renew_task.cancel()
                try:
                    await renew_task
                except asyncio.CancelledError:
                    pass

            if success:
                await self.unlock_item(model_cls, id, owner)

    @track_operation("save_with_lock")
    async def save_with_lock(
        self,
        obj: BaseModel,
        id: str,
        owner_id: str | None = None,
        *,
        update_if_exists: bool = True,
        version_token: str | None = None,
        ttl: int | None = None,
        model_cls: type[BaseModel] | None = None,
    ) -> tuple[bool, str]:
        """
        Save an item while holding its lock.

        This combines locking with optimistic concurrency:
        - Must own the lock (checked first)
        - Version token must match if provided
        - All operations are atomic

        Args:
            obj: Object to save
            id: Item ID
            owner_id: Must match current lock owner
            update_if_exists: Whether to update existing items
            version_token: Optional version token for optimistic concurrency
            ttl: Optional TTL override
            model_cls: Optional model class for index definitions

        Returns:
            Tuple of (success, new_version)
        """
        model_cls = model_cls or obj.__class__
        lock_key = self._build_lock_key(model_cls, id)

        async with self._get_pipeline() as pipe:
            # Verify lock ownership
            await pipe.watch(lock_key)
            current_owner = await self._execute_with_semaphore(
                lambda redis: redis.get(lock_key)
            )

            if not current_owner or current_owner.decode() != owner_id:
                await pipe.unwatch()
                return False, ""

            # Proceed with normal save operation
            return await self.save(
                obj,
                id,
                update_if_exists=update_if_exists,
                version_token=version_token,
                ttl=ttl,
                model_cls=model_cls,
            )

    @track_operation("get_sorted_set_size")
    async def get_sorted_set_size(
        self,
        model_cls: type[BaseModel],
        field_path: str | list[str],
    ) -> int:
        """Get the size of the sorted set (queue)."""
        if isinstance(field_path, str):
            field_path = field_path.split(".")

        # Validate index type
        index = self._get_index(model_cls, field_path)
        if index.index_type != IndexType.SORTED_SET:
            raise ValueError(
                f"Field {'.'.join(field_path)} must be indexed as SORTED_SET, got {index.index_type}"
            )

        idx_key = self._build_index_key(model_cls, index)
        try:
            return await self._execute_with_semaphore(
                lambda redis: redis.zcard(idx_key)
            )
        except Exception as e:
            logger.error(f"Failed to get size of sorted set for field {model_cls.__name__}.{'.'.join(field_path)}: {e}")
            raise

    @track_operation("get_sorted_set_items")
    async def get_sorted_set_items(
        self,
        model_cls: type[BaseModel],
        field_path: str | list[str],
        withscores: bool = False,
    ) -> list[BaseModel] | list[tuple[BaseModel, float]]:
        """Get all items in the sorted set (queue)."""
        if isinstance(field_path, str):
            field_path = field_path.split(".")

        # Validate index type
        index = self._get_index(model_cls, field_path)
        if index.index_type != IndexType.SORTED_SET:
            raise ValueError(
                f"Field {'.'.join(field_path)} must be indexed as SORTED_SET, got {index.index_type}"
            )

        idx_key = self._build_index_key(model_cls, index)
        try:
            return await self._execute_with_semaphore(
                lambda redis: redis.zrange(idx_key, 0, -1, withscores=withscores)
            )
        except Exception as e:
            logger.error(f"Failed to get items from sorted set for field {model_cls.__name__}.{'.'.join(field_path)}: {e}")
            raise

    @track_operation("get_sorted_set_top_item_id")
    async def get_sorted_set_top_item_id(
        self,
        model_cls: type[BaseModel],
        field_path: str | list[str],
        query: QueryExpr | None = None,
        use_query_planner: bool = False,
    ) -> BaseModel | None:
        """Get the highest-ranked item ID to process."""
        if isinstance(field_path, str):
            field_path = field_path.split(".")

        # Validate index type
        index = self._get_index(model_cls, field_path)
        if index.index_type != IndexType.SORTED_SET:
            raise ValueError(
                f"Field {'.'.join(field_path)} must be indexed as SORTED_SET, got {index.index_type}"
            )

        idx_key = self._build_index_key(model_cls, index)
        try:
            item_id_score_pairs = await self._execute_with_semaphore(
                lambda redis: redis.zrange(
                    idx_key, 0, 0, withscores=True
                )
            )
            if not item_id_score_pairs:
                return None

            if query:
                # TODO: Pass the item_id_score_pairs to the query planner to optimize query execution
                # over the sorted set only instead of the entire dataset
                async with self.redis_client.get_pipeline() as (_, redis):
                    ids = await self._find_ids(model_cls, query, redis, use_planner=use_query_planner)
                    if ids:
                        item_id_score_pairs = [
                            (id, score) for id, score in item_id_score_pairs if id in ids
                        ]
                        item_id_score_pairs = sorted(item_id_score_pairs, key=lambda x: x[1], reverse=True)

            item_id, _ = item_id_score_pairs[0]
            return item_id
        except Exception as e:
            logger.error(f"Failed to get top item from sorted set for field {model_cls.__name__}.{'.'.join(field_path)}: {e}")
            raise

    @track_operation("is_sorted_set_empty")
    async def is_sorted_set_empty(
        self,
        model_cls: type[BaseModel],
        field_path: str | list[str],
    ) -> bool:
        """Check if the sorted set (queue) is empty."""
        if isinstance(field_path, str):
            field_path = field_path.split(".")

        # Validate index type
        index = self._get_index(model_cls, field_path)
        if index.index_type != IndexType.SORTED_SET:
            raise ValueError(
                f"Field {'.'.join(field_path)} must be indexed as SORTED_SET, got {index.index_type}"
            )

        idx_key = self._build_index_key(model_cls, index)
        try:
            return await self._execute_with_semaphore(
                lambda redis: redis.zrange(idx_key, 0, 0) == []
            )
        except Exception as e:
            logger.error(f"Failed to check if sorted set is empty for field {model_cls.__name__}.{'.'.join(field_path)}: {e}")
            raise

    async def append_to_time_series(
        self,
        ts_name: str,
        value: Any,
        timestamp: datetime | None = None,
        ttl: int = timedelta(days=7).total_seconds(),
    ) -> None:
        """Store a value in a time series.

        Args:
            ts_name: Name of the time series to append to
            value: Value to append
            timestamp: Optional timestamp to use instead of the current time
            ttl: Optional TTL override for the item in seconds
        """
        try:
            item = TimeSeriesItem(
                name=ts_name,
                value=value,
                timestamp=timestamp or datetime.now(timezone.utc),
            )
            await self.save(item, id=item.id, ttl=ttl)
        except Exception as e:
            logger.error(f"Error storing time series item: {e}")

    async def get_latest_of_time_series(self, ts_name: str) -> Any | None:
        """Get the most recent item in a time series."""
        try:
            db = TimeSeriesItem.db
            item_id = await self.get_sorted_set_top_item_id(
                model_cls=TimeSeriesItem,
                field_path="timestamp",
                query=db.name == ts_name,
            )
            item = await self.get(TimeSeriesItem, item_id)
            if item:
                return item.value
            return None
        except Exception as e:
            logger.error(f"Error getting latest of time series {ts_name}: {e}")
            return None

    async def get_bounded_list(self, list_name: str) -> list[Any]:
        """Get a bounded list of values.

        Returns:
            List of values
        """
        try:
            db = BoundedList.db
            blists = await self.find(
                model_cls=BoundedList,
                query=db.name == list_name,
            )
            if not blists:
                return []

            blist, _ = blists[0]

            return blist.values
        except Exception as e:
            logger.error(f"Error getting bounded list {list_name}: {e!s}")
            return []

    async def append_to_bounded_list(
        self, list_name: str, item: Any, max_history: int = 100, ttl: int | None = None
    ) -> None:
        """Append new item to a bounded list.

        Args:
            list_name: Name of the list to append to
            item: Item to append
        """
        try:
            db = BoundedList.db
            blists = await self.find(
                model_cls=BoundedList,
                query=db.name == list_name,
            )
            if not blists:
                blist = BoundedList(
                    name=list_name,
                    values=[],
                )
            else:
                blist, _ = blists[0]

            blist.values.append(item)

            # Keep only recent history
            if len(blist.values) > max_history:
                blist.values = blist.values[-max_history:]

            # Store with TTL
            await self.save(blist, id=blist.id, ttl=ttl)
        except Exception as e:
            logger.error(f"Error appending bounded list {list_name}: {e!s}")

    async def initialize_topics(
        self,
        topics: dict[str, Any],
    ) -> bool:
        """Initialize topics with TTL."""
        try:
            async with self._get_pipeline() as pipe:
                for key, value in topics.items():
                    if value is None:
                        continue
                    await self.update_state_topic(
                        key,
                        value,
                        replace_all=True,
                        pipe=pipe,
                        update_type="initialization",
                    )

                await pipe.execute()
            return True
        except Exception as e:
            logger.error(f"Failed to initialize job {self.namespace}: {e!s}")
            return False

    def subscribe_to_state_updates(self, topic_name: str) -> DistributedStateSubscriber:
        # TODO: Add a check to ensure that the topic is valid. For example, store a list
        # of subscribed topics and published topics and ensure they are tracked.
        # These topic list must be stored in Redis and must be updated atomically
        # because the subscriber and publisher may be on different nodes.
        channel = self._build_state_update_channel(topic_name)
        return DistributedStateSubscriber(self.redis_client, channel)

    async def update_state_topic(
        self,
        topic: str,
        updates: dict[str, Any],
        *,
        pipe: Pipeline | None = None,
        replace_all: bool = False,
        update_type: Literal["update", "initialization"] = "update",
    ) -> bool | None:
        """
        Atomic update of a state topic.

        Args:
            updates: dict of updated fields
            replace_all: If True, replace entire topic. If False, update only specified fields.
        """

        async def _update_state_topic(pipe: Pipeline) -> None:
            topic_key = self._build_state_topic_key(topic)
            channel_key = self._build_state_update_channel(topic)
            if replace_all:
                # Delete existing hash and set new one
                # Otherwise, we update only specified fields
                await pipe.delete(topic_key)

            await pipe.hset(topic_key, mapping=updates)
            await pipe.expire(topic_key, self.ttl)  # Refresh TTL

            # Publish update to be picked up by the InferenceJobTrackers
            await pipe.publish(
                channel_key,
                json.dumps(
                    DistributedStateUpdate(
                        topic=topic,
                        timestamp=time.time(),
                        type=update_type,
                        replace_all=replace_all,
                        data=updates,
                    ).model_dump()
                ),
            )

        if pipe is not None:
            await _update_state_topic(pipe)
            return

        try:
            async with self._get_pipeline() as pipe:
                await _update_state_topic(pipe)
                await pipe.execute()
            return True
        except Exception as e:
            logger.error(
                f"Failed to {'replace' if replace_all else 'update'} "
                f"state topic {topic}: {e}"
            )
            return False

    async def get_state_topic(self, topic: str) -> dict[str, Any]:
        """Get all current state topic data."""
        try:
            topic_key = self._build_state_topic_key(topic)
            data = await self._execute_with_semaphore(
                lambda redis: redis.hgetall(topic_key)
            )
            return {k.decode(): json.loads(v.decode()) for k, v in data.items()}
        except Exception as e:
            logger.error(f"Failed to get state topic {topic}: {e}")
            return {}



