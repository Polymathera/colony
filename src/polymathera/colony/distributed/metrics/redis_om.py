from .common import BaseMetricsMonitor


class RedisOMMetricsMonitor(BaseMetricsMonitor):
    """Prometheus metrics for Redis OM."""

    def __init__(self,
                 enable_http_server: bool = True,
                 service_name: str = "redis_om_metrics"):
        super().__init__(enable_http_server, service_name)

        self.logger.debug(f"Initializing RedisOMMetricsMonitor instance {id(self)}...")

        # Operation counters
        self.REDIS_OP_COUNT = self.get_or_create_counter(
            'redis_operation_total',
            'Number of Redis operations',
            ['operation', 'status', 'namespace']
        )

        # Operation latency histograms
        self.REDIS_OP_DURATION = self.get_or_create_histogram(
            'redis_operation_duration_seconds',
            'Duration of Redis operations',
            ['operation', 'namespace'],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0)
        )

        # Index metrics
        self.REDIS_INDEX_SIZE = self.get_or_create_gauge(
            'redis_index_size',
            'Number of entries in Redis index',
            ['model', 'field', 'index_type', 'namespace']
        )

        # Query metrics
        self.REDIS_QUERY_COUNT = self.get_or_create_counter(
            'redis_query_total',
            'Number of Redis queries',
            ['model', 'operation', 'index_type', 'namespace']
        )

        # Connection pool metrics
        self.REDIS_POOL_SIZE = self.get_or_create_gauge(
            'redis_pool_size',
            'Size of Redis connection pool',
            ['namespace']
        )

        self.REDIS_ACTIVE_CONNECTIONS = self.get_or_create_gauge(
            'redis_active_connections',
            'Number of active Redis connections',
            ['namespace']
        )

        # Health check metrics
        self.REDIS_HEALTH_CHECK = self.get_or_create_counter(
            'redis_health_check_total',
            'Number of Redis health checks',
            ['status', 'namespace']
        )

        self.REDIS_MEMORY_USAGE = self.get_or_create_gauge(
            'redis_memory_bytes',
            'Redis memory usage in bytes',
            ['namespace']
        )

        self.REDIS_CONNECTED = self.get_or_create_gauge(
            'redis_connected',
            'Whether Redis is connected',
            ['namespace']
        )

        # Circuit breaker metrics
        self.REDIS_CIRCUIT_STATE = self.get_or_create_gauge(
            'redis_circuit_breaker_state',
            'Circuit breaker state (0=open, 1=half-open, 2=closed)',
            ['namespace']
        )

        self.REDIS_CIRCUIT_FAILURES = self.get_or_create_counter(
            'redis_circuit_breaker_failures_total',
            'Number of circuit breaker failures',
            ['namespace']
        )

        self.REDIS_CIRCUIT_SUCCESSES = self.get_or_create_counter(
            'redis_circuit_breaker_successes_total',
            'Number of circuit breaker successes',
            ['namespace']
        )



from opentelemetry import metrics, trace
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)


# Metrics
QUERY_PLANNING_DURATION = meter.create_histogram(
    name="redis_om_query_planning_duration",
    description="Time spent planning queries",
    unit="seconds",
)
QUERY_EXECUTION_DURATION = meter.create_histogram(
    name="redis_om_query_execution_duration",
    description="Time spent executing queries",
    unit="seconds",
)
QUERY_CACHE_HITS = meter.create_counter(
    name="redis_om_query_cache_hits",
    description="Number of query cache hits",
)
QUERY_CACHE_MISSES = meter.create_counter(
    name="redis_om_query_cache_misses",
    description="Number of query cache misses",
)
QUERY_OPTIMIZATION_SAVINGS = meter.create_histogram(
    name="redis_om_query_optimization_savings",
    description="Estimated cost savings from query optimization",
    unit="operations",
)
