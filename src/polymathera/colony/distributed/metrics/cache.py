from prometheus_client import Counter, Gauge, Histogram


CACHE_HITS = Counter(
    "cache_hits_total",
    "Number of cache hits",
    labelnames=["namespace", "operation", "extras"],
)
CACHE_MISSES = Counter(
    "cache_misses_total",
    "Number of cache misses",
    labelnames=["namespace", "operation", "extras"],
)
CACHE_SIZE = Gauge(
    "cache_size_bytes",
    "Current cache size in bytes",
    labelnames=["namespace", "operation", "extras"],
)
CACHE_ITEM_SERIALIZATION_TIME = Histogram(
    "cache_item_serialization_seconds",
    "Time spent serializing items",
    labelnames=["namespace", "operation", "extras"],
)
CACHE_ITEM_SIZE = Histogram(
    "cache_item_size_bytes",
    "Size of serialized items",
    labelnames=["namespace", "operation", "extras"],
    buckets=[1000, 10000, 100000, 1000000],
)
CACHE_OPERATIONS = Counter(
    "cache_operations_total",
    "Number of cache operations",
    labelnames=["namespace", "operation", "status", "extras"],
)
CACHE_LATENCY = Histogram(
    "cache_latency_seconds",
    "Cache operation duration",
    labelnames=["namespace", "operation", "extras"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1],
)
CACHE_ERRORS = Counter(
    "cache_errors_total",
    "Number of cache errors",
    labelnames=["namespace", "operation", "extras"],
)
CACHE_MEMORY = Gauge(
    "cache_memory_bytes",
    "Estimated memory usage",
    labelnames=["namespace", "operation", "extras"],
)
CACHE_EVICTIONS = Counter(
    "cache_evictions_total",
    "Number of cache evictions",
    labelnames=["namespace", "operation", "extras"],
)
CACHE_ENTRY_AGE = Histogram(
    "cache_entry_age_seconds",
    "Age of cache entries",
    labelnames=["namespace", "operation", "extras"],
)
CACHE_BATCH_SIZE = Histogram(
    "cache_batch_size",
    "Batch operation sizes",
    labelnames=["namespace", "operation", "extras"],
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
)
CACHE_COMPRESSION_RATIO = Histogram(
    "cache_compression_ratio",
    "Compression ratios achieved",
    labelnames=["namespace", "operation", "extras"],
    buckets=[1.5, 2, 3, 4, 5, 7, 10],
)
CACHE_HEALTH = Gauge(
    "cache_health",
    "Health status of cache (1 = healthy, 0 = unhealthy)",
    labelnames=["namespace", "extras"],
)
