from prometheus_client import Counter, Histogram, Gauge
from opentelemetry import metrics, trace

tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

# Prometheus Metrics

# Operation counters
BLACKBOARD_OP_COUNT = Counter(
    'blackboard_operation_total',
    'Number of blackboard operations',
    ['operation', 'status', 'vmr_id']
)

# Operation latency histograms
BLACKBOARD_OP_DURATION = Histogram(
    'blackboard_operation_duration_seconds',
    'Duration of blackboard operations',
    ['operation', 'vmr_id'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0)
)

# Query metrics
BLACKBOARD_QUERY_COUNT = Counter(
    'blackboard_query_total',
    'Number of blackboard queries',
    ['vmr_id', 'query_type', 'status']
)

# Query size metrics
BLACKBOARD_QUERY_SIZE = Gauge(
    'blackboard_query_size',
    'Number of queries in blackboard',
    ['vmr_id', 'query_type']  # query_type can be 'open', 'answered', 'external', etc.
)

# OpenTelemetry Metrics

# Operation metrics
BLACKBOARD_OPERATIONS = meter.create_counter(
    name="blackboard_operations",
    description="Number of blackboard operations",
    unit="1",
)

# Query latency metrics
BLACKBOARD_QUERY_LATENCY = meter.create_histogram(
    name="blackboard_query_latency",
    description="Latency of blackboard query operations",
    unit="ms",
)

# Query batch metrics
BLACKBOARD_BATCH_SIZE = meter.create_histogram(
    name="blackboard_batch_size",
    description="Size of query batches",
    unit="1",
)

# Query state metrics
BLACKBOARD_QUERY_STATES = meter.create_counter(
    name="blackboard_query_states",
    description="Number of queries in different states",
    unit="1",
)

# Knowledge base metrics
BLACKBOARD_KB_SIZE = meter.create_counter(
    name="blackboard_kb_size",
    description="Size of knowledge base operations",
    unit="bytes",
)
