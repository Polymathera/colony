import time
import opentelemetry.metrics as metrics
from contextlib import asynccontextmanager
from functools import wraps
import os
import socket
import threading
import logging
from typing import Iterable
from prometheus_client import start_http_server, Histogram, Counter, Gauge, REGISTRY


# Global state for node-wide metrics server
_prometheus_metrics_server_started = False
_prometheus_metrics_server_lock = threading.Lock()

logger = logging.getLogger(__name__)

def get_metrics_port() -> int | None:
    port_str = os.environ.get('PROMETHEUS_PORT')
    if not port_str:
        logger.error("PROMETHEUS_PORT environment variable not set")
        return None
    try:
        port = int(port_str)
        return port
    except ValueError:
        logger.error(f"Invalid PROMETHEUS_PORT value: {port_str}")
        return None

def ensure_node_metrics_server() -> bool:
    """
    Ensure a metrics HTTP server is running on this node.

    This function is thread-safe and will only start one server per process.
    It uses a fixed port from the PROMETHEUS_PORT environment variable.

    If a persistent node metrics service is running, this will detect it
    and avoid starting a duplicate server.

    Returns True if server is running (either started now or was already running).
    """
    global _prometheus_metrics_server_started

    with _prometheus_metrics_server_lock:
        if _prometheus_metrics_server_started:
            return True

        # Use fixed port from environment variable
        port = get_metrics_port()
        if not port:
            return False

        # First, check if something is already running on the port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            result = sock.connect_ex(('localhost', port))
            sock.close()

            if result == 0:
                # Port is in use, check if it's serving metrics
                try:
                    import urllib.request
                    import urllib.error
                    response = urllib.request.urlopen(f'http://localhost:{port}/metrics', timeout=5)
                    if response.getcode() == 200:
                        logger.info(f"Metrics server already running on port {port} (persistent service)")
                        _prometheus_metrics_server_started = True
                        return True
                except (urllib.error.URLError, urllib.error.HTTPError, Exception):
                    pass

                # Port is in use but not serving metrics properly
                logger.warning(f"Port {port} is in use but not serving metrics, cannot start server")
                return False

        except Exception:
            pass

        # Port is available, try to start our own HTTP server
        try:
            start_http_server(port)
            _prometheus_metrics_server_started = True
            logger.info(f"Started Prometheus metrics server on port {port}")
            return True
        except Exception as e:
            logger.error(f"Failed to start metrics server on port {port}: {e}")
            return False


class BaseMetricsMonitor:
    """Base class for Prometheus metrics monitoring using node-global HTTP server."""

    def __init__(self,
                 enable_http_server: bool = True,
                 service_name: str = "service"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.service_name = service_name

        self._metrics = {}

        # Set up multiprocess mode for Ray actors
        # This allows metrics from Ray worker processes to be aggregated
        # by the persistent node metrics service
        multiproc_dir = os.environ.get('PROMETHEUS_MULTIPROC_DIR')
        if not multiproc_dir:
            # Create a temporary directory for multiprocess metrics
            import tempfile
            multiproc_dir = tempfile.mkdtemp(prefix='prometheus_multiproc_')
            os.environ['PROMETHEUS_MULTIPROC_DIR'] = multiproc_dir
            self.logger.info(f"Created multiprocess directory: {multiproc_dir}")
        elif not os.path.exists(multiproc_dir):
            os.makedirs(multiproc_dir, exist_ok=True)
        else:
            self.logger.info(f"Using existing multiprocess directory: {multiproc_dir}")

        # Start the HTTP server if requested
        if enable_http_server:
            ensure_node_metrics_server()

    def get_metrics_endpoint(self) -> str | None:
        """Get the metrics endpoint URL if server is running."""
        if self.is_server_running():
            port = get_metrics_port()
            return f"http://localhost:{port}/metrics" if port else None
        return None

    def is_server_running(self) -> bool:
        """Check if the metrics server is running."""
        return _prometheus_metrics_server_started

    def get_or_create_histogram(self, name: str, description: str, labelnames: Iterable[str], buckets: Iterable[float] | None = None) -> Histogram:
        if name not in self._metrics:
            self._metrics[name] = self.create_histogram(name, description, labelnames, buckets)
        return self._metrics[name]

    def get_or_create_counter(self, name: str, description: str, labelnames: Iterable[str]) -> Counter:
        if name not in self._metrics:
            self._metrics[name] = self.create_counter(name, description, labelnames)
        return self._metrics[name]

    def get_or_create_gauge(self, name: str, description: str, labelnames: Iterable[str], multiprocess_mode: str = "all") -> Gauge:
        if name not in self._metrics:
            self._metrics[name] = self.create_gauge(name, description, labelnames, multiprocess_mode)
        return self._metrics[name]

    def create_histogram(self, name: str, description: str, labelnames: Iterable[str] | None = None, buckets: Iterable[float] | None = None) -> Histogram:
        """Create a histogram metric."""
        try:
            return Histogram(
                name,
                description,
                labelnames=labelnames or [],
                buckets=buckets or Histogram.DEFAULT_BUCKETS
            )
        except ValueError as e:
            # Metric with this name already registered – reuse existing collector
            existing = REGISTRY._names_to_collectors.get(name)  # type: ignore[attr-defined]
            if existing and isinstance(existing, Histogram):
                self.logger.debug(
                    "Reusing already-registered histogram '%s' (%s)", name, existing
                )
                return existing
            raise

    def create_counter(self, name: str, description: str, labelnames: Iterable[str] | None = None) -> Counter:
        """Create a counter metric."""
        try:
            return Counter(
                name,
                description,
                labelnames=labelnames or []
            )
        except ValueError:
            existing = REGISTRY._names_to_collectors.get(name)  # type: ignore[attr-defined]
            if existing and isinstance(existing, Counter):
                self.logger.debug("Reusing already-registered counter '%s'", name)
                return existing
            raise

    def create_gauge(self, name: str, description: str, labelnames: Iterable[str] | None = None, multiprocess_mode: str = "all") -> Gauge:
        """Create a gauge metric."""
        try:
            return Gauge(
                name,
                description,
                labelnames=labelnames or [],
                multiprocess_mode=multiprocess_mode
            )
        except ValueError:
            existing = REGISTRY._names_to_collectors.get(name)  # type: ignore[attr-defined]
            if existing and isinstance(existing, Gauge):
                self.logger.debug("Reusing already-registered gauge '%s'", name)
                return existing
            raise


@asynccontextmanager
async def record_duration(histogram: metrics.Histogram, attributes: dict | None = None):
    """Records duration of a code block using a histogram metric.

    Args:
        histogram: The histogram metric to record the duration
        attributes: Optional attributes to attach to the metric

    Example:
        async with record_duration(my_histogram, {"operation": "query"}):
            await do_something()
    """
    try:
        # Use monotonic time for more accurate duration measurement
        start_time = time.monotonic()
        yield
    except Exception as e:
        # Still record duration even if an error occurred
        duration = time.monotonic() - start_time
        histogram.record(duration, attributes)
        raise
    else:
        duration = time.monotonic() - start_time
        histogram.record(duration, attributes)


class TrackOperation:
    """Decorator to track operation metrics"""
    def __init__(
        self,
        duration_histogram: metrics.Histogram,
        op_counter: metrics.Counter
    ):
        self.histogram = duration_histogram
        self.op_counter = op_counter

    def __call__(self, operation: str):
        def decorator(func):
            @wraps(func)
            async def wrapper(self_1, *args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(self_1, *args, **kwargs)
                    self.op_counter.labels(
                        operation=operation,
                        status="success",
                        namespace=self_1.namespace
                    ).inc()
                    return result
                except Exception as e:
                    self.op_counter.labels(
                        operation=operation,
                        status="error",
                        namespace=self_1.namespace
                    ).inc()
                    raise
                finally:
                    duration = time.time() - start_time
                    self.histogram.labels(
                        operation=operation,
                        namespace=self_1.namespace
                    ).observe(duration)
            return wrapper
        return decorator
