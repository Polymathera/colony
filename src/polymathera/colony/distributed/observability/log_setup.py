"""One-call setup for the Kafka log pipeline.

Call ``attach_kafka_log_handler()`` from any ``@serving.initialize_deployment``
hook to start publishing all logs from the ``polymathera.colony`` namespace
to the ``colony.logs`` Kafka topic.

Safe to call multiple times — only the first call attaches the handler.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_handler: Any = None  # KafkaLogHandler singleton


async def attach_kafka_log_handler(
    kafka_bootstrap: str | None = None,
    topic: str = "colony.logs",
    level: int = logging.INFO,
) -> bool:
    """Attach KafkaLogHandler to the ``polymathera.colony`` root logger.

    Args:
        kafka_bootstrap: Kafka broker address.  Falls back to
            ``$KAFKA_BOOTSTRAP`` env var.  If neither is set, does nothing.
        topic: Kafka topic name.
        level: Minimum log level to publish.

    Returns:
        True if the handler was attached (or was already attached), False if
        Kafka is unavailable.
    """
    global _handler

    if _handler is not None:
        return True  # Already attached

    bootstrap = kafka_bootstrap or os.environ.get("KAFKA_BOOTSTRAP")
    if not bootstrap:
        logger.debug("KAFKA_BOOTSTRAP not set — log pipeline disabled")
        return False

    try:
        from .log_handler import KafkaLogHandler

        handler = KafkaLogHandler(
            kafka_bootstrap=bootstrap,
            topic=topic,
            level=level,
        )
        await handler.start()

        # Attach to the colony root logger — captures all sub-loggers
        colony_logger = logging.getLogger("polymathera.colony")
        colony_logger.addHandler(handler)
        _handler = handler

        logger.info("KafkaLogHandler attached to polymathera.colony logger (topic=%s)", topic)
        return True

    except Exception as e:
        logger.warning("Failed to attach KafkaLogHandler: %s", e)
        return False


async def detach_kafka_log_handler() -> None:
    """Stop and remove the KafkaLogHandler."""
    global _handler

    if _handler is None:
        return

    colony_logger = logging.getLogger("polymathera.colony")
    colony_logger.removeHandler(_handler)

    await _handler.stop()
    _handler = None
