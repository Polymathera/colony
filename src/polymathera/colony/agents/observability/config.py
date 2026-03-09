"""Configuration for the Colony tracing system."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TracingConfig:
    """Configuration for agent observability/tracing.

    Attributes:
        enabled: Master switch — when False, no tracing overhead
        kafka_bootstrap: Kafka broker address
        kafka_topic: Kafka topic for span events
        sample_rate: Fraction of traces to capture (0.0-1.0)
        max_input_chars: Truncate input summaries beyond this length
        max_output_chars: Truncate output summaries beyond this length
        flush_interval: Seconds between background flushes to Kafka
        flush_batch_size: Max spans per Kafka produce batch
        capture_infer_inputs: Include full LLM prompt text (expensive)
        capture_action_results: Include action result data in spans
    """

    enabled: bool = False
    kafka_bootstrap: str = "kafka:9092"
    kafka_topic: str = "colony.spans"
    sample_rate: float = 1.0
    max_input_chars: int = 1500
    max_output_chars: int = 1500
    max_infer_chars: int = 200000
    flush_interval: float = 0.5
    flush_batch_size: int = 50
    capture_infer_inputs: bool = True
    capture_action_results: bool = True
