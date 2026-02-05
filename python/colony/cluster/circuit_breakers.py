"""Circuit breaker configurations for LLM cluster operations.

This module provides centralized circuit breaker policies for protecting
against cascading failures in distributed LLM operations.
"""

from circuitbreaker import circuit

# Circuit breaker for critical inference operations
# Higher threshold (10 failures) as inference failures may be transient
# and we want to maintain availability for critical operations
inference_circuit = circuit(
    failure_threshold=10,
    recovery_timeout=30,
    expected_exception=Exception,
    name="vllm_inference"
)

# Circuit breaker for page loading operations
# Moderate tolerance (5 failures) for KV cache page operations
page_loading_circuit = circuit(
    failure_threshold=5,
    recovery_timeout=30,
    expected_exception=Exception,
    name="vllm_page_loading"
)

# Circuit breaker for S3 operations
# Stricter threshold (5 failures) since S3 issues are often systemic
# Longer recovery timeout (60s) to allow for AWS service recovery
s3_operations_circuit = circuit(
    failure_threshold=5,
    recovery_timeout=60,
    expected_exception=Exception,
    name="vllm_s3_operations"
)