"""
Centralized retry utilities for the polymathera codebase.

This module provides a unified way to handle retries with proper logging
across all components, ensuring consistent behavior and debugging capabilities.
"""

import logging
from typing import Any, Callable

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    before_log,
    after_log,
    before_sleep_log,
    RetryCallState,
)


def create_retry_with_logging(
    logger: logging.Logger,
    stop_attempts: int = 3,
    wait_multiplier: float = 1,
    wait_min: float = 4,
    wait_max: float = 10,
    **retry_kwargs: Any
) -> Callable:
    """
    Create a retry decorator with comprehensive logging.

    Args:
        logger: Logger instance to use for all retry logging
        stop_attempts: Maximum number of retry attempts
        wait_multiplier: Multiplier for exponential backoff
        wait_min: Minimum wait time between retries
        wait_max: Maximum wait time between retries
        **retry_kwargs: Additional arguments to pass to the retry decorator

    Returns:
        Configured retry decorator with logging
    """
    return retry(
        stop=stop_after_attempt(stop_attempts),
        wait=wait_exponential(multiplier=wait_multiplier, min=wait_min, max=wait_max),
        before=before_log(logger, logging.DEBUG),
        after=after_log(logger, logging.WARNING),
        before_sleep=before_sleep_log(logger, logging.WARNING, exc_info=True),
        **retry_kwargs
    )


def create_custom_retry_callback(logger: logging.Logger) -> Callable[[RetryCallState], None]:
    """
    Create a custom retry callback that logs detailed information about each attempt.

    Args:
        logger: Logger instance to use

    Returns:
        Callback function for retry state logging
    """
    def log_retry_attempt_1(retry_state: RetryCallState) -> None:
        """Log detailed information about the retry attempt."""
        if retry_state.attempt_number == 1:
            logger.info(
                f"Starting {retry_state.fn.__name__} (attempt {retry_state.attempt_number})"
            )
        else:
            logger.warning(
                f"Retrying {retry_state.fn.__name__} (attempt {retry_state.attempt_number}) "
                f"after {retry_state.seconds_since_start:.2f}s - "
                f"Previous attempt failed: {retry_state.outcome.exception()}"
            )

    def log_retry_attempt_2(retry_state: RetryCallState) -> None:
        logger.warning(f"Retrying {retry_state.outcome} due to {retry_state.outcome.value}")

    return log_retry_attempt_1




def create_comprehensive_retry(
    logger: logging.Logger,
    stop_attempts: int = 3,
    wait_multiplier: float = 1,
    wait_min: float = 4,
    wait_max: float = 10,
    **retry_kwargs: Any
) -> Callable:
    """
    Create a retry decorator with comprehensive custom logging that shows all intermediate errors.

    This provides the most detailed logging including the specific exception from each failed attempt.

    Args:
        logger: Logger instance to use for all retry logging
        stop_attempts: Maximum number of retry attempts
        wait_multiplier: Multiplier for exponential backoff
        wait_min: Minimum wait time between retries
        wait_max: Maximum wait time between retries
        **retry_kwargs: Additional arguments to pass to the retry decorator

    Returns:
        Configured retry decorator with comprehensive logging
    """
    return retry(
        stop=stop_after_attempt(stop_attempts),
        wait=wait_exponential(multiplier=wait_multiplier, min=wait_min, max=wait_max),
        before_sleep=create_custom_retry_callback(logger),
        **retry_kwargs
    )


# Convenience functions for common retry patterns
def standard_retry(logger: logging.Logger, **kwargs: Any) -> Callable:
    """Standard retry with 3 attempts and exponential backoff."""
    return create_retry_with_logging(logger, **kwargs)


def aggressive_retry(logger: logging.Logger, **kwargs: Any) -> Callable:
    """More aggressive retry with 5 attempts and longer waits."""
    return create_retry_with_logging(
        logger,
        stop_attempts=5,
        wait_min=2,
        wait_max=30,
        **kwargs
    )


def quick_retry(logger: logging.Logger, **kwargs: Any) -> Callable:
    """Quick retry with shorter waits for fast operations."""
    return create_retry_with_logging(
        logger,
        stop_attempts=3,
        wait_min=1,
        wait_max=5,
        **kwargs
    )