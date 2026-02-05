"""S3 model loading utilities for LLM cluster.

This module provides utilities for downloading and extracting models from S3
with exponential backoff retry and circuit breaker protection.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import shutil
import tarfile
import tempfile
from typing import TYPE_CHECKING

import boto3

if TYPE_CHECKING:
    pass

from .circuit_breakers import s3_operations_circuit

logger = logging.getLogger(__name__)


class S3ModelLoader:
    """Handles downloading and extracting models from S3.

    Features:
    - Exponential backoff retry (5s, 10s, 20s, 40s, 80s, 160s, 300s max)
    - Circuit breaker protection (opens after 5 consecutive failures)
    - Automatic cleanup on process exit

    Example:
        ```python
        loader = S3ModelLoader(
            bucket="my-models",
            model_name="meta-llama/Llama-3.1-8B",
            retry_attempts=10
        )
        model_path = await loader.download_and_extract()
        if model_path:
            # Use model_path for vLLM initialization
            ...
        ```
    """

    def __init__(
        self,
        bucket: str,
        model_name: str,
        retry_attempts: int = 10,
    ):
        """Initialize S3 model loader.

        Args:
            bucket: S3 bucket name
            model_name: Model name (e.g., "meta-llama/Llama-3.1-8B")
            retry_attempts: Number of retry attempts (default: 10)
        """
        self.bucket = bucket
        self.model_name = model_name
        self.retry_attempts = retry_attempts

        # Exponential backoff parameters
        self.base_delay = 5.0
        self.max_delay = 300.0
        self.jitter_factor = 0.1

    @s3_operations_circuit
    async def download_and_extract(self) -> str | None:
        """Download and extract model from S3 with exponential backoff retry.

        Returns:
            Path to extracted model directory, or None if download failed

        Note:
            - Uses exponential backoff: 5s, 10s, 20s, 40s, 80s, 160s, 300s (capped)
            - Circuit breaker opens after 5 consecutive failures, recovers after 60s
        """
        logger.info(f"Downloading model {self.model_name} from S3 bucket {self.bucket}")

        # Get S3 model key
        archive_key = f"models/{self.model_name}/{self.model_name.replace('/', '_')}.tar.gz"

        for attempt in range(self.retry_attempts):
            try:
                s3_client = boto3.client("s3")

                # Check if archive exists
                try:
                    s3_client.head_object(Bucket=self.bucket, Key=archive_key)
                    logger.info(f"Found model archive at s3://{self.bucket}/{archive_key}")
                except Exception as e:
                    logger.error(f"Model archive not found at s3://{self.bucket}/{archive_key}: {e}")
                    return None

                # Create temporary directory
                temp_dir = tempfile.mkdtemp(prefix=f"model_{self.model_name.replace('/', '_')}_")
                archive_path = os.path.join(temp_dir, f"{self.model_name.replace('/', '_')}.tar.gz")
                model_dir = os.path.join(temp_dir, self.model_name.replace('/', '_'))

                # Download archive
                logger.info(f"Downloading model archive (attempt {attempt + 1}/{self.retry_attempts})")
                s3_client.download_file(self.bucket, archive_key, archive_path)

                # Extract archive
                logger.info(f"Extracting model archive to {model_dir}")
                with tarfile.open(archive_path, 'r:gz') as tar:
                    tar.extractall(path=temp_dir)

                # Remove archive to save space
                os.remove(archive_path)

                logger.info(f"Successfully downloaded and extracted model to {model_dir}")

                # Register cleanup handler
                import atexit
                atexit.register(lambda: shutil.rmtree(temp_dir, ignore_errors=True))

                return model_dir

            except Exception as e:
                # Calculate delay with exponential backoff and jitter
                delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                jitter = delay * self.jitter_factor * (2 * random.random() - 1)
                actual_delay = delay + jitter

                if attempt < self.retry_attempts - 1:
                    logger.warning(
                        f"Failed to download model from S3 (attempt {attempt + 1}/{self.retry_attempts}): {e}. "
                        f"Retrying in {actual_delay:.1f}s..."
                    )
                    await asyncio.sleep(actual_delay)
                else:
                    logger.error(
                        f"Failed to download model from S3 after {self.retry_attempts} attempts: {e}",
                        exc_info=True
                    )
                    return None

        return None