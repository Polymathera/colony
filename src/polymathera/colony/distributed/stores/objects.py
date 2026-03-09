from __future__ import annotations
import logging
from typing import Any

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None  # type: ignore[assignment]

from ..configs import ObjectStorageConfig, ObjectStorageBackendType
from ...utils.retry import standard_retry

logger = logging.getLogger(__name__)


class ObjectStorage:

    def __init__(self, config: ObjectStorageConfig | None = None):
        self.config: ObjectStorageConfig | None = config
        self.s3_client = None

    async def initialize(self) -> None:
        self.config = await ObjectStorageConfig.check_or_get_component(self.config)
        self.s3_client = boto3.client(
            "s3",
            region_name=self.config.region_name,
            aws_access_key_id=self.config.access_key,
            aws_secret_access_key=self.config.secret_key,
        )
        logger.info("Initialized object storage")

    async def cleanup(self) -> None:
        from ...utils import cleanup_dynamic_asyncio_tasks
        try:
            await cleanup_dynamic_asyncio_tasks(self, raise_exceptions=False)
        except Exception as e:
            logger.warning(f"Error cleaning up ObjectStorage tasks: {e}")

    @standard_retry(logger)
    def store_object(self, bucket, key, data):
        try:
            self.s3_client.put_object(Bucket=bucket, Key=key, Body=data)
        except ClientError as e:
            logger.error(f"Failed to store object in S3: {e!s}")
            raise

    @standard_retry(logger)
    def get_object(self, bucket, key):
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            return response["Body"].read()
        except ClientError as e:
            logger.error(f"Failed to retrieve object from S3: {e!s}")
            raise

    @standard_retry(logger)
    def delete_object(self, bucket: str, key: str) -> None:
        try:
            self.s3_client.delete_object(Bucket=bucket, Key=key)
            logger.info(f"Successfully deleted object {key} from bucket {bucket}")
        except ClientError as e:
            logger.error(f"Failed to delete object {key} from bucket {bucket}: {e!s}")
            raise

    @standard_retry(logger)
    def list_files(self, bucket: str, prefix: str) -> list[dict[str, Any]]:
        try:
            response = self.s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
            if "Contents" not in response:
                return []
            return [
                {
                    "Key": item["Key"],
                    "Size": item["Size"],
                    "LastModified": item["LastModified"],
                }
                for item in response["Contents"]
            ]
        except ClientError as e:
            logger.error(
                f"Failed to list objects in bucket {bucket} with prefix {prefix}: {e!s}"
            )
            raise
