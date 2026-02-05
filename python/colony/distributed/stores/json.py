import hashlib
import json
import logging
from typing import Any
from pydantic import Field

import boto3
from botocore.exceptions import ClientError


from ...distributed import get_polymathera
from ..configs import JsonStorageConfig
from ...utils.retry import standard_retry


logger = logging.getLogger(__name__)




class JsonStorage:
    """
    PROMPT: This class is used to store and retrieve JSON data in a distributed manner.
    Make this class robust and production-ready for the specific needs of
    Polymathera, handle failures, errors, edge cases and unexpected events.
    Also, use the most popular Python frameworks and AWS SDKs for storage of JSON data.
    This class will be instantiated and initialized in every microservice in the
    Polymathera system (which can have thousands of microservices).

    This class uses AWS DynamoDB for persistent storage and distributed cache.
    """

    def __init__(self, config: JsonStorageConfig | None = None):
        self.config = config
        self.cache = None
        self.dynamodb = None
        self.table = None
        self.cache_ttl = None

    async def initialize(self) -> None:
        self.config = await JsonStorageConfig.check_or_get_component(self.config)
        await self._initialize_cache()
        self.dynamodb = boto3.resource("dynamodb", region_name=self.config.aws_region)
        self.table = self.dynamodb.Table(self.config.dynamodb_table)
        self.cache_ttl = self.config.cache_ttl
        logger.info("Initialized JSON storage")

    async def cleanup(self) -> None:
        from ...utils import cleanup_dynamic_asyncio_tasks

        try:
            await cleanup_dynamic_asyncio_tasks(self, raise_exceptions=False)
        except Exception as e:
            logger.warning(f"Error cleaning up JSONStorage tasks: {e}")

    async def _initialize_cache(self):
        self.cache = await get_polymathera().create_distributed_simple_cache(
            # dict[str, Any],
            namespace="json:data",  # TODO: Scope is global to all VMRs?
            config=self.config.cache_config,
        )

    @standard_retry(logger)
    async def save(self, data: dict[str, Any], metadata: dict[str, Any]) -> None:
        """
        Save JSON data with associated metadata.

        :param d: The JSON data to save
        :param metadata: Metadata associated with the JSON data. All keys and values must be JSON serializable.
        """
        try:
            item_id = self._generate_id(metadata)
            item = {
                "item_id": item_id,
                "data": json.dumps(data),
                "metadata": json.dumps(metadata),
            }
            await self.table.put_item(Item=item)
            await self.cache.set(item_id, data, ttl=self.cache_ttl)
            logger.info(f"Successfully saved item with ID: {item_id}")
        except ClientError as e:
            logger.error(f"Failed to save item to DynamoDB: {e!s}")
            raise

    @standard_retry(logger)
    async def load(self, metadata: dict[str, Any]) -> dict[str, Any] | None:
        """
        Load JSON data based on metadata.

        :param metadata: Metadata used to identify the JSON data
        :return: The JSON data if found, None otherwise
        """
        item_id = self._generate_id(metadata)
        logger.info(f"_____ JsonStorage: Loading item with ID: {item_id}: {metadata}")

        # Try to get from cache first
        cached_data = await self.cache.get(item_id)
        if cached_data:
            logger.info(f"_____ JsonStorage: Cache hit for item ID: {item_id}")
            return cached_data

        try:
            logger.info(f"_____ JsonStorage: Loading item with ID: {item_id} from DynamoDB")
            response = self.table.get_item(Key={"item_id": item_id})
            logger.info(f"_____ JsonStorage: Loading item with ID: {item_id} from DynamoDB: {response}")
            item = response.get("Item")
            if item:
                data = json.loads(item["data"])
                await self.cache.set(item_id, data, ttl=self.cache_ttl)
                logger.info(f"_____ JsonStorage: Successfully loaded item with ID: {item_id}")
                return data
            else:
                logger.info(f"_____ JsonStorage: No item found with ID: {item_id}")
                return None
        except ClientError as e:
            logger.error(f"_____ JsonStorage: Failed to load item from DynamoDB: {e!s}")
            raise

    @standard_retry(logger)
    async def contains(self, metadata: dict[str, Any]) -> bool:
        """
        Check if JSON data with given metadata exists.

        :param metadata: Metadata used to identify the JSON data
        :return: True if the data exists, False otherwise
        """
        item_id = self._generate_id(metadata)

        # Check cache first
        if await self.cache.exists(item_id):
            logger.info(f"_____ JsonStorage: Cache hit for contains check, item ID: {item_id}")
            return True

        try:
            response = self.table.get_item(
                Key={"item_id": item_id}, ProjectionExpression="id"
            )
            return "Item" in response
        except ClientError as e:
            logger.error(f"Failed to check item existence in DynamoDB: {e!s}")
            raise

    def _generate_id(self, metadata: dict[str, Any]) -> str:
        """Generate a unique ID based on metadata."""
        try:
            logger.info(f"_____ JsonStorage: Generating ID for metadata: {metadata}")
            metadata_str = json.dumps(metadata, sort_keys=True)
            logger.info(f"_____ JsonStorage: Generated ID for metadata: {metadata_str}")
            return hashlib.md5(metadata_str.encode()).hexdigest()
        except Exception as e:
            logger.error(f"_____ JsonStorage: Failed to generate ID for metadata: {e!s}")
            raise
