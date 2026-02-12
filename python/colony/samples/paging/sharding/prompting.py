
from abc import ABC, abstractmethod
from typing import Any


class ShardedInferencePromptStrategy(ABC):

    @abstractmethod
    async def get_shard_setup_prompt(
        self, shard_type: str, shard_content: str, shard_metadata: dict[str, Any]
    ) -> str:
        """
        Generates the immutable, task-agnostic portion of the prompt
        that can be cached in GPU KV memory
        """
        pass


class IdentityPromptStrategy(ShardedInferencePromptStrategy):
    """Simple prompt strategy that returns shard content as-is.

    Suitable for:
    - Page graph building (where annotated_content is stored as page text)
    - Remote LLM deployments (system prompt is configured separately)
    - Cases where no prompt annotation is needed
    """

    async def get_shard_setup_prompt(
        self, shard_type: str, shard_content: str, shard_metadata: dict[str, Any]
    ) -> str:
        return shard_content

