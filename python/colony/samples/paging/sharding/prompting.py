
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

