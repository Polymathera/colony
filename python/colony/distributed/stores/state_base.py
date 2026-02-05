from abc import ABC, abstractmethod
from pydantic import BaseModel


class StateStorageBackend(ABC):
    """Base class for storage backends"""

    @abstractmethod
    async def get_with_version(self, key: str) -> tuple[str | None, int]:
        """Get value and version atomically"""
        pass

    @abstractmethod
    async def compare_and_swap(self, key: str, value: str, version: int) -> bool:
        """Atomic compare-and-swap operation"""
        pass

    @abstractmethod
    async def cleanup(self, key: str) -> None:
        """Cleanup resources"""
        pass


class StateStorageBackendFactory(ABC):
    """Factory for creating StateStorageBackend instances"""

    @abstractmethod
    def create_backend(self, config: BaseModel) -> StateStorageBackend:
        """Create a StateStorageBackend instance based on the provided config"""
        pass

