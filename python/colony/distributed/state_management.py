from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from enum import Enum
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

from .stores.state_base import StateStorageBackendFactory, StateStorageBackend
from .stores.state_redis import RedisStateStorageBackendFactory, RedisStateStorageConfig

logger = logging.getLogger(__name__)


class SharedState(BaseModel):
    """Shared state container base class for Polymathera actor pools"""

    writable: bool = Field(
        default=False,
        description="Whether the state is writable (acquired by a write transaction)",
    )


T = TypeVar("T", bound=SharedState)


class StateTransactionType(Enum):
    """Type of state transaction"""

    READ = "read"
    WRITE = "write"


class StateTransactionError(Exception):
    """Raised when a state transaction fails due to version mismatch"""
    pass


class StateManager(Generic[T]):
    """
    Generic state manager that supports multiple storage backends and optimistic locking.
    The state type T must be a Pydantic model to ensure proper serialization/deserialization.
    """

    def __init__(
        self,
        state_type: type[T],
        state_key: str,
        config: BaseModel = RedisStateStorageConfig(),
        factory: StateStorageBackendFactory | None = None,
    ):
        """
        Initialize state manager.

        Args:
            state_type: The Pydantic model class for the state
            state_key: Unique key for this state instance
            config: Configuration for the storage backend
            factory: StateStorageBackendFactory | None = None,
        """
        self.factory = factory or RedisStateStorageBackendFactory()
        self.state_type = state_type
        self.state_key = state_key
        self.config = config
        self.state_storage: StateStorageBackend | None = None
        self.state_lock = asyncio.Lock()
        self._state_version = 0
        self._current_state: T | None = None

    async def initialize(self) -> None:
        self.state_storage = self.factory.create_backend(self.config)

    async def _load_state(self) -> T:
        """Load state from storage with version"""
        try:
            state_data, version = await self.state_storage.get_with_version(
                self.state_key
            )
            if state_data:
                self._state_version = version
                return self.state_type.model_validate_json(state_data)
            return self.state_type()
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return self.state_type()

    async def read_transaction(self, max_retries: int = 3) -> AsyncIterator[T]:
        """Read-only transaction iterator"""
        async for state in self.transaction(
            StateTransactionType.READ, max_retries=max_retries
        ):
            yield state

    async def write_transaction(self, max_retries: int = 3) -> AsyncIterator[T]:
        """Write transaction iterator"""
        logger.info(f"StateManager.write_transaction: {self.state_key}")
        async for state in self.transaction(
            StateTransactionType.WRITE, max_retries=max_retries
        ):
            yield state

    async def transaction(
        self,
        transaction_type: StateTransactionType = StateTransactionType.WRITE,
        max_retries: int = 3,
    ) -> AsyncIterator[T]:
        """
        State transaction iterator that yields fresh state on each retry.
        It supporots atomic state operations with optimistic locking.

        For write transactions:
        1. Loads current state and version
        2. Yields state for modification
        3. Attempts to save state with compare-and-swap
        4. If save fails (version changed), loads new state and yields again
        5. Continues until success or max retries (raises StateTransactionError)

        For read transactions:
        1. Loads current state
        2. Yields state once
        3. No version check or save

        Usage:
            async for state in state_manager.transaction():
                # This block may execute multiple times with fresh state
                # Make allocation decisions based on current state
                state.allocate_gpu(...)  # These decisions must be idempotent
        """
        retry_count = 0

        while True:
            logger.info(
                f"StateManager.transaction: {self.state_key} {transaction_type} {retry_count}/{max_retries}"
            )
            try:
                async with self.state_lock:
                    logger.info("StateManager.transaction: lock acquired")
                    # Load fresh state
                    current_state = await self._load_state()
                    logger.info("StateManager.transaction: state loaded")
                    current_version = self._state_version
                    self._current_state = current_state.model_copy()

                    self._current_state.writable = (
                        transaction_type == StateTransactionType.WRITE
                    )

                    # Yield state to caller. Let caller modify state
                    yield self._current_state

                    # For read transactions, we're done
                    if transaction_type == StateTransactionType.READ:
                        break

                    logger.info("StateManager.transaction: saving state")
                    # For write transactions, try to save state with atomic compare-and-swap
                    success = await self.state_storage.compare_and_swap(
                        key=self.state_key,
                        value=self._current_state.model_dump_json(),
                        version=current_version,
                    )

                    if success:
                        logger.info("StateManager.transaction: state saved")
                        # Update version and exit
                        self._state_version += 1
                        break

                    logger.info(
                        "StateManager.transaction: save failed, retry with fresh state"
                    )
                    # Save failed, retry with fresh state if attempts remain
                    retry_count += 1
                    if retry_count >= max_retries:
                        raise StateTransactionError(
                            f"Failed to save state after {max_retries} retries"
                        )

                    await asyncio.sleep(self.config.retry_delay * retry_count)
                    continue

            except Exception as e:
                if not isinstance(e, StateTransactionError):
                    logger.error(f"Error in state transaction: {e}")
                raise
            finally:
                # Reset state after each attempt
                if self._current_state:
                    self._current_state.writable = False

    async def get_current_state(self) -> T:
        """Get the current state without starting a transaction"""
        async for state in self.transaction(StateTransactionType.READ):
            return state

    async def cleanup(self):
        """Cleanup resources"""
        await self.state_storage.cleanup(self.state_key)
