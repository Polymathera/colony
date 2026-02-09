
from typing import Any

from .context_page_source import ContextPageSource

class ContextPageSourceFactory:
    """Factory for creating ContextPageSource instances."""

    _registry: dict[str, type[ContextPageSource]] = {}

    @staticmethod
    def register_new_source_type(source_type: str):
        """A decorator to register a new ContextPageSource type."""
        def decorator(source_class: type[ContextPageSource]):
            if not hasattr(ContextPageSourceFactory, "_registry"):
                ContextPageSourceFactory._registry = {}
            ContextPageSourceFactory._registry[source_type] = source_class
            return source_class
        return decorator

    @staticmethod
    def create(
        source_type: str = "file_grouper",
        *args: Any,
        **kwargs: Any
    ) -> ContextPageSource:
        """Create and initialize a ContextPageSource.

        Args:
            source_type: Type of source to create ("file_grouper", etc.)
            *args: Positional arguments for source constructor
            **kwargs: Keyword arguments for source constructor

        Returns:
            Initialized ContextPageSource instance
        """
        if source_type in ContextPageSourceFactory._registry:
            return ContextPageSourceFactory._registry[source_type](*args, **kwargs)
        else:
            raise ValueError(f"Unknown ContextPageSource type: {source_type}")

