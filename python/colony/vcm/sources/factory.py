
from typing import Any

from .context_page_source import ContextPageSource
from .blackboard_page_source import BlackboardContextPageSource
from .file_grouper_page_source import FileGrouperContextPageSource


class ContextPageSourceFactory:
    """Factory for creating ContextPageSource instances."""

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
        if source_type == "file_grouper":
            return FileGrouperContextPageSource(*args, **kwargs)
        elif source_type == "blackboard":
            return BlackboardContextPageSource(*args, **kwargs)
        else:
            raise ValueError(f"Unknown ContextPageSource type: {source_type}")

