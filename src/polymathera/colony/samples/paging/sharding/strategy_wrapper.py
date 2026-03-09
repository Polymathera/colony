"""Wrapper for GitRepoShardingStrategy that exposes file-to-page mapping for code analysis agents.

This wrapper provides access to the mapping of files to pages (shards), allowing code analysis
agents to understand which pages contain which files without needing to inspect shard contents.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from .types import RepositoryShard
from .prompting import ShardedInferencePromptStrategy
from .strategy import GitRepoShardingStrategy, ShardingConfig
from polymathera.colony.utils import setup_logger

logger = setup_logger(__name__)


class GitRepoShardingWithMapping(GitRepoShardingStrategy):
    """GitRepoShardingStrategy wrapper that exposes file-to-page mapping.

    This wrapper tracks which files end up in which pages/shards during the sharding process,
    providing easy lookup for code analysis agents.

    The mapping allows agents to:
    - Find which page(s) contain a specific file
    - Get all files in a specific page
    - Understand page structure without loading page content

    Example:
        ```python
        strategy = GitRepoShardingWithMapping(prompt_strategy, config)
        await strategy.initialize()

        # Create shards
        shards = await strategy.create_shards(repo_id, repo)

        # Query file-to-page mapping
        page_id = strategy.get_page_for_file("src/main.py")
        files = strategy.get_files_in_page(page_id)

        # Get complete mapping
        mapping = strategy.get_file_to_page_map()
        reverse_mapping = strategy.get_page_to_files_map()
        ```
    """

    def __init__(
        self,
        prompt_strategy: ShardedInferencePromptStrategy,
        config: ShardingConfig | None = None,
    ):
        super().__init__(prompt_strategy=prompt_strategy, config=config)

        # Mappings built during sharding
        self._file_to_page: dict[str, str] = {}  # file_path → page_id
        self._page_to_files: dict[str, list[str]] = defaultdict(list)  # page_id → [file_paths]
        self._page_metadata: dict[str, dict[str, Any]] = {}  # page_id → metadata

    def get_file_to_page_map(self) -> dict[str, str]:
        """Get complete file-to-page mapping.

        Returns:
            Dictionary mapping file paths to page IDs.

        Example:
            ```python
            mapping = strategy.get_file_to_page_map()
            for file_path, page_id in mapping.items():
                print(f"{file_path} → {page_id}")
            ```
        """
        return dict(self._file_to_page)

    def get_page_to_files_map(self) -> dict[str, list[str]]:
        """Get complete page-to-files mapping.

        Returns:
            Dictionary mapping page IDs to lists of file paths.

        Example:
            ```python
            mapping = strategy.get_page_to_files_map()
            for page_id, files in mapping.items():
                print(f"Page {page_id}: {len(files)} files")
            ```
        """
        return {k: list(v) for k, v in self._page_to_files.items()}

    def get_page_for_file(self, file_path: str) -> str | None:
        """Get the page ID containing a specific file.

        Args:
            file_path: Path to the file

        Returns:
            Page ID containing the file, or None if file not found

        Example:
            ```python
            page_id = strategy.get_page_for_file("src/main.py")
            if page_id:
                print(f"File is in page {page_id}")
            ```
        """
        return self._file_to_page.get(file_path)

    def get_files_in_page(self, page_id: str) -> list[str]:
        """Get all files in a specific page.

        Args:
            page_id: ID of the page/shard

        Returns:
            List of file paths in the page

        Example:
            ```python
            files = strategy.get_files_in_page("page-123")
            print(f"Page contains {len(files)} files")
            for file in files:
                print(f"  - {file}")
            ```
        """
        return list(self._page_to_files.get(page_id, []))

    def get_page_metadata(self, page_id: str) -> dict[str, Any]:
        """Get metadata for a specific page.

        Args:
            page_id: ID of the page/shard

        Returns:
            Dictionary with page metadata:
                - file_count: Number of files in page
                - total_tokens: Total token count
                - languages: Set of programming languages
                - has_binary: Whether page contains binary files

        Example:
            ```python
            metadata = strategy.get_page_metadata("page-123")
            print(f"Languages: {metadata['languages']}")
            print(f"Token count: {metadata['total_tokens']}")
            ```
        """
        return dict(self._page_metadata.get(page_id, {}))

    def get_page_count(self) -> int:
        """Get total number of pages created.

        Returns:
            Number of pages/shards

        Example:
            ```python
            print(f"Created {strategy.get_page_count()} pages")
            ```
        """
        return len(self._page_to_files)

    def get_file_count(self) -> int:
        """Get total number of files mapped.

        Returns:
            Number of unique files across all pages

        Example:
            ```python
            print(f"Processed {strategy.get_file_count()} files")
            ```
        """
        return len(self._file_to_page)

    async def create_shards(
        self, repo_id: str, repo, file_filter=None
    ) -> list[RepositoryShard]:
        """Create shards and build file-to-page mapping.

        Overrides parent to capture file-to-page relationships.

        Args:
            repo_id: Unique identifier for the repository
            repo: Git repository object
            file_filter: Optional filter function for files

        Returns:
            List of repository shards
        """
        # Clear existing mappings
        self._file_to_page.clear()
        self._page_to_files.clear()
        self._page_metadata.clear()

        # Create shards using parent implementation
        shards = await super().create_shards(repo_id, repo, file_filter)

        # Build mappings from created shards
        await self._build_mappings(shards)

        logger.info(
            f"Built file-to-page mapping: {self.get_file_count()} files across "
            f"{self.get_page_count()} pages"
        )

        return shards

    async def _build_mappings(self, shards: list[RepositoryShard]) -> None:
        """Build file-to-page and page-to-files mappings from shards.

        Args:
            shards: List of created repository shards
        """
        for shard in shards:
            page_id = shard.shard_id

            # Extract unique files from shard segments
            files_in_shard = set()
            for segment in shard.metadata.file_segments:
                files_in_shard.add(segment.file_path)

            # Update mappings
            for file_path in files_in_shard:
                self._file_to_page[file_path] = page_id
                self._page_to_files[page_id].append(file_path)

            # Extract page metadata
            self._page_metadata[page_id] = {
                "file_count": len(files_in_shard),
                "total_tokens": shard.metadata.total_tokens if shard.metadata else 0,
                "languages": set(
                    seg.language for seg in shard.metadata.file_segments if seg.language
                ),
                "has_binary": (
                    any(shard.metadata.binary_files) if shard.metadata and shard.metadata.binary_files else False
                ),
                "created_at": shard.metadata.created_at if shard.metadata else None,
            }

    def query_files_by_language(self, language: str) -> list[str]:
        """Find all files of a specific programming language.

        Args:
            language: Programming language name (e.g., "python", "javascript")

        Returns:
            List of file paths for that language

        Example:
            ```python
            py_files = strategy.query_files_by_language("python")
            print(f"Found {len(py_files)} Python files")
            ```
        """
        result = []
        for page_id, metadata in self._page_metadata.items():
            if language in metadata.get("languages", set()):
                result.extend(self.get_files_in_page(page_id))
        return result

    def query_pages_by_language(self, language: str) -> list[str]:
        """Find all pages containing a specific programming language.

        Args:
            language: Programming language name

        Returns:
            List of page IDs containing that language

        Example:
            ```python
            pages = strategy.query_pages_by_language("python")
            print(f"Python code appears in {len(pages)} pages")
            ```
        """
        return [
            page_id
            for page_id, metadata in self._page_metadata.items()
            if language in metadata.get("languages", set())
        ]