"""
Base class for codebase indexers.
"""
from abc import ABC, abstractmethod
from typing import Generator, List, Optional

from ..models import Chunk, IndexTag, IndexingProgressUpdate, PathAndCacheKey, RefreshIndexResults


class CodebaseIndex(ABC):
    """Base class for all codebase indexers.

    Each indexer is responsible for creating and maintaining a specific
    type of index (chunks, full-text, vector embeddings, etc).
    """

    @property
    @abstractmethod
    def artifact_id(self) -> str:
        """Get the unique identifier for this artifact type."""
        pass

    @property
    @abstractmethod
    def relative_expected_time(self) -> float:
        """Get the relative expected time for this indexer (used for progress calculation)."""
        pass

    @abstractmethod
    async def update(
        self,
        tag: IndexTag,
        results: RefreshIndexResults,
        mark_complete: callable,
        repo_name: Optional[str] = None
    ) -> Generator[IndexingProgressUpdate, None, None]:
        """Update the index with new and changed files.

        Args:
            tag: The tag identifying the branch and directory being indexed
            results: The results of the refresh operation
            mark_complete: Callback to mark files as complete
            repo_name: Optional repository name

        Yields:
            Progress updates
        """
        pass

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        tags: List[IndexTag],
        n: int = 10,
        directory: Optional[str] = None,
        filter_paths: Optional[List[str]] = None
    ) -> List[Chunk]:
        """Retrieve chunks from the index matching the query.

        Args:
            query: The search query
            tags: The tags to search in
            n: Maximum number of results to return
            directory: Optional directory to limit search to
            filter_paths: Optional list of file paths to filter results to

        Returns:
            List of matching chunks
        """
        pass
