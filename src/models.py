"""
Data models for the repository indexing and context search system.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


class IndexResultType(str, Enum):
    """Types of results that can come from the indexing process."""
    COMPUTE = "compute"
    ADD_TAG = "add_tag"
    REMOVE_TAG = "remove_tag"
    DELETE = "delete"


class IndexingStatus(str, Enum):
    """Status of the indexing process."""
    SCANNING = "scanning"
    INDEXING = "indexing"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class IndexTag:
    """A tag that identifies a specific version of a file in the index."""
    directory: str
    branch: str
    artifact_id: str

    @property
    def tag_string(self) -> str:
        """Convert tag to string format for storage."""
        return f"{self.directory}::{self.branch}::{self.artifact_id}"

    @classmethod
    def from_string(cls, tag_string: str) -> "IndexTag":
        """Create tag from string format."""
        directory, branch, artifact_id = tag_string.split("::")
        return cls(directory=directory, branch=branch, artifact_id=artifact_id)


@dataclass
class PathAndCacheKey:
    """Information about a file to be indexed."""
    path: str
    cache_key: str  # Usually a hash of the file contents


@dataclass
class IndexingProgressUpdate:
    """Progress update during indexing."""
    progress: float  # 0.0 to 1.0
    desc: str
    status: IndexingStatus


@dataclass
class RefreshIndexResults:
    """Results of refreshing the index."""
    compute: List[PathAndCacheKey]
    add_tag: List[PathAndCacheKey]
    remove_tag: List[PathAndCacheKey]
    delete: List[PathAndCacheKey]


@dataclass
class Chunk:
    """A chunk of code from a file."""
    filepath: str
    start_line: int
    end_line: int
    content: str
    digest: str  # Cache key/hash
    index: int = 0  # Position within file

    @property
    def location(self) -> str:
        """Get the location string for this chunk."""
        return f"{self.filepath}:{self.start_line}-{self.end_line}"


@dataclass
class ContextItem:
    """An item returned from context search."""
    name: str
    description: str
    content: str
    filepath: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None

    @classmethod
    def from_chunk(cls, chunk: Chunk, description: Optional[str] = None) -> "ContextItem":
        """Create a context item from a chunk."""
        return cls(
            name=chunk.filepath,
            description=description or f"Lines {chunk.start_line}-{chunk.end_line}",
            content=chunk.content,
            filepath=chunk.filepath,
            start_line=chunk.start_line,
            end_line=chunk.end_line
        )
