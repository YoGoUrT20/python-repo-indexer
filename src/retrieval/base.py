"""
Base class for retrieval pipelines.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from ..models import Chunk, ContextItem, IndexTag


@dataclass
class RetrievalOptions:
    """Options for retrieval."""
    tags: List[IndexTag]
    query: str
    n_retrieve: int = 20
    n_final: int = 10
    include_embeddings: bool = True
    filter_directory: Optional[str] = None
    filter_paths: Optional[List[str]] = None


class RetrievalPipeline(ABC):
    """Base class for retrieval pipelines."""
    
    @abstractmethod
    async def retrieve(self, options: RetrievalOptions) -> List[Chunk]:
        """Retrieve chunks based on options.
        
        Args:
            options: Retrieval options
            
        Returns:
            List of retrieved chunks
        """
        pass
    
    @abstractmethod
    async def run(self, options: RetrievalOptions) -> List[ContextItem]:
        """Run the retrieval pipeline.
        
        Args:
            options: Retrieval options
            
        Returns:
            List of retrieved context items
        """
        pass 