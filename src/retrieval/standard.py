"""
Standard retrieval pipeline implementation.
"""
import asyncio
from typing import Dict, List, Optional, Set, Tuple

from ..indexers import FullTextSearchCodebaseIndex, VectorCodebaseIndex
from ..models import Chunk, ContextItem, IndexTag
from .base import RetrievalOptions, RetrievalPipeline


class StandardRetrievalPipeline(RetrievalPipeline):
    """Standard retrieval pipeline that combines vector and text search.
    
    This pipeline:
    1. Retrieves results from both vector and full-text search
    2. Combines and deduplicates results
    3. Ranks them based on a combination of scores
    4. Returns the top N results
    """
    
    def __init__(
        self,
        db_path: str,
        vector_model_name: str = "all-MiniLM-L6-v2",
        vector_weight: float = 0.7,
        fulltext_weight: float = 0.3
    ):
        """Initialize the standard retrieval pipeline.
        
        Args:
            db_path: Path to the database
            vector_model_name: Name of the vector embedding model
            vector_weight: Weight for vector search results (0-1)
            fulltext_weight: Weight for full-text search results (0-1)
        """
        self.db_path = db_path
        self.vector_model_name = vector_model_name
        self.vector_weight = vector_weight
        self.fulltext_weight = fulltext_weight
        
        # Initialize indexers
        self.vector_index = VectorCodebaseIndex(db_path, model_name=vector_model_name)
        self.fulltext_index = FullTextSearchCodebaseIndex(db_path)
        
        # Simple debug flag
        self.debug = False
    
    def _debug(self, message: str) -> None:
        """Print debug message if debug is enabled."""
        if self.debug:
            print(f"[DEBUG:Pipeline] {message}")
    
    async def retrieve(self, options: RetrievalOptions) -> List[Chunk]:
        """Retrieve chunks using both vector and full-text search.
        
        Args:
            options: Retrieval options
            
        Returns:
            Combined list of chunks
        """
        self.debug = True  # Enable debug output
        
        # Fallback to a simple database query if needed
        try:
            # First, try a fallback approach to ensure we get at least some results
            fallback_chunks = await self._simple_fallback_retrieve(options)
            self._debug(f"Fallback retrieval found {len(fallback_chunks)} chunks")
            
            # Run both retrievals concurrently
            vector_future = None
            fulltext_future = None
            
            if options.include_embeddings:
                self._debug("Vector search enabled, starting retrieval...")
                vector_future = asyncio.create_task(
                    self.vector_index.retrieve(
                        query=options.query,
                        tags=options.tags,
                        n=options.n_retrieve,
                        directory=options.filter_directory,
                        filter_paths=options.filter_paths
                    )
                )
            
            self._debug("Starting full-text search retrieval...")
            fulltext_future = asyncio.create_task(
                self.fulltext_index.retrieve(
                    query=options.query,
                    tags=options.tags,
                    n=options.n_retrieve,
                    directory=options.filter_directory,
                    filter_paths=options.filter_paths
                )
            )
            
            # Gather results
            results = []
            scores: Dict[str, float] = {}
            seen_chunks: Set[str] = set()
            
            # Get vector results if available
            if vector_future:
                try:
                    vector_results = await vector_future
                    self._debug(f"Vector search returned {len(vector_results)} results")
                    
                    # Score based on position (higher position = higher score)
                    for i, chunk in enumerate(vector_results):
                        chunk_key = f"{chunk.filepath}:{chunk.start_line}-{chunk.end_line}"
                        
                        if chunk_key not in seen_chunks:
                            seen_chunks.add(chunk_key)
                            # Score inversely proportional to position (0.0-1.0)
                            score = 1.0 - (i / len(vector_results)) if vector_results else 0.0
                            scores[chunk_key] = score * self.vector_weight
                            results.append(chunk)
                except Exception as e:
                    self._debug(f"Error in vector retrieval: {e}")
            
            # Get full-text results
            try:
                fulltext_results = await fulltext_future
                self._debug(f"Full-text search returned {len(fulltext_results)} results")
                
                # Score and add full-text results
                for i, chunk in enumerate(fulltext_results):
                    chunk_key = f"{chunk.filepath}:{chunk.start_line}-{chunk.end_line}"
                    
                    # Score inversely proportional to position (0.0-1.0)
                    score = 1.0 - (i / len(fulltext_results)) if fulltext_results else 0.0
                    score = score * self.fulltext_weight
                    
                    if chunk_key in seen_chunks:
                        # Add to existing score
                        scores[chunk_key] += score
                    else:
                        # Add as new result
                        seen_chunks.add(chunk_key)
                        scores[chunk_key] = score
                        results.append(chunk)
            except Exception as e:
                self._debug(f"Error in full-text retrieval: {e}")
            
            # Sort by combined score
            results_with_scores = [(chunk, scores.get(f"{chunk.filepath}:{chunk.start_line}-{chunk.end_line}", 0.0)) 
                                for chunk in results]
            results_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Take top N
            top_results = [chunk for chunk, _ in results_with_scores[:options.n_final]]
            self._debug(f"Combined results: {len(top_results)}")
            
            # If no results from standard retrieval, use fallback results
            if not top_results and fallback_chunks:
                self._debug("Using fallback results since standard retrieval returned no results")
                return fallback_chunks[:options.n_final]
                
            return top_results
        except Exception as e:
            self._debug(f"Error in retrieval pipeline: {e}")
            import traceback
            traceback.print_exc()
            
            # Return fallback results if available
            if fallback_chunks:
                self._debug("Using fallback results due to error")
                return fallback_chunks[:options.n_final]
            return []
    
    async def _simple_fallback_retrieve(self, options: RetrievalOptions) -> List[Chunk]:
        """Simple fallback retrieval method that directly queries the database.
        
        This ensures we get at least some results even if the more sophisticated
        search methods fail.
        
        Args:
            options: Retrieval options
            
        Returns:
            List of chunks
        """
        from ..database import Database
        
        db = Database(self.db_path)
        tag_string = options.tags[0].tag_string if options.tags else None
        
        if not tag_string:
            return []
        
        try:
            # Simple direct query to get any chunks
            sql = """
            SELECT c.* FROM chunks c
            JOIN chunk_tags ct ON c.id = ct.chunk_id
            WHERE ct.tag = ?
            """
            params = [tag_string]
            
            # Add text search if possible (very basic)
            if options.query:
                search_terms = options.query.lower().split()
                if search_terms:
                    # Add basic text search condition
                    search_conditions = []
                    for term in search_terms[:3]:  # Limit to first 3 terms for simplicity
                        if len(term) > 3:  # Only use terms longer than 3 chars
                            search_conditions.append("LOWER(c.content) LIKE ?")
                            params.append(f"%{term}%")
                    
                    if search_conditions:
                        sql += " AND (" + " OR ".join(search_conditions) + ")"
            
            # Add directory filter if provided
            if options.filter_directory:
                sql += " AND c.path LIKE ?"
                params.append(f"{options.filter_directory}%")
            
            # Add limit
            sql += " ORDER BY c.id DESC LIMIT ?"
            params.append(options.n_retrieve)
            
            # Execute query
            self._debug(f"Executing fallback query with {len(params)} parameters")
            results = db.execute(sql, tuple(params))
            self._debug(f"Fallback query returned {len(results)} results")
            
            # Convert to Chunk objects
            chunks = []
            for row in results:
                chunks.append(Chunk(
                    filepath=row["path"],
                    start_line=row["start_line"],
                    end_line=row["end_line"],
                    content=row["content"],
                    digest=row["cache_key"],
                    index=row["chunk_index"]
                ))
            
            return chunks
        except Exception as e:
            self._debug(f"Error in fallback retrieval: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    async def run(self, options: RetrievalOptions) -> List[ContextItem]:
        """Run the retrieval pipeline.
        
        Args:
            options: Retrieval options
            
        Returns:
            List of context items
        """
        # Retrieve chunks
        chunks = await self.retrieve(options)
        
        # Convert to context items
        context_items = []
        for chunk in chunks:
            context_items.append(ContextItem.from_chunk(
                chunk,
                description=f"File: {chunk.filepath}, Lines: {chunk.start_line}-{chunk.end_line}"
            ))
        
        return context_items 