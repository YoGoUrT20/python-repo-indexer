"""
Full-text search indexer implementation.
"""
from .chunk_index import ChunkCodebaseIndex
import os
import re
from typing import Generator, List, Optional, Tuple

from ..database import Database
from ..models import Chunk, IndexResultType, IndexTag, IndexingProgressUpdate, PathAndCacheKey, RefreshIndexResults
from .base import CodebaseIndex


class FullTextSearchCodebaseIndex(CodebaseIndex):
    """Index for full-text search using SQLite FTS5."""

    ARTIFACT_ID = "sqliteFts"

    def __init__(self, db_path: str, path_weight_multiplier: float = 10.0, bm25_threshold: float = 10.0):
        """Initialize the full-text search indexer.

        Args:
            db_path: Path to the database file
            path_weight_multiplier: Weight multiplier for paths (vs. content)
            bm25_threshold: Threshold for BM25 ranking
        """
        self.db = Database(db_path)
        self.path_weight_multiplier = path_weight_multiplier
        self.bm25_threshold = bm25_threshold
        self.debug = False

    def _debug(self, message: str) -> None:
        """Print debug message if debug is enabled."""
        if self.debug:
            print(f"[DEBUG:FTS] {message}")

    @property
    def artifact_id(self) -> str:
        return self.ARTIFACT_ID

    @property
    def relative_expected_time(self) -> float:
        return 0.2

    async def update(
        self,
        tag: IndexTag,
        results: RefreshIndexResults,
        mark_complete: callable,
        repo_name: Optional[str] = None
    ) -> Generator[IndexingProgressUpdate, None, None]:
        """Update the full-text search index.

        Args:
            tag: Tag for the branch and directory
            results: Results of the refresh operation
            mark_complete: Callback to mark files as complete
            repo_name: Optional repository name

        Yields:
            Progress updates
        """
        tag_string = tag.tag_string

        # Process 'compute' files
        for i, item in enumerate(results.compute):
            try:
                # Get chunks for this file
                chunks = self.db.execute(
                    """
                    SELECT * FROM chunks WHERE path = ? AND cache_key = ?
                    """,
                    (item.path, item.cache_key)
                )

                # Insert each chunk into FTS
                for chunk in chunks:
                    # Insert into FTS
                    cursor = self.db.execute_write(
                        """
                        INSERT INTO fts (path, content) VALUES (?, ?)
                        """,
                        (item.path, chunk["content"])
                    )
                    last_id = cursor.lastrowid

                    # Insert metadata
                    self.db.execute_write(
                        """
                        INSERT INTO fts_metadata (id, path, cache_key, chunk_id)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(id) DO UPDATE SET
                        path = excluded.path,
                        cache_key = excluded.cache_key,
                        chunk_id = excluded.chunk_id
                        """,
                        (last_id, item.path, item.cache_key, chunk["id"])
                    )

                # Mark as complete
                await mark_complete([item], IndexResultType.COMPUTE)

                # Yield progress update
                yield IndexingProgressUpdate(
                    progress=i / len(results.compute) if results.compute else 1.0,
                    desc=f"Indexing {os.path.basename(item.path)}",
                    status="indexing"
                )

            except Exception as e:
                print(f"Error indexing file {item.path}: {e}")

        # Process 'addTag' files
        for item in results.add_tag:
            await mark_complete([item], IndexResultType.ADD_TAG)

        # Process 'removeTag' files
        for item in results.remove_tag:
            await mark_complete([item], IndexResultType.REMOVE_TAG)

        # Process 'delete' files
        for item in results.delete:
            try:
                # Delete from FTS
                self.db.execute_write(
                    """
                    DELETE FROM fts WHERE rowid IN (
                        SELECT id FROM fts_metadata WHERE path = ? AND cache_key = ?
                    )
                    """,
                    (item.path, item.cache_key)
                )

                # Delete metadata
                self.db.execute_write(
                    """
                    DELETE FROM fts_metadata WHERE path = ? AND cache_key = ?
                    """,
                    (item.path, item.cache_key)
                )

                await mark_complete([item], IndexResultType.DELETE)

            except Exception as e:
                print(f"Error deleting file {item.path} from FTS: {e}")

    def _build_tag_filter(self, tags: List[IndexTag]) -> Tuple[str, List[str]]:
        """Build SQL filter for tags."""
        tag_strings = [f"{ChunkCodebaseIndex.ARTIFACT_ID}::{tag.directory}::{tag.branch}" for tag in tags]
        return f"AND chunk_tags.tag IN ({','.join(['?'] * len(tag_strings))})", tag_strings

    def _build_path_filter(self, filter_paths: Optional[List[str]]) -> Tuple[str, List[str]]:
        """Build SQL filter for paths."""
        if not filter_paths or len(filter_paths) == 0:
            return "", []

        return f"AND fts_metadata.path IN ({','.join(['?'] * len(filter_paths))})", filter_paths

    def _tokenize_query(self, query: str) -> List[str]:
        """Break a query into individual tokens.

        Args:
            query: Input query string

        Returns:
            List of tokenized terms
        """
        # Remove non-alphanumeric characters and split by whitespace
        tokens = re.sub(r'[^\w\s]', ' ', query).lower().split()

        # Filter out common words and short tokens
        stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'for', 'in', 'to', 'is', 'of', 'at'}
        tokens = [token for token in tokens if token not in stopwords and len(token) > 2]

        return tokens

    def _sanitize_query(self, query: str) -> str:
        """Sanitize the query for FTS5.

        Args:
            query: Original query

        Returns:
            Sanitized query
        """
        # Tokenize the query
        tokens = self._tokenize_query(query)

        # If no valid tokens, return a simple match-all query
        if not tokens:
            return "*"

        # Join tokens with simple AND
        sanitized = " AND ".join(tokens)

        # Ensure the query is well-formed
        sanitized = sanitized.replace("?", "").replace("*", "").replace('"', "")

        return sanitized

    def _build_fallback_query(self, query: str) -> str:
        """Build a fallback query string using LIKE conditions.

        Args:
            query: Original query string

        Returns:
            SQL query fragment for LIKE conditions
        """
        tokens = self._tokenize_query(query)
        if not tokens:
            return ""

        # Build LIKE conditions for each token
        conditions = []
        for token in tokens:
            conditions.append(f"content LIKE '%{token}%'")

        return " OR ".join(conditions)

    def _build_query(self,
                     query: str,
                     tags: List[IndexTag],
                     filter_paths: Optional[List[str]]) -> Tuple[str, List]:
        """Build SQL query for FTS retrieval."""
        tag_filter, tag_params = self._build_tag_filter(tags)
        path_filter, path_params = self._build_path_filter(filter_paths)

        # Try sanitized FTS query first
        sanitized_query = self._sanitize_query(query)
        self._debug(f"Sanitized query: '{sanitized_query}'")

        # Build the SQL query with proper FTS5 syntax
        sql = f"""
        SELECT fts_metadata.chunk_id, fts_metadata.path, fts.content, rank
        FROM fts
        JOIN fts_metadata ON fts.rowid = fts_metadata.id
        JOIN chunk_tags ON fts_metadata.chunk_id = chunk_tags.chunk_id
        WHERE fts MATCH ?
        {tag_filter}
        {path_filter}
        ORDER BY bm25(fts, {self.path_weight_multiplier})
        LIMIT ?
        """

        params = [sanitized_query] + tag_params + path_params
        return sql, params

    def _build_fallback_sql_query(self,
                                  query: str,
                                  tags: List[IndexTag],
                                  filter_paths: Optional[List[str]]) -> Tuple[str, List]:
        """Build a fallback SQL query that doesn't use FTS5 MATCH.

        Args:
            query: Search query
            tags: Tags to filter by
            filter_paths: Paths to filter by

        Returns:
            SQL query and parameters
        """
        tag_filter, tag_params = self._build_tag_filter(tags)
        path_filter, path_params = self._build_path_filter(filter_paths)

        # Use direct table access instead of the FTS virtual table
        sql = f"""
        SELECT fts_metadata.chunk_id, fts_metadata.path, fts.content
        FROM fts
        JOIN fts_metadata ON fts.rowid = fts_metadata.id
        JOIN chunk_tags ON fts_metadata.chunk_id = chunk_tags.chunk_id
        WHERE (fts.content LIKE ? OR fts.path LIKE ?)
        {tag_filter}
        {path_filter}
        LIMIT ?
        """

        # Use simple text matching with LIKE
        # For simplicity, we just use the raw query with % wildcards
        search_term = f"%{query}%"
        params = [search_term, search_term] + tag_params + path_params

        return sql, params

    async def retrieve(
        self,
        query: str,
        tags: List[IndexTag],
        n: int = 10,
        directory: Optional[str] = None,
        filter_paths: Optional[List[str]] = None
    ) -> List[Chunk]:
        """Retrieve chunks from the index using full-text search.

        Args:
            query: Text query to search for
            tags: Tags to search in
            n: Maximum number of results
            directory: Optional directory filter
            filter_paths: Optional file paths to filter by

        Returns:
            List of matching chunks
        """
        self.debug = True  # Enable debugging

        # Apply directory filter if provided
        if directory and filter_paths is None:
            filter_paths = [directory]

        self._debug(f"Searching for: '{query}'")
        self._debug(f"Tags: {[tag.tag_string for tag in tags]}")

        # Try standard FTS query first
        try:
            # Build query
            sql, params = self._build_query(query, tags, filter_paths)
            params.append(n)

            # Execute query
            self._debug(f"Executing FTS query: {sql}")
            self._debug(f"With parameters: {params}")
            results = self.db.execute(sql, tuple(params))
            self._debug(f"FTS query returned {len(results)} results")

            # If we got results, process them
            if results:
                # Filter by BM25 score
                results = [r for r in results if r.get("rank", 0) <= self.bm25_threshold]

                # Get the actual chunks
                chunk_ids = [r["chunk_id"] for r in results]
                if chunk_ids:
                    chunks_sql = f"""
                    SELECT * FROM chunks 
                    WHERE id IN ({','.join(['?'] * len(chunk_ids))})
                    """

                    db_chunks = self.db.execute(chunks_sql, tuple(chunk_ids))

                    # Convert to Chunk objects
                    chunks = []
                    for db_chunk in db_chunks:
                        chunks.append(Chunk(
                            filepath=db_chunk["path"],
                            start_line=db_chunk["start_line"],
                            end_line=db_chunk["end_line"],
                            content=db_chunk["content"],
                            digest=db_chunk["cache_key"],
                            index=db_chunk["chunk_index"]
                        ))

                    self._debug(f"Returning {len(chunks)} chunks from FTS search")
                    return chunks
        except Exception as e:
            self._debug(f"Error in FTS query: {e}")
            import traceback
            traceback.print_exc()

        # If we get here, either there was an error or no results
        # Try fallback query
        self._debug("FTS query failed or returned no results, trying fallback query")
        try:
            # Build fallback query
            sql, params = self._build_fallback_sql_query(query, tags, filter_paths)
            params.append(n)

            # Execute query
            self._debug(f"Executing fallback query: {sql}")
            self._debug(f"With parameters: {params}")
            results = self.db.execute(sql, tuple(params))
            self._debug(f"Fallback query returned {len(results)} results")

            if not results:
                # Last resort: just return any chunks matching the tag
                self._debug("Fallback query returned no results, retrieving any chunks with matching tag")
                tag_string = f"{ChunkCodebaseIndex.ARTIFACT_ID}::{tags[0].directory}::{tags[0].branch}" if tags else ""
                if tag_string:
                    any_chunks_sql = """
                    SELECT c.* FROM chunks c
                    JOIN chunk_tags ct ON c.id = ct.chunk_id
                    WHERE ct.tag = ?
                    LIMIT ?
                    """
                    any_chunks = self.db.execute(any_chunks_sql, (tag_string, n))

                    if any_chunks:
                        chunks = []
                        for db_chunk in any_chunks:
                            chunks.append(Chunk(
                                filepath=db_chunk["path"],
                                start_line=db_chunk["start_line"],
                                end_line=db_chunk["end_line"],
                                content=db_chunk["content"],
                                digest=db_chunk["cache_key"],
                                index=db_chunk["chunk_index"]
                            ))

                        self._debug(f"Last resort query returned {len(chunks)} chunks")
                        return chunks

                self._debug("All queries failed, returning empty result")
                return []

            # Get the actual chunks from chunk IDs
            chunk_ids = [r["chunk_id"] for r in results]
            chunks_sql = f"""
            SELECT * FROM chunks 
            WHERE id IN ({','.join(['?'] * len(chunk_ids))})
            """

            db_chunks = self.db.execute(chunks_sql, tuple(chunk_ids))

            # Convert to Chunk objects
            chunks = []
            for db_chunk in db_chunks:
                chunks.append(Chunk(
                    filepath=db_chunk["path"],
                    start_line=db_chunk["start_line"],
                    end_line=db_chunk["end_line"],
                    content=db_chunk["content"],
                    digest=db_chunk["cache_key"],
                    index=db_chunk["chunk_index"]
                ))

            self._debug(f"Returning {len(chunks)} chunks from fallback search")
            return chunks

        except Exception as e:
            self._debug(f"Error in fallback query: {e}")
            import traceback
            traceback.print_exc()
            return []


# Imported here to avoid circular imports
