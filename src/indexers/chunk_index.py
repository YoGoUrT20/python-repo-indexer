"""
Chunker indexer implementation.
"""
import asyncio
import os
from pathlib import Path
from typing import Generator, List, Optional

from tqdm import tqdm

from ..chunking import chunk_file
from ..database import Database
from ..models import Chunk, IndexResultType, IndexTag, IndexingProgressUpdate, PathAndCacheKey, RefreshIndexResults
from ..utils import get_file_hash
from .base import CodebaseIndex


class ChunkCodebaseIndex(CodebaseIndex):
    """Index that breaks files into chunks for further processing."""

    ARTIFACT_ID = "chunks"

    def __init__(self, db_path: str, max_chunk_size: int = 100):
        """Initialize the chunk indexer.

        Args:
            db_path: Path to the database file
            max_chunk_size: Maximum number of lines per chunk
        """
        self.db = Database(db_path)
        self.max_chunk_size = max_chunk_size

    @property
    def artifact_id(self) -> str:
        return self.ARTIFACT_ID

    @property
    def relative_expected_time(self) -> float:
        return 1.0

    async def update(
        self,
        tag: IndexTag,
        results: RefreshIndexResults,
        mark_complete: callable,
        repo_name: Optional[str] = None
    ) -> Generator[IndexingProgressUpdate, None, None]:
        """Update the chunks index.

        Args:
            tag: Tag for the branch and directory
            results: Results of the refresh operation
            mark_complete: Callback to mark files as complete
            repo_name: Optional name of the repository

        Yields:
            Progress updates
        """
        tag_string = tag.tag_string

        # Process 'compute' files
        for i, item in enumerate(results.compute):
            # Chunk the file
            try:
                chunks = chunk_file(item.path, self.max_chunk_size)

                # Store chunks in database
                with self.db.get_session() as session:
                    # Insert chunks
                    for chunk in chunks:
                        # Convert to database model
                        db_chunk = {
                            "path": chunk.filepath,
                            "cache_key": chunk.digest,
                            "chunk_index": chunk.index,
                            "start_line": chunk.start_line,
                            "end_line": chunk.end_line,
                            "content": chunk.content
                        }

                        # Insert chunk and get ID
                        result = self.db.execute_write(
                            """
                            INSERT INTO chunks (path, cache_key, chunk_index, start_line, end_line, content)
                            VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (
                                db_chunk["path"],
                                db_chunk["cache_key"],
                                db_chunk["chunk_index"],
                                db_chunk["start_line"],
                                db_chunk["end_line"],
                                db_chunk["content"]
                            )
                        )
                        chunk_id = result.lastrowid

                        # Add tag for this chunk
                        self.db.execute_write(
                            """
                            INSERT INTO chunk_tags (chunk_id, tag)
                            VALUES (?, ?)
                            """,
                            (chunk_id, tag_string)
                        )

                # Mark as complete
                await mark_complete([item], IndexResultType.COMPUTE)

                # Yield progress update
                yield IndexingProgressUpdate(
                    progress=i / len(results.compute) if results.compute else 1.0,
                    desc=f"Chunking {os.path.basename(item.path)}",
                    status="indexing"
                )

            except Exception as e:
                print(f"Error chunking file {item.path}: {e}")
                import traceback
                traceback.print_exc()

        # Process 'addTag' files
        for item in results.add_tag:
            try:
                # Get chunk IDs for this file
                chunk_ids = self.db.execute(
                    """
                    SELECT id FROM chunks WHERE path = ? AND cache_key = ?
                    """,
                    (item.path, item.cache_key)
                )

                # Add tag for each chunk
                for chunk_id_row in chunk_ids:
                    self.db.execute_write(
                        """
                        INSERT OR IGNORE INTO chunk_tags (chunk_id, tag)
                        VALUES (?, ?)
                        """,
                        (chunk_id_row["id"], tag_string)
                    )

                await mark_complete([item], IndexResultType.ADD_TAG)

            except Exception as e:
                print(f"Error adding tag to file {item.path}: {e}")

        # Process 'removeTag' files
        for item in results.remove_tag:
            try:
                # Get chunk IDs for this file
                chunk_ids = self.db.execute(
                    """
                    SELECT id FROM chunks WHERE path = ? AND cache_key = ?
                    """,
                    (item.path, item.cache_key)
                )

                # Remove tag for each chunk
                for chunk_id_row in chunk_ids:
                    self.db.execute_write(
                        """
                        DELETE FROM chunk_tags WHERE chunk_id = ? AND tag = ?
                        """,
                        (chunk_id_row["id"], tag_string)
                    )

                await mark_complete([item], IndexResultType.REMOVE_TAG)

            except Exception as e:
                print(f"Error removing tag from file {item.path}: {e}")

        # Process 'delete' files
        for item in results.delete:
            try:
                # Get chunk IDs for this file
                chunk_ids = self.db.execute(
                    """
                    SELECT id FROM chunks WHERE path = ? AND cache_key = ?
                    """,
                    (item.path, item.cache_key)
                )

                # For each chunk, check if it has other tags
                for chunk_id_row in chunk_ids:
                    chunk_id = chunk_id_row["id"]
                    tag_count = self.db.execute(
                        """
                        SELECT COUNT(*) as count FROM chunk_tags WHERE chunk_id = ?
                        """,
                        (chunk_id,)
                    )

                    # If no other tags, delete the chunk
                    if tag_count[0]["count"] <= 1:
                        self.db.execute_write(
                            """
                            DELETE FROM chunks WHERE id = ?
                            """,
                            (chunk_id,)
                        )

                    # Delete the tag
                    self.db.execute_write(
                        """
                        DELETE FROM chunk_tags WHERE chunk_id = ? AND tag = ?
                        """,
                        (chunk_id, tag_string)
                    )

                await mark_complete([item], IndexResultType.DELETE)

            except Exception as e:
                print(f"Error deleting file {item.path}: {e}")

    async def retrieve(
        self,
        query: str,
        tags: List[IndexTag],
        n: int = 10,
        directory: Optional[str] = None,
        filter_paths: Optional[List[str]] = None
    ) -> List[Chunk]:
        """Retrieve chunks from the index.
        Note: This method returns all chunks matching the criteria since this
        index doesn't perform any semantic search itself.

        Args:
            query: Query text (not used in this index)
            tags: Tags to search in
            n: Maximum number of results to return
            directory: Optional directory filter
            filter_paths: Optional paths to filter by

        Returns:
            List of chunks
        """
        tag_strings = [tag.tag_string for tag in tags]

        # Build the query
        sql_query = """
        SELECT DISTINCT c.* 
        FROM chunks c
        JOIN chunk_tags ct ON c.id = ct.chunk_id
        WHERE ct.tag IN ({})
        """.format(','.join(['?'] * len(tag_strings)))

        params = list(tag_strings)

        # Add directory filter if provided
        if directory:
            sql_query += " AND c.path LIKE ?"
            params.append(f"{directory}%")

        # Add path filter if provided
        if filter_paths and len(filter_paths) > 0:
            path_placeholders = ','.join(['?'] * len(filter_paths))
            sql_query += f" AND c.path IN ({path_placeholders})"
            params.extend(filter_paths)

        # Add limit
        sql_query += " LIMIT ?"
        params.append(n)

        # Execute query
        db_chunks = self.db.execute(sql_query, tuple(params))

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

        return chunks
