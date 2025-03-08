"""
Vector embedding indexer implementation for semantic search.
"""
from .chunk_index import ChunkCodebaseIndex
import json
import os
import uuid
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ..database import Database
from ..models import Chunk, IndexResultType, IndexTag, IndexingProgressUpdate, PathAndCacheKey, RefreshIndexResults
from .base import CodebaseIndex


class VectorCodebaseIndex(CodebaseIndex):
    """Indexer for vector embeddings using Sentence Transformers."""

    ARTIFACT_ID = "vector_embeddings"

    def __init__(
        self,
        db_path: str,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32
    ):
        """Initialize the vector indexer.

        Args:
            db_path: Path to the database file
            model_name: Name of the sentence transformer model to use
            batch_size: Batch size for generating embeddings
        """
        self.db = Database(db_path)
        self.model_name = model_name
        self.batch_size = batch_size
        self._create_tables()

        # Initialize the embedding model
        self.embedding_model = SentenceTransformer(model_name)

    def _create_tables(self):
        """Create tables for vector embeddings."""
        self.db.execute_write("""
        CREATE TABLE IF NOT EXISTS vector_embeddings (
            id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            cache_key TEXT NOT NULL,
            chunk_id INTEGER NOT NULL,
            embedding TEXT NOT NULL,
            tag TEXT NOT NULL,
            FOREIGN KEY (chunk_id) REFERENCES chunks (id)
        )
        """)

        self.db.execute_write("""
        CREATE INDEX IF NOT EXISTS idx_vector_embeddings_tag 
        ON vector_embeddings (tag)
        """)

        self.db.execute_write("""
        CREATE INDEX IF NOT EXISTS idx_vector_embeddings_path 
        ON vector_embeddings (path)
        """)

    @property
    def artifact_id(self) -> str:
        return f"{self.ARTIFACT_ID}::{self.model_name}"

    @property
    def relative_expected_time(self) -> float:
        return 5.0  # Embedding generation is relatively expensive

    def _encode_text(self, texts: List[str]) -> np.ndarray:
        """Encode texts into embeddings using the model.

        Args:
            texts: List of texts to encode

        Returns:
            Array of embeddings
        """
        return self.embedding_model.encode(texts, batch_size=self.batch_size)

    def _vector_to_string(self, vector: np.ndarray) -> str:
        """Convert a vector to a JSON string for storage."""
        return json.dumps(vector.tolist())

    def _string_to_vector(self, vector_str: str) -> np.ndarray:
        """Convert a JSON string back to a vector."""
        return np.array(json.loads(vector_str))

    def _cosine_similarity(self, query_vector: np.ndarray, vectors: List[np.ndarray]) -> List[float]:
        """Calculate cosine similarity between query and vectors."""
        # Normalize query vector
        query_norm = np.linalg.norm(query_vector)
        if query_norm > 0:
            query_vector = query_vector / query_norm

        similarities = []
        for vector in vectors:
            # Normalize vector
            vector_norm = np.linalg.norm(vector)
            if vector_norm > 0:
                vector = vector / vector_norm

            # Calculate similarity
            similarity = np.dot(query_vector, vector)
            similarities.append(float(similarity))

        return similarities

    async def update(
        self,
        tag: IndexTag,
        results: RefreshIndexResults,
        mark_complete: callable,
        repo_name: Optional[str] = None
    ) -> Generator[IndexingProgressUpdate, None, None]:
        """Update the vector embeddings index.

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

                if not chunks:
                    continue

                # Encode chunks in batches
                chunk_texts = [chunk["content"] for chunk in chunks]
                embeddings = self._encode_text(chunk_texts)

                # Store embeddings
                for j, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    embedding_id = str(uuid.uuid4())

                    self.db.execute_write(
                        """
                        INSERT INTO vector_embeddings (id, path, cache_key, chunk_id, embedding, tag)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            embedding_id,
                            item.path,
                            item.cache_key,
                            chunk["id"],
                            self._vector_to_string(embedding),
                            tag_string
                        )
                    )

                # Mark as complete
                await mark_complete([item], IndexResultType.COMPUTE)

                # Yield progress update
                yield IndexingProgressUpdate(
                    progress=i / len(results.compute),
                    desc=f"Embedding {os.path.basename(item.path)}",
                    status="indexing"
                )

            except Exception as e:
                print(f"Error embedding file {item.path}: {e}")

        # Process 'addTag' files
        for item in results.add_tag:
            try:
                # Get embeddings for this file from another tag
                existing_embeddings = self.db.execute(
                    """
                    SELECT e.* FROM vector_embeddings e
                    JOIN chunks c ON e.chunk_id = c.id
                    WHERE c.path = ? AND c.cache_key = ?
                    LIMIT 1
                    """,
                    (item.path, item.cache_key)
                )

                if not existing_embeddings:
                    continue

                # Get chunks for this file
                chunks = self.db.execute(
                    """
                    SELECT * FROM chunks WHERE path = ? AND cache_key = ?
                    """,
                    (item.path, item.cache_key)
                )

                # Copy embeddings with the new tag
                for chunk in chunks:
                    # Check if this chunk already has an embedding with this tag
                    existing = self.db.execute(
                        """
                        SELECT * FROM vector_embeddings 
                        WHERE chunk_id = ? AND tag = ?
                        """,
                        (chunk["id"], tag_string)
                    )

                    if existing:
                        continue

                    # Get embedding for this chunk from another tag
                    chunk_embedding = self.db.execute(
                        """
                        SELECT * FROM vector_embeddings
                        WHERE chunk_id = ?
                        LIMIT 1
                        """,
                        (chunk["id"],)
                    )

                    if not chunk_embedding:
                        continue

                    # Create new embedding with this tag
                    embedding_id = str(uuid.uuid4())
                    self.db.execute_write(
                        """
                        INSERT INTO vector_embeddings (id, path, cache_key, chunk_id, embedding, tag)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            embedding_id,
                            item.path,
                            item.cache_key,
                            chunk["id"],
                            chunk_embedding[0]["embedding"],
                            tag_string
                        )
                    )

                await mark_complete([item], IndexResultType.ADD_TAG)

            except Exception as e:
                print(f"Error adding tag to file {item.path}: {e}")

        # Process 'removeTag' files
        for item in results.remove_tag:
            try:
                # Delete embeddings with this tag
                self.db.execute_write(
                    """
                    DELETE FROM vector_embeddings 
                    WHERE path = ? AND cache_key = ? AND tag = ?
                    """,
                    (item.path, item.cache_key, tag_string)
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

                for chunk_id_row in chunk_ids:
                    chunk_id = chunk_id_row["id"]

                    # Check if there are other tags for this chunk
                    tag_count = self.db.execute(
                        """
                        SELECT COUNT(*) as count FROM vector_embeddings 
                        WHERE chunk_id = ?
                        """,
                        (chunk_id,)
                    )

                    # If only one tag (this one), delete all embeddings
                    if tag_count[0]["count"] <= 1:
                        self.db.execute_write(
                            """
                            DELETE FROM vector_embeddings WHERE chunk_id = ?
                            """,
                            (chunk_id,)
                        )
                    else:
                        # Otherwise, just delete this tag
                        self.db.execute_write(
                            """
                            DELETE FROM vector_embeddings 
                            WHERE chunk_id = ? AND tag = ?
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
        """Retrieve chunks from the index using vector similarity.

        Args:
            query: Text query to search for
            tags: Tags to search in
            n: Maximum number of results
            directory: Optional directory filter
            filter_paths: Optional file paths to filter by

        Returns:
            List of matching chunks
        """
        # Encode query
        query_embedding = self._encode_text([query])[0]

        # Build query
        tag_strings = [tag.tag_string for tag in tags]
        tag_placeholders = ",".join(["?"] * len(tag_strings))

        sql = f"""
        SELECT e.chunk_id, e.embedding, c.* 
        FROM vector_embeddings e
        JOIN chunks c ON e.chunk_id = c.id
        WHERE e.tag IN ({tag_placeholders})
        """

        params = list(tag_strings)

        # Add directory filter if provided
        if directory:
            sql += " AND e.path LIKE ?"
            params.append(f"{directory}%")

        # Add path filter if provided
        if filter_paths and len(filter_paths) > 0:
            path_placeholders = ",".join(["?"] * len(filter_paths))
            sql += f" AND e.path IN ({path_placeholders})"
            params.extend(filter_paths)

        # Execute query
        results = self.db.execute(sql, tuple(params))

        # Calculate similarities
        embeddings = [self._string_to_vector(r["embedding"]) for r in results]
        similarities = self._cosine_similarity(query_embedding, embeddings)

        # Sort by similarity
        chunk_similarity_pairs = list(zip(results, similarities))
        chunk_similarity_pairs.sort(key=lambda x: x[1], reverse=True)

        # Take top N
        top_results = chunk_similarity_pairs[:n]

        # Convert to Chunk objects
        chunks = []
        for db_chunk, similarity in top_results:
            chunks.append(Chunk(
                filepath=db_chunk["path"],
                start_line=db_chunk["start_line"],
                end_line=db_chunk["end_line"],
                content=db_chunk["content"],
                digest=db_chunk["cache_key"],
                index=db_chunk["index"]
            ))

        return chunks


# Imported here to avoid circular imports
