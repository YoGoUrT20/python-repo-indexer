"""
Main repository indexer class.
"""
import asyncio
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from tqdm import tqdm

from .database import Database
from .indexers import ChunkCodebaseIndex, CodebaseIndex, FullTextSearchCodebaseIndex, VectorCodebaseIndex
from .models import (
    ContextItem,
    IndexResultType,
    IndexTag,
    IndexingProgressUpdate,
    PathAndCacheKey,
    RefreshIndexResults
)
from .retrieval import RetrievalOptions, StandardRetrievalPipeline
from .utils import get_file_hash, get_file_modification_time, get_repo_branch, walk_directory


class RepoIndexer:
    """Main repository indexer class.

    This class orchestrates the entire indexing and searching process.
    """

    def __init__(
        self,
        repo_path: Union[str, Path],
        db_path: Optional[Union[str, Path]] = None,
        ignore_patterns: Optional[List[str]] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        files_per_batch: int = 50,
        debug: bool = False
    ):
        """Initialize the repository indexer.

        Args:
            repo_path: Path to the repository
            db_path: Path to the database file (default: <repo_path>/.repo_index/index.db)
            ignore_patterns: Patterns of files to ignore
            embedding_model: Name of the embedding model to use
            files_per_batch: Number of files to process in a batch
            debug: Enable debug logging
        """
        self.repo_path = Path(repo_path)
        self.debug = debug

        if not self.repo_path.exists():
            raise ValueError(f"Repository path does not exist: {self.repo_path}")

        # Set up database path
        if db_path is None:
            db_path = self.repo_path / ".repo_index" / "index.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self.db = Database(self.db_path)

        # Set up indexers
        self.indexers: List[CodebaseIndex] = []
        self.files_per_batch = files_per_batch
        self.ignore_patterns = ignore_patterns or []
        self.embedding_model = embedding_model

    def _log(self, message: str) -> None:
        """Log a debug message if debug is enabled."""
        if self.debug:
            print(f"[DEBUG] {message}")

    def _get_indexers(self) -> List[CodebaseIndex]:
        """Get the list of indexers to use."""
        if not self.indexers:
            # Initialize indexers - ChunkCodebaseIndex must come first
            self.indexers = [
                ChunkCodebaseIndex(str(self.db_path)),
                FullTextSearchCodebaseIndex(str(self.db_path)),
                VectorCodebaseIndex(str(self.db_path), model_name=self.embedding_model)
            ]
        return self.indexers

    async def _mark_complete(self, items: List[PathAndCacheKey], result_type: IndexResultType) -> None:
        """Mark files as complete in the index."""
        # This is a callback that gets passed to indexers
        # No need to implement anything here as we handle status updates in the database
        # during the indexing process
        pass

    async def _compute_refresh_results(self) -> RefreshIndexResults:
        """Compute what files need to be indexed, retagged, or removed."""
        # Get branch
        branch = get_repo_branch(self.repo_path)
        self._log(f"Current branch: {branch}")

        # Get all files in the repository
        repo_files = set()
        for file_path in walk_directory(self.repo_path, self.ignore_patterns):
            relative_path = file_path.relative_to(self.repo_path)
            repo_files.add(str(relative_path))

        self._log(f"Found {len(repo_files)} files in repository")

        # Get all files in the catalog
        with self.db.get_session() as session:
            catalog_results = self.db.execute("SELECT path FROM index_catalog")
            catalog_files = {row["path"] for row in catalog_results}

        self._log(f"Found {len(catalog_files)} files in catalog")

        # Files to add, update, or remove
        compute = []
        add_tag = []
        remove_tag = []
        delete = []

        # Files to add or update
        for file_path in repo_files:
            abs_path = self.repo_path / file_path

            try:
                # Get file hash and modification time
                file_hash = get_file_hash(abs_path)
                mod_time = get_file_modification_time(abs_path)

                # Check if file exists in catalog
                catalog_entry = self.db.execute(
                    "SELECT * FROM index_catalog WHERE path = ?",
                    (file_path,)
                )

                if not catalog_entry:
                    # New file, add to compute
                    compute.append(PathAndCacheKey(
                        path=str(abs_path),
                        cache_key=file_hash
                    ))

                    # Add to catalog
                    self.db.execute_write(
                        """
                        INSERT INTO index_catalog (path, cache_key, last_modified, last_indexed)
                        VALUES (?, ?, ?, ?)
                        """,
                        (file_path, file_hash, mod_time, time.time())
                    )

                elif catalog_entry[0]["cache_key"] != file_hash or catalog_entry[0]["last_modified"] < mod_time:
                    # File changed, update
                    compute.append(PathAndCacheKey(
                        path=str(abs_path),
                        cache_key=file_hash
                    ))

                    # Update catalog
                    self.db.execute_write(
                        """
                        UPDATE index_catalog 
                        SET cache_key = ?, last_modified = ?, last_indexed = ?
                        WHERE path = ?
                        """,
                        (file_hash, mod_time, time.time(), file_path)
                    )

                else:
                    # File exists and hasn't changed, check if it's tagged for this branch
                    tag_string = f"{str(self.repo_path)}::{branch}::chunks"
                    tag_entry = self.db.execute(
                        """
                        SELECT * FROM index_tags 
                        WHERE path = ? AND cache_key = ? AND tag = ?
                        """,
                        (file_path, file_hash, tag_string)
                    )

                    if not tag_entry:
                        # File exists but needs to be tagged for this branch
                        add_tag.append(PathAndCacheKey(
                            path=str(abs_path),
                            cache_key=file_hash
                        ))

                        # Add tag
                        self.db.execute_write(
                            """
                            INSERT INTO index_tags (path, cache_key, tag)
                            VALUES (?, ?, ?)
                            """,
                            (file_path, file_hash, tag_string)
                        )

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

        # Files to remove
        for file_path in catalog_files - repo_files:
            catalog_entry = self.db.execute(
                "SELECT * FROM index_catalog WHERE path = ?",
                (file_path,)
            )

            if catalog_entry:
                abs_path = self.repo_path / file_path
                cache_key = catalog_entry[0]["cache_key"]

                # Add to delete list
                delete.append(PathAndCacheKey(
                    path=str(abs_path),
                    cache_key=cache_key
                ))

                # Remove from catalog
                self.db.execute_write(
                    "DELETE FROM index_catalog WHERE path = ?",
                    (file_path,)
                )

        self._log(f"Files to compute: {len(compute)}")
        self._log(f"Files to add tag: {len(add_tag)}")
        self._log(f"Files to remove tag: {len(remove_tag)}")
        self._log(f"Files to delete: {len(delete)}")

        return RefreshIndexResults(
            compute=compute,
            add_tag=add_tag,
            remove_tag=remove_tag,
            delete=delete
        )

    def _batch_results(self, results: RefreshIndexResults) -> List[RefreshIndexResults]:
        """Split refresh results into batches."""
        batches = []

        for i in range(0, max(len(results.compute), 1), self.files_per_batch):
            batch = RefreshIndexResults(
                compute=results.compute[i:i+self.files_per_batch],
                add_tag=results.add_tag[i:i+self.files_per_batch] if i < len(results.add_tag) else [],
                remove_tag=results.remove_tag[i:i+self.files_per_batch] if i < len(results.remove_tag) else [],
                delete=results.delete[i:i+self.files_per_batch] if i < len(results.delete) else []
            )
            batches.append(batch)

        return batches if batches else [RefreshIndexResults(compute=[], add_tag=[], remove_tag=[], delete=[])]

    async def index(self, force_reindex: bool = False) -> None:
        """Index the repository.

        Args:
            force_reindex: If True, reindex all files even if they haven't changed
        """
        # Clear index if forcing reindex
        if force_reindex:
            await self.clear_index()

        # Compute what needs to be indexed
        print("Computing files to index...")
        results = await self._compute_refresh_results()

        # Get indexers
        indexers = self._get_indexers()

        # Get tag
        branch = get_repo_branch(self.repo_path)

        # Create tag for each indexer
        tags = {
            indexer.artifact_id: IndexTag(
                directory=str(self.repo_path),
                branch=branch,
                artifact_id=indexer.artifact_id
            )
            for indexer in indexers
        }

        # Print summary
        print(f"Found {len(results.compute)} files to index, {len(results.add_tag)} to tag, "
              f"{len(results.remove_tag)} to untag, and {len(results.delete)} to delete.")

        if not any([results.compute, results.add_tag, results.remove_tag, results.delete]) and not force_reindex:
            print("Nothing to do.")
            if force_reindex:
                print("Forcing reindex of all files...")
                # If we're forcing a reindex, we want to force compute on all repo files
                repo_files = []
                for file_path in walk_directory(self.repo_path, self.ignore_patterns):
                    abs_path = str(file_path)
                    file_hash = get_file_hash(file_path)
                    repo_files.append(PathAndCacheKey(path=abs_path, cache_key=file_hash))

                results = RefreshIndexResults(
                    compute=repo_files,
                    add_tag=[],
                    remove_tag=[],
                    delete=[]
                )
                print(f"Force reindexing {len(repo_files)} files")
            else:
                # Verify database has content
                chunk_count = self.db.execute("SELECT COUNT(*) as count FROM chunks")
                if chunk_count and chunk_count[0]["count"] > 0:
                    print(f"Database already contains {chunk_count[0]['count']} chunks")
                    return
                else:
                    print("Database appears empty. Forcing reindex...")
                    return await self.index(force_reindex=True)

        # Process in batches
        batches = self._batch_results(results)

        # Process each batch
        for i, batch in enumerate(batches):
            print(f"Processing batch {i+1}/{len(batches)}...")

            # Process each indexer
            for indexer in indexers:
                tag = tags[indexer.artifact_id]

                progress_bar = tqdm(
                    total=100,
                    desc=f"Indexing with {indexer.artifact_id}",
                    unit="%"
                )

                # Update index
                last_progress = 0
                async for progress in indexer.update(tag, batch, self._mark_complete, None):
                    # Update progress bar
                    current_progress = int(progress.progress * 100)
                    if current_progress > last_progress:
                        progress_bar.update(current_progress - last_progress)
                        last_progress = current_progress

                progress_bar.close()

        # Verify the indexing was successful
        chunk_count = self.db.execute("SELECT COUNT(*) as count FROM chunks")
        fts_count = self.db.execute("SELECT COUNT(*) as count FROM fts")
        vector_count = self.db.execute("SELECT COUNT(*) as count FROM vector_embeddings")

        print("Indexing complete.")
        print(f"Database contains:")
        print(f" - {chunk_count[0]['count']} chunks")
        print(f" - {fts_count[0]['count'] if fts_count else 0} full-text search entries")
        print(f" - {vector_count[0]['count'] if vector_count else 0} vector embeddings")

    async def clear_index(self) -> None:
        """Clear the index."""
        # Get tables
        tables = self.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )

        # Drop all tables
        for table in tables:
            self.db.execute_write(f"DROP TABLE IF EXISTS {table['name']}")

        # Recreate database schema
        self.db = Database(self.db_path)

        print("Index cleared.")

    async def search(
        self,
        query: str,
        n: int = 10,
        directory: Optional[str] = None,
        include_embeddings: bool = True
    ) -> List[ContextItem]:
        """Search the repository for relevant context.

        Args:
            query: Search query
            n: Maximum number of results
            directory: Optional directory to limit search to
            include_embeddings: Whether to include vector embeddings in search

        Returns:
            List of context items
        """
        print(f"Searching for: '{query}'...")

        # Verify database has content
        chunk_count = self.db.execute("SELECT COUNT(*) as count FROM chunks")
        if not chunk_count or chunk_count[0]["count"] == 0:
            print("Database is empty. Please index the repository first.")
            return []

        self._log(f"Database contains {chunk_count[0]['count']} chunks")

        # Get branch
        branch = get_repo_branch(self.repo_path)

        # Create tag
        tag = IndexTag(
            directory=str(self.repo_path),
            branch=branch,
            artifact_id="chunks"  # Use chunks as base tag
        )

        self._log(f"Using tag: {tag.tag_string}")

        # Create retrieval options
        options = RetrievalOptions(
            tags=[tag],
            query=query,
            n_retrieve=n*2,
            n_final=n,
            include_embeddings=include_embeddings,
            filter_directory=directory
        )

        # Create retrieval pipeline
        pipeline = StandardRetrievalPipeline(
            db_path=str(self.db_path),
            vector_model_name=self.embedding_model
        )

        # Run retrieval
        try:
            results = await pipeline.run(options)
            self._log(f"Found {len(results)} results")
            return results
        except Exception as e:
            print(f"Error during search: {e}")
            import traceback
            traceback.print_exc()
            return []
