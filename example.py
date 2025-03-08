"""
Example usage of the repository indexer.
"""
import asyncio
import os
import sys
import time
from pathlib import Path

from src.repo_indexer import RepoIndexer


async def main():
    repo_path = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))

    # Convert to absolute path
    repo_path = Path(repo_path).resolve()
    print(f"Using repository at: {repo_path}")

    # Check if path exists
    if not repo_path.exists():
        print(f"Error: Repository path {repo_path} does not exist.")
        return

    # Create the indexer with debug mode enabled
    print("Initializing indexer...")
    indexer = RepoIndexer(
        repo_path=repo_path,
        embedding_model="all-MiniLM-L6-v2",
        files_per_batch=50,
        debug=True
    )

    # Index the repository with force_reindex=True to ensure we have data
    print("Indexing repository...")
    start_time = time.time()
    await indexer.index(force_reindex=True)
    end_time = time.time()
    print(f"Indexing completed in {end_time - start_time:.2f} seconds.")

    print("\nRunning example searches...")

    # List of queries that should match current repo
    queries = [
        "database connection",
        "chunking file",
        "sqlite",
        "index tag",
        "vector embeddings",
        "retrieval pipeline",
        "search context",
        "full text search",
    ]

    # Run searches
    successful_searches = 0
    for query in queries:
        print(f"\n\nSearching for: '{query}'")
        print("=" * 80)

        start_time = time.time()
        results = await indexer.search(query, n=3)
        end_time = time.time()

        if not results:
            print("No results found.")
            continue

        successful_searches += 1
        print(f"Found {len(results)} results in {end_time - start_time:.2f} seconds:\n")

        for i, result in enumerate(results):
            print(f"Result {i+1}:")
            print(f"File: {result.filepath}")
            print(f"Lines: {result.start_line}-{result.end_line}")
            print("Content snippet:")
            print("-" * 40)
            # Print first 5 lines of content
            content_lines = result.content.splitlines()
            preview = "\n".join(content_lines[:min(5, len(content_lines))])
            print(preview + ("..." if len(content_lines) > 5 else ""))
            print("-" * 40)
            print()

    print("=" * 80)
    print(f"Summary: {successful_searches} out of {len(queries)} searches returned results.")
    if successful_searches == 0:
        print("No search returned results. This might indicate an issue with the indexing process.")
        print("Check the error messages above for more information.")
    else:
        print("The indexing and search system is working!")


if __name__ == "__main__":
    asyncio.run(main())
