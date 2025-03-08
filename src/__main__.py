"""
Command-line interface for the repository indexer.
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

from .repo_indexer import RepoIndexer


async def index_command(args):
    """Run the indexer command."""
    indexer = RepoIndexer(
        repo_path=args.repo_path,
        db_path=args.db_path,
        ignore_patterns=args.ignore,
        embedding_model=args.model,
        files_per_batch=args.batch_size
    )

    await indexer.index(force_reindex=args.force)


async def search_command(args):
    """Run the search command."""
    indexer = RepoIndexer(
        repo_path=args.repo_path,
        db_path=args.db_path,
        embedding_model=args.model
    )

    results = await indexer.search(
        query=args.query,
        n=args.limit,
        directory=args.directory,
        include_embeddings=not args.no_embeddings
    )

    if not results:
        print("No results found.")
        return

    print(f"Found {len(results)} results:\n")

    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"File: {result.filepath}")
        print(f"Lines: {result.start_line}-{result.end_line}")
        print(f"Description: {result.description}")
        print("Content:")
        print("-" * 80)
        print(result.content)
        print("-" * 80)
        print()


async def clear_command(args):
    """Run the clear command."""
    indexer = RepoIndexer(
        repo_path=args.repo_path,
        db_path=args.db_path
    )

    await indexer.clear_index()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Repository indexing and context search tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Parent parser for common arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--repo-path", "-r", type=str, default=".",
                               help="Path to the repository (default: current directory)")
    parent_parser.add_argument("--db-path", "-d", type=str, default=None,
                               help="Path to the database file (default: <repo_path>/.repo_index/index.db)")
    parent_parser.add_argument("--model", "-m", type=str, default="all-MiniLM-L6-v2",
                               help="Embedding model to use (default: all-MiniLM-L6-v2)")

    # Index command
    index_parser = subparsers.add_parser("index", parents=[parent_parser],
                                         help="Index a repository")
    index_parser.add_argument("--force", "-f", action="store_true",
                              help="Force reindexing of all files")
    index_parser.add_argument("--ignore", "-i", type=str, nargs="+", default=[],
                              help="Patterns of files to ignore")
    index_parser.add_argument("--batch-size", "-b", type=int, default=50,
                              help="Number of files to process in a batch")

    # Search command
    search_parser = subparsers.add_parser("search", parents=[parent_parser],
                                          help="Search a repository")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", "-l", type=int, default=10,
                               help="Maximum number of results (default: 10)")
    search_parser.add_argument("--directory", type=str, default=None,
                               help="Limit search to a specific directory")
    search_parser.add_argument("--no-embeddings", action="store_true",
                               help="Disable vector embeddings search")

    # Clear command
    clear_parser = subparsers.add_parser("clear", parents=[parent_parser],
                                         help="Clear the index")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "index":
        asyncio.run(index_command(args))
    elif args.command == "search":
        asyncio.run(search_command(args))
    elif args.command == "clear":
        asyncio.run(clear_command(args))

    return 0


if __name__ == "__main__":
    sys.exit(main())
