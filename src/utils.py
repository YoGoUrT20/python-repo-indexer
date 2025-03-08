"""
Utility functions for the repository indexing system.
"""
import hashlib
import os
from pathlib import Path
from typing import Generator, List, Set, Tuple, Union

import pygit2


def get_file_hash(file_path: Union[str, Path]) -> str:
    """Calculate a hash of a file's contents. """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    return file_hash


def get_repo_branch(repo_path: Union[str, Path]) -> str:
    """Get the current branch of a git repository.

    Args:
        repo_path: Path to the repository

    Returns:
        The current branch name or "NONE" if not a git repository
    """
    try:
        repo = pygit2.Repository(repo_path)
        if repo.head_is_detached:
            return repo.head.target.hex[:10]
        return repo.head.shorthand
    except (pygit2.GitError, KeyError):
        return "NONE"


def should_ignore_file(file_path: Union[str, Path], ignore_patterns: List[str] = None) -> bool:
    """Check if a file should be ignored based on patterns.

    Args:
        file_path: Path to the file
        ignore_patterns: List of glob patterns to ignore

    Returns:
        True if the file should be ignored, False otherwise
    """
    default_patterns = [
        ".git/",
        "__pycache__/",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        "venv/",
        "env/",
        "node_modules/",
        "*.so",
        "*.dylib",
        "*.dll",
        ".DS_Store",
        "Thumbs.db"
    ]

    patterns = default_patterns
    if ignore_patterns:
        patterns.extend(ignore_patterns)

    path_str = str(file_path)
    for pattern in patterns:
        if pattern.endswith("/"):
            # Directory pattern
            if pattern[:-1] in path_str.split(os.sep):
                return True
        elif pattern.startswith("*."):
            # Extension pattern
            if path_str.endswith(pattern[1:]):
                return True
        elif pattern in path_str:
            # Simple substring pattern
            return True

    return False


def walk_directory(
    directory: Union[str, Path],
    ignore_patterns: List[str] = None
) -> Generator[Path, None, None]:
    """Walk a directory recursively and yield file paths.

    Args:
        directory: Path to the directory
        ignore_patterns: List of glob patterns to ignore

    Yields:
        File paths
    """
    directory = Path(directory)

    for root, dirs, files in os.walk(directory):
        # Filter out ignored directories
        dirs[:] = [d for d in dirs if not should_ignore_file(os.path.join(root, d), ignore_patterns)]

        # Yield non-ignored files
        for file in files:
            file_path = Path(os.path.join(root, file))
            if not should_ignore_file(file_path, ignore_patterns):
                yield file_path


def get_file_modification_time(file_path: Union[str, Path]) -> float:
    """Get the modification time of a file.

    Args:
        file_path: Path to the file

    Returns:
        Modification time as a float (seconds since epoch)
    """
    return os.path.getmtime(file_path)


def count_lines(content: str) -> int:
    """Count the number of lines in a string.

    Args:
        content: The string to count lines in

    Returns:
        Number of lines
    """
    return len(content.splitlines())


def extract_lines(content: str, start_line: int, end_line: int) -> str:
    """Extract specific lines from a string.

    Args:
        content: The string to extract lines from
        start_line: Start line (1-indexed)
        end_line: End line (1-indexed, inclusive)

    Returns:
        Extracted lines as a string
    """
    lines = content.splitlines()
    return "\n".join(lines[start_line-1:end_line])
