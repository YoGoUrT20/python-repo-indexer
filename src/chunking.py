"""
Code chunking system for splitting files into meaningful segments.
"""
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .models import Chunk
from .utils import count_lines, extract_lines, get_file_hash


def should_chunk_file(file_path: Union[str, Path]) -> bool:
    """Determine if a file should be chunked based on its extension.

    Args:
        file_path: Path to the file

    Returns:
        True if the file should be chunked, False otherwise
    """
    # File extensions to chunk (common code files)
    CHUNK_EXTENSIONS = {
        # Programming languages
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".cs",
        ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala", ".hs",
        # Web
        ".html", ".css", ".scss", ".sass", ".less",
        # Data/Config
        ".json", ".yaml", ".yml", ".toml", ".xml", ".md", ".mdx",
        # Shell
        ".sh", ".bash", ".zsh", ".ps1", ".cmd", ".bat",
        # Other common code files
        ".sql", ".graphql", ".proto", ".r", ".dart"
    }

    # Check if the file has an extension we want to chunk
    file_ext = Path(file_path).suffix.lower()
    return file_ext in CHUNK_EXTENSIONS


def estimate_chunk_boundaries(content: str, max_chunk_size: int = 1000) -> List[Tuple[int, int]]:
    """Estimate chunk boundaries based on code structure.

    Args:
        content: File content
        max_chunk_size: Maximum chunk size in characters

    Returns:
        List of (start_line, end_line) tuples
    """
    lines = content.splitlines()
    total_lines = len(lines)

    # For empty or very small files, just return the whole file as one chunk
    if total_lines <= 1:
        return [(1, max(1, total_lines))]

    # Start with largest structures (classes, functions)
    chunks = []
    current_chunk_start = 0

    # Pattern for detecting code block starts
    block_start_patterns = [
        # Function definitions
        r"^\s*(def|function|async def|async function|class)\s+\w+",
        # Class definitions
        r"^\s*(class|interface|abstract class|enum)\s+\w+",
        # Control structures with blocks
        r"^\s*(if|for|while|switch|try|catch)\s*\(",
        # Variable declarations and assignments
        r"^\s*(const|let|var|public|private|protected)\s+\w+",
        # Markdown headers
        r"^#+\s+",
        # HTML tags
        r"^\s*<[a-zA-Z_][a-zA-Z0-9_\-]*",
    ]

    # Compile patterns for better performance
    patterns = [re.compile(pattern) for pattern in block_start_patterns]

    # Create chunks based on structure and size
    for i, line in enumerate(lines):
        # Start a new chunk when:
        # 1. We find a new block start
        # 2. Current chunk has reached max size
        is_block_start = any(pattern.match(line) for pattern in patterns)
        chunk_size = i - current_chunk_start

        if (is_block_start and chunk_size > 0) or chunk_size >= max_chunk_size:
            if current_chunk_start < i:
                chunks.append((current_chunk_start + 1, i))  # Convert to 1-indexed lines
            current_chunk_start = i

    # Add the last chunk if needed
    if current_chunk_start < total_lines:
        chunks.append((current_chunk_start + 1, total_lines))  # Convert to 1-indexed lines

    return chunks


def chunk_file(file_path: Union[str, Path], max_chunk_size: int = 1000) -> List[Chunk]:
    """Chunk a file into meaningful segments.

    Args:
        file_path: Path to the file
        max_chunk_size: Maximum chunk size in lines

    Returns:
        List of Chunk objects
    """
    path = Path(file_path)

    if not path.exists():
        print(f"File not found: {path}")
        return []

    if not should_chunk_file(path):
        return []

    # Skip files that are too large (likely binary files)
    try:
        file_size = path.stat().st_size
        if file_size > 1024 * 1024:  # 1MB
            print(f"Skipping large file (likely binary): {path} ({file_size/1024/1024:.2f} MB)")
            return []
    except Exception:
        pass

    try:
        # First try UTF-8
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            # Try with Latin-1 (which can read any byte sequence)
            with open(path, "r", encoding="latin-1") as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading file {path}: {e}")
            return []
    except Exception as e:
        print(f"Error reading file {path}: {e}")
        return []

    try:
        file_hash = get_file_hash(path)
        chunk_boundaries = estimate_chunk_boundaries(content, max_chunk_size)

        chunks = []
        for i, (start_line, end_line) in enumerate(chunk_boundaries):
            try:
                chunk_content = extract_lines(content, start_line, end_line)
                chunks.append(
                    Chunk(
                        filepath=str(path),
                        start_line=start_line,
                        end_line=end_line,
                        content=chunk_content,
                        digest=file_hash,
                        index=i
                    )
                )
            except Exception as e:
                print(f"Error creating chunk {i} for file {path}: {e}")

        return chunks
    except Exception as e:
        print(f"Error chunking file {path}: {e}")
        return []
