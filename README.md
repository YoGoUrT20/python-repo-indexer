# Repository Indexer and Context Search

A Python implementation of a repository indexing and context search system.

## Features

- Repository scanning and file tracking
- File chunking based on code structure
- Full-text search using SQLite FTS5
- Vector embeddings for semantic search
- Context-aware retrieval pipeline

## Getting Started

### Requirements

- Python 3.8+
- See `requirements.txt` for dependency details

### Installation

```bash
# Clone the repository
git clone https://github.com/YoGoUrT20/python-repo-indexer.git
cd python-repo-indexer

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

```python
from repo_indexer import RepoIndexer

# Initialize the indexer
indexer = RepoIndexer(repo_path="/path/to/repository")

# Index the repository
indexer.index()

# Search for context
results = indexer.search("How does the authentication system work?")
for result in results:
    print(f"File: {result.filepath}, Lines: {result.start_line}-{result.end_line}")
    print(result.content)
```

## Architecture

The system consists of several components:

1. **Repository Scanner**: Scans repository files and tracks changes
2. **Chunker**: Splits files into meaningful code chunks
3. **Indexers**:
   - **FullTextIndexer**: Creates SQLite FTS index for keyword search
   - **VectorIndexer**: Creates embeddings for semantic search
4. **Retrieval Pipeline**: Combines search results for optimal context retrieval

## License

Apache-2.0