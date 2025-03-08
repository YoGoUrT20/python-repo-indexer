"""
Database module for the repository indexing system.
Handles SQLite connection and schema creation.
"""
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker


# Define base class for SQLAlchemy models
Base = declarative_base()


class IndexEntry(Base):
    """Represents an entry in the index catalog."""
    __tablename__ = "index_catalog"
    
    id = sa.Column(sa.Integer, primary_key=True)
    path = sa.Column(sa.String, nullable=False)
    cache_key = sa.Column(sa.String, nullable=False)
    last_modified = sa.Column(sa.Float, nullable=False)
    last_indexed = sa.Column(sa.Float, nullable=False)


class IndexTag(Base):
    """Represents a tag on an indexed file."""
    __tablename__ = "index_tags"
    
    id = sa.Column(sa.Integer, primary_key=True)
    tag = sa.Column(sa.String, nullable=False)
    path = sa.Column(sa.String, nullable=False)
    cache_key = sa.Column(sa.String, nullable=False)
    
    __table_args__ = (
        sa.UniqueConstraint('tag', 'path', 'cache_key'),
    )


class Chunk(Base):
    """Represents a chunk of a file."""
    __tablename__ = "chunks"
    
    id = sa.Column(sa.Integer, primary_key=True)
    path = sa.Column(sa.String, nullable=False)
    cache_key = sa.Column(sa.String, nullable=False)
    chunk_index = sa.Column(sa.Integer, nullable=False)
    start_line = sa.Column(sa.Integer, nullable=False)
    end_line = sa.Column(sa.Integer, nullable=False)
    content = sa.Column(sa.Text, nullable=False)


class ChunkTag(Base):
    """Represents a tag on a chunk."""
    __tablename__ = "chunk_tags"
    
    id = sa.Column(sa.Integer, primary_key=True)
    chunk_id = sa.Column(sa.Integer, sa.ForeignKey("chunks.id"), nullable=False)
    tag = sa.Column(sa.String, nullable=False)


# Create FTS tables using raw SQL since SQLAlchemy doesn't support virtual tables well
def create_fts_tables(conn: sqlite3.Connection) -> None:
    """Create FTS5 tables for full-text search."""
    conn.executescript("""
    CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(
        path,
        content,
        tokenize = 'trigram'
    );

    CREATE TABLE IF NOT EXISTS fts_metadata (
        id INTEGER PRIMARY KEY,
        path TEXT NOT NULL,
        cache_key TEXT NOT NULL,
        chunk_id INTEGER NOT NULL,
        FOREIGN KEY (chunk_id) REFERENCES chunks (id),
        FOREIGN KEY (id) REFERENCES fts (rowid)
    );
    """)


class Database:
    """Handles database operations for the indexing system."""
    
    def __init__(self, db_path: Union[str, Path]):
        """Initialize the database.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create SQLAlchemy engine
        self.engine = sa.create_engine(f"sqlite:///{self.db_path}")
        self.Session = sessionmaker(bind=self.engine)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize the database schema."""
        # Create tables
        Base.metadata.create_all(self.engine)
        
        # Create FTS tables using raw connection
        with sqlite3.connect(self.db_path) as conn:
            create_fts_tables(conn)
    
    def execute(self, query: str, params: Optional[Tuple[Any, ...]] = None) -> List[Dict[str, Any]]:
        """Execute a raw SQL query and return results as dictionaries."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            try:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                try:
                    results = [dict(row) for row in cursor.fetchall()]
                except sqlite3.Error:
                    results = []
            except sqlite3.Error as e:
                print(f"SQLite error executing query: {e}")
                print(f"Query: {query}")
                print(f"Params: {params}")
                results = []
            
            return results
    
    def execute_write(self, query: str, params: Optional[Tuple[Any, ...]] = None) -> sqlite3.Cursor:
        """Execute a raw SQL write query and return the cursor."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                conn.commit()
            except sqlite3.Error as e:
                print(f"SQLite error executing write query: {e}")
                print(f"Query: {query}")
                print(f"Params: {params}")
                conn.rollback()
            return cursor
    
    def get_session(self) -> Session:
        """Get a SQLAlchemy session."""
        return self.Session()
    
    def execute_many(self, query: str, params_list: List[Tuple[Any, ...]]) -> None:
        """Execute a query with multiple parameter sets."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.executemany(query, params_list)
                conn.commit()
            except sqlite3.Error as e:
                print(f"SQLite error executing batch query: {e}")
                print(f"Query: {query}")
                print(f"Number of parameter sets: {len(params_list)}")
                conn.rollback() 