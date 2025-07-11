"""
Vector Database Agnostic Framework

This package provides a generalized interface for working with vector databases,
allowing easy switching between different implementations (Qdrant, Pinecone, etc.)
"""

from .base_database import (
    BaseVectorDatabase,
    DatabaseConfig,
    Document,
    SearchResult,
    SearchResults,
    VectorDatabaseFactory
)

# Import implementations to register them with the factory
from . import qdrant_manager

__all__ = [
    'BaseVectorDatabase',
    'DatabaseConfig', 
    'Document',
    'SearchResult',
    'SearchResults',
    'VectorDatabaseFactory'
] 