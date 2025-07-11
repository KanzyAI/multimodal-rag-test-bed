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
from .pinecone_manager import PineconeManager
from .qdrant_manager import QdrantManager

VectorDatabaseFactory.register_database("pinecone", PineconeManager)
VectorDatabaseFactory.register_database("qdrant", QdrantManager)


__all__ = [
    'BaseVectorDatabase',
    'DatabaseConfig', 
    'Document',
    'SearchResult',
    'SearchResults',
    'VectorDatabaseFactory'
] 