import uuid
import stamina
import logging
from typing import Literal, List, Set, Optional, Any, Dict, Union
from .base_database import BaseVectorDatabase, DatabaseConfig, Document, SearchResult, SearchResults, VectorDatabaseFactory
from pinecone import PineconeAsyncio, ServerlessSpec, PodSpec

class PineconeManager(BaseVectorDatabase):
    
    # Distance metric mapping
    DISTANCE_MAPPING = {
        "cosine": "cosine",
        "euclidean": "euclidean", 
        "dot": "dotproduct",
        "manhattan": "euclidean",  # Pinecone doesn't support manhattan, fallback to euclidean
    }
    
    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        
        # Raise error for multi-vector type since Pinecone doesn't support it natively
        if config.vector_type == "multi":
            raise ValueError(
                "Multi-vector type is not supported by Pinecone. "
                "Pinecone only supports single dense or sparse vectors per document. "
                "Please use vector_type='single' instead."
            )
        
        # Configuration for operations
        self.timeout = config.extra_params.get("timeout", 120)
        self.environment = config.extra_params.get("environment", "us-east-1-aws")
        self.cloud = config.extra_params.get("cloud", "aws")
        self.region = config.extra_params.get("region", "us-east-1")
        
        # Pinecone client will be initialized in async context
        self.pc = None
        
    async def _ensure_client(self):
        """Ensure Pinecone client is initialized."""
        if self.pc is None:
            self.pc = PineconeAsyncio(api_key=self.config.api_key)

    async def initialize_collection(self) -> None:
        """Initialize collection (index) if it doesn't already exist."""
        try:
            await self._ensure_client()
            
            async with self.pc as pc:
                # Check if index already exists
                if not await pc.has_index(self.collection_name):
                    # Map generic distance metric to Pinecone-specific
                    metric = self.DISTANCE_MAPPING.get(
                        self.config.distance_metric.lower(), 
                        "cosine"
                    )
                    
                    # Create serverless spec (default) or pod spec based on config
                    if self.config.extra_params.get("use_pods", False):
                        spec = PodSpec(
                            environment=self.environment,
                            pod_type=self.config.extra_params.get("pod_type", "p1.x1"),
                            pods=self.config.extra_params.get("pods", 1),
                            replicas=self.config.extra_params.get("replicas", 1),
                            shards=self.config.extra_params.get("shards", 1)
                        )
                    else:
                        spec = ServerlessSpec(
                            cloud=self.cloud,
                            region=self.region
                        )

                    await pc.create_index(
                        name=self.collection_name,
                        dimension=self.vector_size,
                        metric=metric,
                        spec=spec,
                        deletion_protection="disabled"  # Allow deletion for development
                    )

                    logging.info(f"Index {self.collection_name} created successfully.")
                else:
                    logging.info(f"Index {self.collection_name} already exists.")
                
        except Exception as e:
            logging.error(f"Index setup error: {e}")
            raise
    
    async def get_indexed_files(self) -> Set[str]:
        """Retrieves all indexed file keys from the collection."""
        indexed_files = set()
        
        try:
            await self._ensure_client()
            
            # For now, return empty set since Pinecone doesn't have a direct way
            # to list all document IDs without querying. In a real implementation,
            # you might want to maintain a separate metadata store for tracking files.
            logging.warning("get_indexed_files is not efficiently implementable with Pinecone async API. Returning empty set.")
            
        except Exception as e:
            logging.warning(f"Could not retrieve indexed files: {e}")
            
        return indexed_files

    @stamina.retry(on=Exception, attempts=3)
    async def index_document(self, embedding: Union[List[float], List[List[float]]], metadata: Dict[str, Any]) -> bool:
        """Index a single document with embedding and metadata."""
        try:
            # Check for multi-vector type and raise error
            if isinstance(embedding[0], list):
                raise ValueError(
                    "Multi-vector embeddings are not supported by Pinecone. "
                    "Please provide a single vector as List[float] instead of List[List[float]]."
                )
            
            await self._ensure_client()
            document_id = self.generate_document_id(metadata)
            
            # Use the synchronous client for data operations since async operations
            # need to be done through a different pattern in Pinecone SDK v7+
            # We'll create a synchronous client for data operations
            from pinecone import Pinecone
            sync_pc = Pinecone(api_key=self.config.api_key)
            index = sync_pc.Index(self.collection_name)
            
            index.upsert(
                vectors=[{
                    "id": document_id,
                    "values": embedding,
                    "metadata": metadata
                }]
            )
            
            return True
        except Exception as e:
            logging.error(f"Error during document indexing: {e}")
            return False

    async def search(
        self, 
        query_embedding: Union[List[float], List[List[float]]], 
        limit: int = 5,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> SearchResults:
        """Perform similarity search and return SearchResults."""
        # Check for multi-vector query and raise error
        if isinstance(query_embedding[0], list):
            raise ValueError(
                "Multi-vector queries are not supported by Pinecone. "
                "Please provide a single vector as List[float] instead of List[List[float]]."
            )
        
        await self._ensure_client()
        
        # Use the synchronous client for data operations since async operations
        # need to be done through a different pattern in Pinecone SDK v7+
        from pinecone import Pinecone
        sync_pc = Pinecone(api_key=self.config.api_key)
        index = sync_pc.Index(self.collection_name)
        
        pinecone_results = index.query(
            vector=query_embedding,
            top_k=limit,
            filter=filter_conditions,
            include_metadata=True,
            include_values=False
        )
        
        search_results = []
        for match in pinecone_results.matches:
            search_results.append(
                SearchResult(
                    document_id=str(match.id),
                    score=match.score,
                    metadata=match.metadata or {}
                )
            )
        
        return SearchResults(search_results)

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.pc:
            await self.pc.close()


# Register PineconeManager with the factory
VectorDatabaseFactory.register_database("pinecone", PineconeManager) 