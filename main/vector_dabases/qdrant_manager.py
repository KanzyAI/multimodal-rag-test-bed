import uuid
import stamina
import logging
from typing import Literal, List, Set, Optional, Any, Dict, Union
from qdrant_client.http import models
from qdrant_client.models import SearchParams
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from .base_database import BaseVectorDatabase, DatabaseConfig, Document, SearchResult, SearchResults, VectorDatabaseFactory


class QdrantManager(BaseVectorDatabase):
    
    # Distance metric mapping
    DISTANCE_MAPPING = {
        "cosine": models.Distance.COSINE,
        "euclidean": models.Distance.EUCLID,
        "dot": models.Distance.DOT,
        "manhattan": models.Distance.MANHATTAN,
    }
    
    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        
        # Get timeout from extra_params or use default
        timeout = config.extra_params.get("timeout", 120)

        self.qdrant_client = AsyncQdrantClient(
            url=config.url,
            api_key=config.api_key,
            timeout=timeout,
        )
        
        # Configuration for operations
        self.scroll_limit = config.extra_params.get("scroll_limit", 10000)
        self.search_timeout = config.extra_params.get("search_timeout", 100)
        self.exact_search = config.extra_params.get("exact_search", True)
        
    async def initialize_collection(self) -> None:
        """Initialize collection if it doesn't already exist."""
        try:
            existing_collections = (await self.qdrant_client.get_collections()).collections
            existing_names = [col.name for col in existing_collections]

            if self.collection_name not in existing_names:
                # Map generic distance metric to Qdrant-specific
                distance = self.DISTANCE_MAPPING.get(
                    self.config.distance_metric.lower(), 
                    models.Distance.COSINE
                )
                
                vector_config = models.VectorParams(
                    size=self.vector_size,
                    distance=distance,
                )

                if self.vector_type == "multi":
                    vector_config = models.VectorParams(
                        size=self.vector_size,
                        distance=distance,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM
                        ),
                    )

                await self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    on_disk_payload=True,
                    vectors_config=vector_config,
                )

                logging.info(f"Collection {self.collection_name} created successfully with {self.vector_type} vector type.")
            else:
                logging.info(f"Collection {self.collection_name} already exists.")
        except Exception as e:
            logging.error(f"Collection setup error: {e}")
            raise
    
    async def get_indexed_files(self) -> Set[str]:
        """Retrieves all indexed file keys from the collection."""
        indexed_files = set()
        
        scroll = await self.qdrant_client.scroll(
            collection_name=self.collection_name,
            scroll_filter=None,
            limit=self.scroll_limit,
            with_payload=True, 
        )

        for point in scroll[0]:
            payload = point.payload
            if payload:
                file_key = payload.get("filename", "")
                indexed_files.add(file_key)

        return indexed_files

    @stamina.retry(on=Exception, attempts=3)
    async def index_document(self, embedding: Union[List[float], List[List[float]]], metadata: Dict[str, Any]) -> bool:
        """Index a single document with embedding and metadata."""
        try:
            document_id = self.generate_document_id(metadata)

            point = models.PointStruct(
                id=document_id,
                vector=embedding,
                payload=metadata,
            )
            
            await self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[point],
                wait=True,
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
        qdrant_results = await self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=filter_conditions,
            with_payload=True,
            with_vectors=False,
            limit=limit,
            search_params=SearchParams(
                exact=self.exact_search,
            ),
            timeout=self.search_timeout,
        )
        
        search_results = []
        for point in qdrant_results.points:
            search_results.append(
                SearchResult(
                    document_id=str(point.id),
                    score=point.score,
                    metadata=point.payload or {}
                )
            )
        
        return SearchResults(search_results)

# Register QdrantManager with the factory
VectorDatabaseFactory.register_database("qdrant", QdrantManager)