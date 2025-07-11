import os
import asyncio
from pipelines.text.indexing import TextIndexing
from main.embedders.text.linq import MultiTextEmbedder
from vector_dabases import DatabaseConfig, VectorDatabaseFactory

if __name__ == "__main__":

    text_embedder = MultiTextEmbedder()

    text_config = DatabaseConfig(
        url=os.getenv("TEXT_MULTI_QDRANT_URL"),
        api_key=os.getenv("TEXT_MULTI_QDRANT_API_KEY"),
        collection_name=f"text-multi-{os.getenv('TASK')}",
        vector_size=128,
        vector_type="multi",
    )

    database_mapping = {
        "text": VectorDatabaseFactory.create_database(os.getenv("DATABASE_TYPE"), text_config)
    }

    indexing_instance = TextIndexing(
          task=f"ibm-research/REAL-MM-RAG_{os.getenv('TASK')}", 
        embedder=text_embedder,
        database_mapping=database_mapping,
    )
    
    asyncio.run(indexing_instance())
