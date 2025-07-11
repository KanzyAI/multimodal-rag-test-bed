import os
import asyncio
from main.pipelines.visual.indexing import ColpaliIndexing
from main.vector_dabases import DatabaseConfig, VectorDatabaseFactory

if __name__ == "__main__":

    visual_config = DatabaseConfig(
        url=os.getenv("VISUAL_MEANPOOL_QDRANT_URL"),
        api_key=os.getenv("VISUAL_MEANPOOL_QDRANT_API_KEY"),
        collection_name=f"visual-meanpool-{os.getenv('TASK')}",
        vector_size=128,
        vector_type="multi",
    )

    database_mapping = {
        "visual": VectorDatabaseFactory.create_database(os.getenv("DATABASE_TYPE"), visual_config)
    }

    indexing_instance = ColpaliIndexing(
        task=f"ibm-research/REAL-MM-RAG_{os.getenv('TASK')}", 
        database_mapping=database_mapping,
        model_name="Metric-AI/ColQwen2.5-7b-multilingual-v1.0",
        model_type="multi",
    )

    asyncio.run(indexing_instance())
