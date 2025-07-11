import os
import asyncio
from main.pipelines.visual.indexing import ColpaliIndexing
from main.vector_dabases import DatabaseConfig, VectorDatabaseFactory

if __name__ == "__main__":

    visual_config = DatabaseConfig(
        url=os.getenv("DATABASE_URL"),
        api_key=os.getenv("DATABASE_API_KEY"),
        collection_name=f"multimodal-dense-{os.getenv('TASK').lower()}",
        vector_size=1024,
        vector_type="single",
    )

    database_mapping = {
        "visual-single": VectorDatabaseFactory.create_database(os.getenv("DATABASE_NAME"), visual_config)
    }

    indexing_instance = ColpaliIndexing(
        task=f"ibm-research/REAL-MM-RAG_{os.getenv('TASK')}", 
        database_mapping=database_mapping,
        model_name="nomic-ai/nomic-embed-multimodal-7b",
        model_type="single",
    )

    asyncio.run(indexing_instance())
