import os
import asyncio
from main.pipelines.visual.retrieval import ColpaliRetrieval
from main.vector_dabases import DatabaseConfig, VectorDatabaseFactory

if __name__ == "__main__":

    visual_config = DatabaseConfig(
        url=os.getenv("VISUAL_MULTI_QDRANT_URL"),
        api_key=os.getenv("VISUAL_MULTI_QDRANT_API_KEY"),
        collection_name=f"visual-multi-{os.getenv('TASK')}",
        vector_size=128,
        vector_type="multi",
    )

    database_mapping = {
        "visual": VectorDatabaseFactory.create_database(os.getenv("DATABASE_NAME"), visual_config)
    }

    retrieval_instance = ColpaliRetrieval(
        task=f"ibm-research/REAL-MM-RAG_{os.getenv('TASK')}", 
        database_mapping=database_mapping,
        qrels_file=os.path.join(os.getcwd(), f"results/retrieval/{os.getenv('TASK').lower()}/visual/multi"),
        model_name="Metric-AI/ColQwen2.5-7b-multilingual-v1.0",
        model_type="multi",
    )
    
    asyncio.run(retrieval_instance())
