import os
import asyncio
from pipelines.visual.retrieval import ColpaliRetrieval
from vector_dabases import DatabaseConfig, VectorDatabaseFactory

if __name__ == "__main__":

    visual_config = DatabaseConfig(
        url=os.getenv("API_KEY"),
        api_key=os.getenv("VISUAL_SINGLE_QDRANT_API_KEY"),
        collection_name=f"visual-{os.getenv('TASK').lower()}",
        vector_size=1024,
        vector_type="single",
    )

    database_mapping = {
        "visual": VectorDatabaseFactory.create_database(os.getenv("DATABASE_NAME"), visual_config)
    }

    retrieval_instance = ColpaliRetrieval(
        task=f"ibm-research/REAL-MM-RAG_{os.getenv('TASK')}", 
        database_mapping=database_mapping,
        qrels_file=os.path.join(os.getcwd(), f"results/retrieval/{os.getenv('TASK').lower()}/visual/single/"),
        model_name="nomic-ai/nomic-embed-multimodal-7b"
    )
    
    asyncio.run(retrieval_instance())
