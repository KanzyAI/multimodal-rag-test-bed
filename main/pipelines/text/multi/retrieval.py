import os
import asyncio
from pipelines.text.retrieval import TextRetrieval
from vector_dabases import DatabaseConfig, VectorDatabaseFactory
from main.embedders.text.linq import MultiTextEmbedder

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

    retrieval_instance = TextRetrieval(
          task=f"ibm-research/REAL-MM-RAG_{os.getenv('TASK')}", 
        embedder=text_embedder,
        database_mapping=database_mapping,
        qrels_file=os.path.join(os.getcwd(), f"results/retrieval/{os.getenv('TASK').lower()}/text/multi"),
    )
    asyncio.run(retrieval_instance())
