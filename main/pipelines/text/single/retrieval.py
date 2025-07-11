import os
import asyncio
from pipelines.text.retrieval import TextRetrieval
from vector_dabases import DatabaseConfig, VectorDatabaseFactory
from main.embedders.text.linq import SingleTextEmbedder

if __name__ == "__main__":

    text_embedder = SingleTextEmbedder()
    
    text_config = DatabaseConfig(
        url=os.getenv("API_KEY"),
        api_key=os.getenv("TEXT_SINGLE_QDRANT_API_KEY"),
        collection_name=f"text-single-{os.getenv('TASK')}",
        vector_size=4096,
        vector_type="single",
    )

    database_mapping = {
        "text": VectorDatabaseFactory.create_database(os.getenv("DATABASE_NAME"), text_config)
    }

    retrieval_instance = TextRetrieval(
        task=f"ibm-research/REAL-MM-RAG_{os.getenv('TASK')}", 
        embedder=text_embedder,
        database_mapping=database_mapping,
        qrels_file=os.path.join(os.getcwd(), f"results/retrieval/{os.getenv('TASK').lower()}/text/single"),
    )
    asyncio.run(retrieval_instance())
