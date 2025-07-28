import os
import asyncio
import logging
import numpy as np
from typing import Dict, List, Tuple
from scipy.sparse import vstack, csr_matrix
from yasem import SpladeEmbedder
from pipelines.baselines.base_sparse_retrieval import BaseSparseRetrieval

class SPLADEModel:
    """SPLADE-v3 implementation using the YASEM library wrapper."""
    
    def __init__(self, task: str):
        self.task = task
        self.model_name = "naver/splade-v3"
        self.embedder = SpladeEmbedder(self.model_name)
        self.document_embeddings = []
        self.documents = []
        
        logging.info(f"Initialized SPLADE-v3 using YASEM with model: {self.model_name}")
        
    async def prepare_corpus(self, corpus: List[str], documents: List[Dict]):
        """Prepare the corpus by encoding all documents."""
        self.documents = documents
        logging.info(f"Encoding {len(corpus)} documents with SPLADE-v3...")
        self.document_embeddings = self.embedder.encode(corpus)
        logging.info("SPLADE-v3 document encoding completed.")
    
    async def search(self, query: str, top_k: int = 10) -> Dict[str, float]:
        """Search for documents given a query."""
        query_embedding_list = self.embedder.encode([query])

        if isinstance(query_embedding_list, list):
            query_embedding = vstack(query_embedding_list)
        else:
            query_embedding = query_embedding_list

        similarities = self.embedder.similarity(query_embedding, self.document_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = {
            self.documents[idx]['filename']: float(similarities[idx])
            for idx in top_indices
        }
        return results
    
    def get_query_expansion(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        query_embedding = self.embedder.encode([query])[0]
        token_values = self.embedder.get_token_values(query_embedding)
        return sorted(token_values.items(), key=lambda x: x[1], reverse=True)[:top_k]


class SPLADERetrieval(BaseSparseRetrieval):
    """SPLADE-v3 Retrieval implementation using YASEM."""
    
    def __init__(
        self,
        task: str,
        qrels_file: str,
        log_dir: str,
        top_k: int = 5,
        **kwargs
    ):
        super().__init__(
            task=task,
            qrels_file=qrels_file,
            log_dir=log_dir,
            top_k=top_k,
            **kwargs
        )
        self.splade_model = SPLADEModel(task)
        logging.info("Initialized SPLADE-v3 retrieval using YASEM")
        
    async def _prepare_retrieval_model(self):
        await self.splade_model.prepare_corpus(self.corpus, self.documents)

    async def search(self, query: str, top_k: int = None) -> Dict[str, float]:
        if top_k is None:
            top_k = self.top_k
        return await self.splade_model.search(query, top_k)
    
    def get_query_expansion(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        return self.splade_model.get_query_expansion(query, top_k)


async def main():
    task = "ibm-research/REAL-MM-RAG_FinReport"
    qrels_file = os.path.join(os.getcwd(), f"results/retrieval/{task.split('_')[-1].lower()}/baseline/splade")
    log_dir = os.path.join(os.getcwd(), f"logs/retrieval/{task.split('_')[-1].lower()}/baseline/splade")
    
    os.makedirs(qrels_file, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    splade_retrieval = SPLADERetrieval(
        task=task,
        qrels_file=qrels_file,
        log_dir=log_dir,
    )
    
    await splade_retrieval()
    
if __name__ == "__main__":
    asyncio.run(main())
