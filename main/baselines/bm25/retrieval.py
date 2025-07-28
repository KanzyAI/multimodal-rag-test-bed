import os
import asyncio
import logging
from datasets import load_dataset

from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from pipelines.baselines.base_sparse_retrieval import BaseSparseRetrieval

import nltk
nltk.download('punkt')

class BM25:
    def __init__(self, task):
        self.task = task
        self.preprocessor = None  # Will be set by parent class
        
    async def prepare_corpus(self, corpus, documents):
        self.documents = documents
        self.corpus = corpus
        self.tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    async def embed_query(self, query):
        tokenized_query = word_tokenize(query.lower())
        return self.bm25.get_scores(tokenized_query)
    
    async def get_top_documents(self, query, top_k=10):
        tokenized_query = word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokenized_query)
        scored_docs = [(scores[i], i) for i in range(len(scores))]
        return sorted(scored_docs, key=lambda x: x[0], reverse=True)[:top_k]

    async def search(self, query, top_k=10):
        top_docs = await self.get_top_documents(query, top_k)
        
        results = {}
        for score, doc_idx in top_docs:
            doc = self.documents[doc_idx]
            filename = doc['filename']
            results[filename] = score
        return results
    

class BM25Retrieval(BaseSparseRetrieval):
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
        
        self.bm25_model = BM25(task)
        
    async def _prepare_retrieval_model(self):
        """Prepare the BM25 model with the corpus."""
        await self.bm25_model.prepare_corpus(self.corpus, self.documents)

    async def search(self, query, top_k=None):
        """Search using BM25."""
        if top_k is None:
            top_k = self.top_k
        return await self.bm25_model.search(query, top_k)

async def main():
    
    task=f"ibm-research/REAL-MM-RAG_FinReport"
    qrels_file = os.path.join(os.getcwd(), f"results/retrieval/{task.split('_')[-1].lower()}/baseline/bm25")
    log_dir = os.path.join(os.getcwd(), f"logs/retrieval/{task.split('_')[-1].lower()}/baseline/bm25")
    
    os.makedirs(qrels_file, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    bm25_retrieval = BM25Retrieval(
        task=task,
        qrels_file=qrels_file,
        log_dir=log_dir,
    )
    
    await bm25_retrieval()
    
if __name__ == "__main__":
    asyncio.run(main()) 