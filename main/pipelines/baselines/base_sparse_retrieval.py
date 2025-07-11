import os
import asyncio
import logging
import time
import json
from abc import ABC, abstractmethod
from datasets import load_dataset
from pipelines.text.preprocessor import Preprocessor
from pipelines.base.base_retrieval import BaseRetrieval

class BaseSparseRetrieval(BaseRetrieval):
    """Base class for sparse retrieval methods like BM25 and SPLADE."""
    
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
            database_mapping={},
            **kwargs
        )
        
        self.top_k = top_k
        self.preprocessor = Preprocessor(
            output_path=os.path.join(os.getcwd(), f"main/baselines/baseline_chunks_{task.split('_')[-1]}.json"), 
            task=task
        )
        
    async def prepare_corpus(self):
        """Prepare the corpus for retrieval. Should be implemented by subclasses."""
        corpus = []
        dataset = load_dataset(self.task, split="test")
        self.documents = []
        
        semaphore = asyncio.Semaphore(os.cpu_count() - 2)
        
        async def process_row(row):
            async with semaphore:
                filename = row['image_filename']
                image = row['image']
                chunks = await self.preprocessor(image, filename)
                
                for chunk in chunks:
                    corpus.append(chunk)
                    self.documents.append({"text": chunk, "filename": filename})
        
        tasks = [process_row(row) for row in dataset]
        await asyncio.gather(*tasks)

        self.corpus = corpus
        await self._prepare_retrieval_model()

    @abstractmethod
    async def _prepare_retrieval_model(self):
        """Prepare the specific retrieval model (BM25, SPLADE, etc.)"""
        pass

    @abstractmethod
    async def search(self, query, top_k=None):
        """Search for documents given a query. Should return dict of {filename: score}"""
        pass

    async def embed_query(self, query):
        """
        For sparse methods, we don't actually need embeddings, but we need to maintain
        the same interface as BaseRetrieval. Return None for both text and visual
        embeddings since sparse methods will handle the query directly.
        """
        return None, None

    async def process_query(self, query):
        """
        Override the base process_query to use sparse search directly
        instead of the qdrant-based retrieval.
        """
        total_start_time = time.monotonic()
        embed_duration = 0
        retrieval_duration = 0
        status = "Pending"
        error_message = None

        try:
            async with self.semaphore:
                status = "Processing"
                logging.debug(f"Processing query with {self.__class__.__name__}: {query}")

                # Sparse methods don't need separate embedding step, but we'll time the search
                embed_start_time = time.monotonic()
                embed_duration = time.monotonic() - embed_start_time

                # --- Sparse Search Timing ---
                retrieval_start_time = time.monotonic()
                retrieved = await self.search(query, top_k=self.top_k)
                retrieval_duration = time.monotonic() - retrieval_start_time
                
                logging.info(f"{self.__class__.__name__} search returned {len(retrieved)} results for query: {query}")

                # Store results
                self.qrels[query] = retrieved

                with open(self.qrels_file, "w") as f:
                    json.dump(self.qrels, f, indent=4)

                status = "Success"

        except Exception as e:
            logging.error(f"Error during {self.__class__.__name__} retrieval for query {query}: {e}", exc_info=True)
            status = "Retrieval Error"
            error_message = str(e)

        finally:
            total_duration = time.monotonic() - total_start_time
            self.timing_logs.append({
                "query": query,
                "embed_duration_sec": embed_duration,
                "retrieval_duration_sec": retrieval_duration,
                "total_duration_sec": total_duration,
                "status": status,
                "error": error_message
            })

    async def __call__(self):
        """
        Override the main entry point since sparse methods don't need qdrant initialization.
        """
        await self.prepare_corpus()

        try:
            for query_column in ["query", "rephrase_level_1", "rephrase_level_2", "rephrase_level_3"]:
                await self.process_all_queries(query_column)
                logging.info(f"{self.__class__.__name__} retrieval completed for {query_column}")

        except Exception as e:
            logging.error(f"Error during {self.__class__.__name__} retrieval run: {e}", exc_info=True)
            self.timing_logs.append({
                "method": "__call__",
                "filename": None,
                "duration_sec": 0,
                "status": "Error", 
                "error": f"Error in __call__: {e}"
            })

        finally:
            self._save_performance_logs()
            logging.info(f"{self.__class__.__name__} retrieval run finished.") 