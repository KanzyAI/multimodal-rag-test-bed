import os
import json
import asyncio
from abc import ABC, abstractmethod
from datasets import load_dataset
from langsmith import traceable, get_current_run_tree
from typing import Dict
from vector_dabases import BaseVectorDatabase

class BaseRetrieval(ABC):
    def __init__(
        self,
        task: str,
        qrels_file: str,    
        database_mapping: Dict[str, BaseVectorDatabase],
        **kwargs 
    ):        
        self.task = task
        self.semaphore = asyncio.Semaphore(10) 
        self.hf_token = os.getenv("HF_TOKEN")
        self.qrels_file_base = qrels_file
        self.database_mapping = database_mapping

    @traceable(name="Embed Query")
    async def embed_query(self, query, route):
        run_tree = get_current_run_tree()
        run_tree.extra = {"metadata": {"route": route}}
        return await self.embedder_map[route].embed_query(query)

    @abstractmethod
    @traceable(name="router_selection")
    async def router(self, query):
        pass

    @traceable(name="Search")
    async def search(self, query_embedding, route):
        run_tree = get_current_run_tree()
        run_tree.extra = {"metadata": {"route": route}}
        return await self.database_mapping[route].search(query_embedding)

    @traceable(name="Process Query")
    async def process_query(self, query):
        """Process individual query for retrieval."""

        async with self.semaphore:
            selected_route = await self.router(query)
            
            query_embedding = await self.embed_query(query, selected_route)
            
            retrieved = await self.search(query_embedding, selected_route)
            
            results_dict = {
                result.metadata["filename"]: result.score 
                for result in retrieved.results
            }
            
            self.qrels[query] = {
                "selected_route": selected_route,
                "results": results_dict
            }

            with open(self.qrels_file, "w") as f:
                json.dump(self.qrels, f, indent=4)

    @traceable(name="Load Dataset")
    def load_dataset(self):
        dataset = load_dataset(self.task, token=self.hf_token, split="test")
        return dataset

    async def process_all_queries(self, query_column):
        """Process all queries from the specified Hugging Face dataset split."""
        self.qrels_file = os.path.join(self.qrels_file_base, f"{query_column}.json")

        if os.path.exists(self.qrels_file):
            with open(self.qrels_file, "r") as f:
                self.qrels = json.load(f)
        else:
            with open(self.qrels_file, "w") as f:
                json.dump({}, f, indent=4)
            self.qrels = {}

        dataset = self.load_dataset()
        
        tasks = []
        for row in dataset: 
            if row[query_column] is not None and row[query_column] not in self.qrels.keys():
                tasks.append(self.process_query(row[query_column])) 
        
        if tasks:
            await asyncio.gather(*tasks)    

    @traceable(name=f"Total Retrieval Duration", tags=["retrieval", os.getenv("TASK"), os.getenv("EMBEDDER_NAME"), os.getenv("PIPELINE_NAME"), os.getenv("DATABASE_NAME")])
    async def __call__(self):
        """Main entry point to run the retrieval process."""
        for query_column in ["query", "rephrase_level_1", "rephrase_level_2", "rephrase_level_3"]:
            for database in self.database_mapping.values():
                await database.initialize_collection()

            await self.process_all_queries(query_column) 