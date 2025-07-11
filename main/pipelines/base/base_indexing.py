import os
import asyncio
import random
from datasets import load_dataset
from langsmith import traceable, get_current_run_tree
from typing import Dict
from main.vector_dabases import BaseVectorDatabase

class BaseIndexing():
    def __init__(
        self,
        task: str,
        database_mapping: Dict[str, BaseVectorDatabase],
        **kwargs 
    ):
        self.task = task
        self.preprocessor = None
        self.database_mapping = database_mapping
        self.semaphore = asyncio.Semaphore(10) 
        self.hf_token = os.getenv("HF_TOKEN")
        self.dataset = None

    @traceable(name="Preprocess the document")
    async def preprocess(self, image, filename):
        run_tree = get_current_run_tree()
        run_tree.extra = {"metadata": {"filename": filename}}        
        return await self.preprocessor(image, filename)

    @traceable(name="Embed Document")
    async def embed_document(self, document, route, filename):
        run_tree = get_current_run_tree()
        run_tree.extra = {"metadata": {"route": route, "filename": filename}}

        if route.startswith("text"):
            document = await self.preprocess(document, route)
        return await self.embedder_map[route].embed_document(document)

    @traceable(name="Index Document")
    async def index_document(self, embedding, metadata, route):
        run_tree = get_current_run_tree()
        run_tree.extra = {"metadata": {"filename": metadata["filename"], "route": route}}

        if route.startswith("text"):
            for i, emb in enumerate(embedding):
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_id"] = i
                await self.database_mapping[route].index_document(emb, chunk_metadata)
        else:
            return await self.database_mapping[route].index_document(embedding, metadata)

    @traceable(name="Process Document")
    async def process_file(self, filename, image):
        run_tree = get_current_run_tree()
        run_tree.extra = {"metadata": {"filename": filename}}
            
        async with self.semaphore:
            metadata = {"filename": filename}
            embedding_tasks = [self.embed_document(image, key, filename) for key in self.embedder_map.keys()]
            embeddings = await asyncio.gather(*embedding_tasks)
            embedding_map = {key: emb for key, emb in zip(self.embedder_map.keys(), embeddings)}
            indexing_tasks = [self.index_document(embedding, metadata, key) for key, embedding in embedding_map.items()]
            await asyncio.gather(*indexing_tasks)

    @traceable(name="Load Dataset")
    def load_dataset(self):
        dataset = load_dataset(self.task, token=self.hf_token, split="test")
        self.dataset = dataset
    
    async def process_all_files(self, indexed_files):
        """Process all not indexed files from the specified Hugging Face dataset split."""
        all_keys = set(self.dataset["image_filename"])
        keys_to_process = all_keys - indexed_files
        
        if len(keys_to_process) > 10:
            keys_to_process = random.sample(list(keys_to_process), 10)
        else:
            keys_to_process = list(keys_to_process)

        tasks = []
        for row in self.dataset: 
            if row["image_filename"] in keys_to_process:
                keys_to_process.remove(row["image_filename"])
                tasks.append(self.process_file(row["image_filename"], row["image"])) 
        
        if tasks:
            await asyncio.gather(*tasks) 

    @traceable(name=f"Total Indexing Duration", tags=["indexing", os.getenv("TASK"), os.getenv("EMBEDDER_NAME"), os.getenv("PIPELINE_NAME"), os.getenv("DATABASE_NAME")])
    async def __call__(self):

        self.load_dataset()
        indexed_files = set()

        for database in self.database_mapping.values():
            await database.initialize_collection() 
            indexed_files.update(await database.get_indexed_files())
            
        await self.process_all_files(indexed_files)