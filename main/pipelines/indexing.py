import os
import asyncio
import json
from typing import Dict, Any, TypedDict
from langgraph.graph import StateGraph, START, END
from main.vector_dabases import BaseVectorDatabase
from tqdm.asyncio import tqdm
from langsmith import traceable
from main.pipelines import database_mapping, embedder_mapping, TASK
from main.dataset_loader import load_dataset_for_benchmark

class SingleDocumentState(TypedDict):
    """State for single document processing workflow"""
    filename: str
    image: Any
    chunks: Dict[str, Any]
    metadata: Dict[str, Any]
    embedding: Any
    processed: bool 

class SingleIndexingTask:
    """Handles indexing for a single route/database combination"""
    
    def __init__(   
        self,
        route: str,
        database: BaseVectorDatabase,
        embedder: Any,
        semaphore: asyncio.Semaphore = None,
        progress_bar: tqdm = None
    ):
        self.route = route
        self.database = database
        self.embedder = embedder
        self.semaphore = semaphore or asyncio.Semaphore(30)
        self.progress_bar = progress_bar

    def create_document_graph(self):
        """Create a LangGraph for processing a single document with single route"""
        
        def start_processing(state: SingleDocumentState) -> SingleDocumentState:
            """Initialize document processing"""
            state["metadata"] = {"filename": state["filename"]}
            state["embedding"] = None
            state["processed"] = False
            return state

        def should_preprocess(state: SingleDocumentState) -> str:
            """Determine if preprocessing is needed"""
            if self.route.upper().startswith("TEXT"):
                return "preprocess_document"
            else:
                return "embed_document"

        async def preprocess_document(state: SingleDocumentState) -> SingleDocumentState:
            """Preprocess document for text routes"""
            state["image"] = state["chunks"]
            return state

        async def embed_document(state: SingleDocumentState) -> SingleDocumentState:
            """Embed the document using the embedder"""
            # Handle different input types after preprocessing
            if isinstance(state["image"], list):
                embeddings = await self.embedder.embed_documents(state["image"])
                state["embedding"] = embeddings
            else:
                # Visual route: state["image"] contains image object
                embedding = await self.embedder.embed_document(state["image"])
                state["embedding"] = embedding
            return state

        async def index_document(state: SingleDocumentState) -> SingleDocumentState:
            """Index the embedding to the database"""
            embedding = state["embedding"]
            metadata = state["metadata"]
            
            if self.route.upper().startswith("TEXT"):
                # Handle text embeddings (multiple chunk)
                for i, emb in enumerate(embedding):
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk_id"] = i
                    await self.database.index_document(emb, chunk_metadata)
            else:
                # Handle visual embeddings (single embedding)
                await self.database.index_document(embedding, metadata)
            
            state["processed"] = True
            return state

        def finish_processing(state: SingleDocumentState) -> SingleDocumentState:
            """Complete document processing"""
            if self.progress_bar:
                self.progress_bar.update(1)
                self.progress_bar.set_postfix({"file": state["filename"]})
            return state

        # Create the graph
        workflow = StateGraph(SingleDocumentState)
        
        # Add nodes
        workflow.add_node("start_processing", start_processing)
        workflow.add_node("preprocess_document", preprocess_document)
        workflow.add_node("embed_document", embed_document)
        workflow.add_node("index_document", index_document)
        workflow.add_node("finish_processing", finish_processing)
        
        # Add edges
        workflow.add_edge(START, "start_processing")
        workflow.add_conditional_edges(
            "start_processing",
            should_preprocess,
            {
                "preprocess_document": "preprocess_document",
                "embed_document": "embed_document"
            }
        )
        workflow.add_edge("preprocess_document", "embed_document")
        workflow.add_edge("embed_document", "index_document")
        workflow.add_edge("index_document", "finish_processing")
        workflow.add_edge("finish_processing", END)
        
        return workflow.compile()

    @traceable(name="process_document", tags=["single_indexing_task"])
    async def process_document(self, filename: str, image: Any, chunks: Dict[str, Any]) -> None:
        """Process a single document using LangGraph"""
        async with self.semaphore:
            graph = self.create_document_graph()
            
            try:
                initial_state = SingleDocumentState(
                    filename=filename,
                    image=image,
                    chunks=chunks,
                    metadata={},
                    embedding=None,
                    processed=False
                )
                
                final_state = await graph.ainvoke(initial_state)
                return final_state
            
            except Exception as e:
                print(f"Error processing document {filename}: {e}")
                return None
class BaseIndexing:
    def __init__(
        self,
        **kwargs 
    ):
        self.semaphore = asyncio.Semaphore(10) 
        self.embedder_map = embedder_mapping

        chunks_path = os.path.join(os.path.dirname(__file__), "..", "preprocessing", "chunks", f"{TASK}_chunks.json")
        with open(chunks_path, "r") as f:
            self.chunks = json.load(f)

        self.database_mapping = database_mapping

    async def process_all_files(self, indexed_files: set):
        """Process all not indexed files from the dataset using a single route"""
        
        dataset = load_dataset_for_benchmark(os.getenv("DATASET"))

        all_keys = set(dataset["image_filename"])
        
        # Get single route
        route = os.getenv("PIPELINE_NAME").lower()
        database = self.database_mapping[route]
        
        keys_to_process = all_keys - indexed_files
            
        total_docs = len(keys_to_process)
        
        if total_docs == 0:
            return
            
        with tqdm(total=total_docs, desc="Processing documents", unit="docs") as pbar:
            tasks = []
            
            if keys_to_process:
                embedder = self.embedder_map[route]

                indexing_task = SingleIndexingTask(
                    route=route,
                    database=database,
                    embedder=embedder,
                    semaphore=self.semaphore,
                    progress_bar=pbar
                )
                
                for row in dataset:
                    if row["image_filename"] in keys_to_process:
                        keys_to_process.remove(row["image_filename"])
                        task = indexing_task.process_document(row["image_filename"], row["image"], self.chunks.get(row["image_filename"], {}).get("chunks", []))
                        tasks.append(task)
            
            if tasks:
                await asyncio.gather(*tasks)

    @traceable(name=f"indexing", tags=["indexing", TASK, os.getenv("PIPELINE_NAME")])
    async def __call__(self):
        """Main method that orchestrates the entire indexing process"""
        
        route = list(self.database_mapping.keys())[0]
        
        database = self.database_mapping[route]
        await database.initialize_collection()
        
        indexed_files = await database.get_indexed_files()    
        await self.process_all_files(indexed_files)

if __name__ == "__main__":
    indexing = BaseIndexing()
    asyncio.run(indexing())