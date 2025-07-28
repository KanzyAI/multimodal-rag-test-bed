import os
import json
import random
import asyncio
from datasets import load_dataset
from typing import Dict, Any, TypedDict
from langgraph.graph import StateGraph, START, END
from tqdm.asyncio import tqdm
from main.vector_dabases import BaseVectorDatabase
from langsmith import traceable
from main.pipelines import database_mapping, embedder_mapping, TASK

class SingleQueryState(TypedDict):
    """State for single query processing workflow"""
    query: str
    route: str
    query_embedding: Any
    retrieved_results: Any
    results_dict: Dict[str, float]
    processed: bool


class SingleRetrievalTask:
    """Handles retrieval for a single query with a single route/database"""
    
    def __init__(
        self,
        router,
        database_mapping: Dict[str, BaseVectorDatabase],
        embedder_map: Dict[str, Any],
        semaphore: asyncio.Semaphore = None,
        progress_bar: tqdm = None
    ):
        self.router = router
        self.database_mapping = database_mapping
        self.embedder_map = embedder_map
        self.semaphore = semaphore
        self.progress_bar = progress_bar

    def create_query_graph(self):
        """Create a LangGraph for processing a single query retrieval"""
        
        def start_processing(state: SingleQueryState) -> SingleQueryState:
            """Initialize query processing"""
            state["route"] = self.route
            state["query_embedding"] = None
            state["retrieved_results"] = None
            state["results_dict"] = {}
            state["processed"] = False
            return state

        async def classify_query(state: SingleQueryState) -> SingleQueryState:
            """Classify the query into a route"""

            if self.router is None:
                route = list(self.database_mapping.keys())[0]
            else:
                route = self.router(state["query"])
                
            state["route"] = route
            state["embedder"] = self.embedder_map[route]
            state["database"] = self.database_mapping[route]
            return state

        async def embed_query(state: SingleQueryState) -> SingleQueryState:
            """Embed the query using the embedder"""
            query_embedding = await self.embedder.embed_query(state["query"])
            state["query_embedding"] = query_embedding
            return state

        async def search_database(state: SingleQueryState) -> SingleQueryState:
            """Search the database with the query embedding"""
            retrieved = await self.database.search(state["query_embedding"])
            state["retrieved_results"] = retrieved
            return state

        def process_results(state: SingleQueryState) -> SingleQueryState:
            """Process search results into results dictionary"""
            results_dict = {
                result.metadata["filename"]: result.score 
                for result in state["retrieved_results"].results
            }
            state["results_dict"] = results_dict
            state["processed"] = True
            return state

        def finish_processing(state: SingleQueryState) -> SingleQueryState:
            """Complete query processing"""
            if self.progress_bar:
                self.progress_bar.update(1)
                self.progress_bar.set_postfix({"query": state["query"][:50] + "..." if len(state["query"]) > 50 else state["query"]})
            return state

        # Create the graph
        workflow = StateGraph(SingleQueryState)
        
        # Add nodes
        workflow.add_node("start_processing", start_processing)
        workflow.add_node("classify_query", classify_query)
        workflow.add_node("embed_query", embed_query)
        workflow.add_node("search_database", search_database)
        workflow.add_node("process_results", process_results)
        workflow.add_node("finish_processing", finish_processing)
        
        # Add edges
        workflow.add_edge(START, "start_processing")
        workflow.add_edge("start_processing", "classify_query")
        workflow.add_edge("classify_query", "embed_query")
        workflow.add_edge("start_processing", "embed_query")
        workflow.add_edge("embed_query", "search_database")
        workflow.add_edge("search_database", "process_results")
        workflow.add_edge("process_results", "finish_processing")
        workflow.add_edge("finish_processing", END)
        
        return workflow.compile()

    @traceable(name="process_query", tags=["query_processing"])
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process a single query using LangGraph"""
        async with self.semaphore:
            # Create the graph for this query
            graph = self.create_query_graph()
            
            # Initialize state
            initial_state = SingleQueryState(
                query=query,
                route="",
                query_embedding=None,
                retrieved_results=None,
                results_dict={},
                processed=False
            )
            
            # Execute the graph
            final_state = await graph.ainvoke(initial_state)
            
            return {
                "selected_route": final_state["route"],
                "results": final_state["results_dict"]
            }


class BaseRetrieval():
    def __init__(
        self,
        **kwargs 
    ):        
        self.semaphore = asyncio.Semaphore(2) 
        self.hf_token = os.getenv("HF_TOKEN")
        self.qrels_file_base = os.path.join(os.getcwd(), f"results/{TASK}/{os.getenv('PIPELINE_NAME')}/")
        self.database_mapping = database_mapping
        self.embedder_map = embedder_mapping

        os.makedirs(self.qrels_file_base, exist_ok=True)

    async def process_all_queries(self, query_column: str):
        """Process all queries from the specified query column using parallel SingleRetrievalTask instances"""
        self.qrels_file = os.path.join(self.qrels_file_base, f"{query_column}.json")

        if os.path.exists(self.qrels_file):
            with open(self.qrels_file, "r") as f:
                self.qrels = json.load(f)
        else:
            with open(self.qrels_file, "w") as f:
                json.dump({}, f, indent=4)
            self.qrels = {}

        dataset = load_dataset(os.getenv("DATASET"), token=self.hf_token, split="test")
        
        queries_to_process = []
        for row in dataset:
            if row[query_column] is not None and row[query_column] not in self.qrels.keys():
                queries_to_process.append(row[query_column])
        
        if len(queries_to_process) > 10:
            queries_to_process = random.sample(queries_to_process, 10)
            
        # Create progress bar
        with tqdm(total=len(queries_to_process), desc=f"Processing {query_column} queries", unit="queries") as pbar:
            # Create SingleRetrievalTask
            retrieval_task = SingleRetrievalTask(
                database_mapping=self.database_mapping,
                embedder_map=self.embedder_map,
                semaphore=self.semaphore,
                progress_bar=pbar
            )
            
            # Create tasks for all queries
            tasks = []
            for query in queries_to_process:
                task = retrieval_task.process_query(query)
                tasks.append((query, task))
            
            # Execute all tasks in parallel
            if tasks:
                results = await asyncio.gather(*[task for _, task in tasks])
                
                # Save results
                for (query, _), result in zip(tasks, results):
                    self.qrels[query] = result
                
                # Write results to file
                with open(self.qrels_file, "w") as f:
                    json.dump(self.qrels, f, indent=4)

    @traceable(name=f"retrieval_pipeline", tags=["retrieval", TASK, os.getenv("PIPELINE_NAME")])
    async def __call__(self):
        """Main entry point to run the retrieval process."""
        # Initialize database
        for database in self.database_mapping.values():
            await database.initialize_collection()
        
        # Process all query columns
        for query_column in ["query", "rephrase_level_1", "rephrase_level_2", "rephrase_level_3"]:
            await self.process_all_queries(query_column) 

if __name__ == "__main__":
    retrieval = BaseRetrieval()
    asyncio.run(retrieval())