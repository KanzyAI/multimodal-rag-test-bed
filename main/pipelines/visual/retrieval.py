import os
from main.pipelines.base.base_retrieval import BaseRetrieval
from main.embedders import EmbedderFactory, EmbedderConfig

class ColpaliRetrieval(BaseRetrieval):
    def __init__(
        self,
        task,
        database_mapping,
        qrels_file,
        model_name, 
        model_type,
        **kwargs
    ):
        super().__init__(
            task=task, 
            qrels_file=qrels_file,
            database_mapping=database_mapping,
            **kwargs
        )

        self.embedder_map = {
            f"visual-{model_type}": EmbedderFactory.create_embedder(os.getenv("MULTIMODAL_EMBEDDER"), EmbedderConfig(model_name=model_name))
        }

    async def router(self, query):
        return f"visual-{self.model_type}"