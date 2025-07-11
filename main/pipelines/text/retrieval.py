import os
from main.pipelines.base.base_retrieval import BaseRetrieval
from main.embedders import EmbedderFactory, EmbedderConfig

class TextRetrieval(BaseRetrieval):
    def __init__(
        self,
        task,
        qrels_file,
        model_name,
        model_type,
        database_mapping,
        **kwargs
    ):
        super().__init__(
            task=task, 
            database_mapping=database_mapping,
            qrels_file=qrels_file,
            **kwargs
        )

        self.embedder_map = {
            f"text-{model_type}": EmbedderFactory.create_embedder(os.getenv("TEXT_EMBEDDER"), EmbedderConfig(model_name=model_name))
        }

    async def router(self, query):
        return f"text-{self.model_type}"