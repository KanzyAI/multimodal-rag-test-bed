import os
from main.embedders import EmbedderFactory, EmbedderConfig
from main.pipelines.base.base_indexing import BaseIndexing

class ColpaliIndexing(BaseIndexing):
    def __init__(
        self,
        task,
        database_mapping,
        model_name, 
        model_type,
        **kwargs
    ):
        super().__init__(
            task=task, 
            database_mapping=database_mapping,
            **kwargs
        )
        
        self.embedder_map = {
            f"visual-{model_type}": EmbedderFactory.create_embedder(os.getenv("MULTIMODAL_DENSE_EMBEDDER"), EmbedderConfig(model_name=model_name))
        }

