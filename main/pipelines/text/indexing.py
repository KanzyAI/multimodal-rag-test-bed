import os
from main.pipelines.base.base_indexing import BaseIndexing
from main.pipelines.text.preprocessor import Preprocessor
from main.embedders import EmbedderFactory, EmbedderConfig

class TextIndexing(BaseIndexing):
    def __init__(
        self,
        task,
        model_name,
        model_type,
        database_mapping,
        **kwargs
    ):
        super().__init__(
            task=task, 
            database_mapping=database_mapping,
            **kwargs
        )

        self.preprocessor = Preprocessor(task, output_path=os.path.join(os.getcwd(), "main/text/chunks.json"))

        self.embedder_map = {
            f"text-{model_type}": EmbedderFactory.create_embedder(os.getenv("TEXT_EMBEDDER"), EmbedderConfig(model_name=model_name))
        }


