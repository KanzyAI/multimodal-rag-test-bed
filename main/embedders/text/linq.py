import torch
from sentence_transformers import SentenceTransformer
from ..base_embedder import BaseEmbedder, EmbedderConfig, EmbedderFactory

class LinqEmbedder(BaseEmbedder):
    def __init__(self, config: EmbedderConfig):
        # Set device in config if not provided
        if config.device is None:
            config.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        super().__init__(config)
        
    def _initialize_model(self):
        self.model = SentenceTransformer("Linq-AI-Research/Linq-Embed-Mistral").to(self.device)
        task = 'Given a question, retrieve passages that answer the question'
        self.prompt = f"Instruct: {task}\nQuery: "
        
    async def embed_document(self, document):
        return self.model.encode(document)

    async def embed_query(self, query):
        return self.model.encode(query, prompt=self.prompt)

EmbedderFactory.register_embedder("linq", LinqEmbedder)
