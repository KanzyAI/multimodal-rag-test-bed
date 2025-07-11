# from pylate import indexes, models, retrieve
# from ..base_embedder import BaseEmbedder, EmbedderConfig, EmbedderFactory


# class ColbertEmbedder(BaseEmbedder):
#     """ColBERT embedder for multi-vector text embedding."""
    
#     def __init__(self, config):
#         if isinstance(config, str):
#             config = EmbedderConfig(model_name=config)
        
#         self.embedder_type = "text"
#         self.vector_size = 128  # ColBERT typically uses smaller vectors
#         super().__init__(config)
        
#     def _initialize_model(self):
#         """Initialize the ColBERT model."""
#         model_name = self.config.model_name or "lightonai/GTE-ModernColBERT-v1"
#         batch_size = self.config.batch_size or 32
        
#         self.embedder = models.ColBERT(model_name_or_path=model_name)

#     async def embed_documents(self, texts):
#         """Embed multiple documents."""
#         all_embeddings = self.embedder.encode(
#             texts,
#             batch_size=self.batch_size,
#             is_query=False,  
#             show_progress_bar=True,
#         )
        
#         return all_embeddings

#     async def embed_query(self, query):
#         """Embed a single query."""
#         embeddings = self.embedder.encode([query], is_query=True)
#         return embeddings[0]


# # Register embedder with factory
# EmbedderFactory.register_embedder("colbert", ColbertEmbedder)