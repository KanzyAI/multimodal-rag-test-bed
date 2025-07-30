# from pylate import indexes, models, retrieve
# from ..base_embedder import BaseEmbedder, EmbedderConfig, EmbedderFactory


# class ColbertEmbedder(BaseEmbedder):
#     """Jina ColBERT v2 embedder for multilingual multi-vector text embedding with 8k context length."""
    
#     def __init__(self, config):
#         if isinstance(config, str):
#             config = EmbedderConfig(model_name=config)
        
#         self.embedder_type = "text"
#         self.vector_size = 128  # Jina ColBERT v2 uses 128 dimensions (also supports 96, 64 with Matryoshka)
#         super().__init__(config)
        
#     def _initialize_model(self):
#         """Initialize the Jina ColBERT v2 model."""
#         model_name = self.config.model_name or "jinaai/jina-colbert-v2"
        
#         # Initialize with proper configuration for Jina ColBERT v2
#         self.embedder = models.ColBERT(
#             model_name_or_path=model_name,
#             query_prefix="[QueryMarker]",
#             document_prefix="[DocumentMarker]",
#             attend_to_expansion_tokens=True,
#             trust_remote_code=True,
#         )

#     async def embed_documents(self, texts):
#         """Embed multiple documents."""
#         batch_size = self.config.batch_size or 32
        
#         all_embeddings = self.embedder.encode(
#             texts,
#             batch_size=batch_size,
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