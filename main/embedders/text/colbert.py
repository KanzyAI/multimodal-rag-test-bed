import aiohttp
import os
from typing import List, Union
from ..base_embedder import BaseEmbedder, EmbedderConfig, EmbedderFactory


class ColbertEmbedder(BaseEmbedder):
    """Jina ColBERT v2 embedder using Jina AI's API for multilingual multi-vector text embedding with 8k context length."""
    
    def __init__(self, config):
        if isinstance(config, str):
            config = EmbedderConfig(model_name=config)
        
        self.embedder_type = "text"
        self.vector_size = 128  
        self.api_url = "https://api.jina.ai/v1/multi-vector"
        super().__init__(config)
        
    def _initialize_model(self):
        self.model_name = "jina-colbert-v2"
        self.api_key = os.environ.get('JINA_API_KEY')
        self.dimensions = 128

    async def _make_api_request(self, texts: List[str], input_type: str) -> List[List[List[float]]]:
        """Make request to Jina AI multi-vector API."""
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        data = {
            'model': self.model_name,
            'input_type': input_type,
            'embedding_type': 'float',
            'dimensions': self.dimensions,
            'input': texts
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, headers=headers, json=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Jina API error ({response.status}): {error_text}")
                
                result = await response.json()
                
                embeddings = []
                for item in result['data']:
                    embeddings.append(item['embeddings'])
                
                return embeddings

    async def embed_documents(self, texts: Union[str, List[str]]) -> List[List[List[float]]]:
        """Embed multiple documents using Jina AI API."""
        if isinstance(texts, str):
            texts = [texts]
        
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = await self._make_api_request(batch, 'document')
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    async def embed_document(self, document: str) -> List[float]:
        embeddings = await self.embed_documents([document])
        return embeddings[0]

    async def embed_query(self, query: str) -> List[List[float]]:
        """Embed a single query using Jina AI API."""
        embeddings = await self._make_api_request([query], 'query')
        return embeddings[0]


# Register embedder with factory
EmbedderFactory.register_embedder("colbert", ColbertEmbedder)