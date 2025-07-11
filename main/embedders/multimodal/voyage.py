import voyageai
from PIL import Image
from ..base_embedder import BaseEmbedder, EmbedderFactory


class VoyageMultimodalEmbedder(BaseEmbedder):

    def _initialize_model(self):
        self.client = voyageai.Client()
        self.model_name = "voyage-multimodal-3"

    async def embed_document(self, document):
        result = self.client.multimodal_embed(
            inputs=[[document]],
            model=self.model_name,
            input_type="document"
        )
        return result.embeddings[0]

    async def embed_query(self, query):
        result = self.client.multimodal_embed(
            inputs=[query],
            model=self.model_name,
            input_type="query"
        )
        return result.embeddings[0]


EmbedderFactory.register_embedder("voyage", VoyageMultimodalEmbedder)
