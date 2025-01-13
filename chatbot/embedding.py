from __future__ import annotations

import bentoml
from sentence_transformers import SentenceTransformer
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

@bentoml.service(
    traffic={"timeout": 600},
    resources={"memory": "2Gi"},
)
class SentenceTransformers:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_ID)

    def encode(self, sentences):
        return self.model.encode(sentences)

class BentoMLEmbeddings(BaseEmbedding):
    _model: bentoml.Service = PrivateAttr()

    def __init__(self, embed_model: bentoml.Service, **kwargs) -> None:
        super().__init__(**kwargs)
        self._model = embed_model

    def _get_query_embedding(self, query: str):
        response = self._model.encode(sentences=[query])
        return response[0].tolist()

    def _get_text_embedding(self, text: str):
        response = self._model.encode(sentences=[text])
        return response[0].tolist()

    async def _aget_query_embedding(self, query: str):
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str):
        return self._get_text_embedding(text)