from __future__ import annotations

import os
import bentoml
from bentoml.io import File, JSON, Text
from llama_index.core import Document, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import SimpleNodeParser
from pathlib import Path
from typing import Annotated
import openai
from embedding import SentenceTransformers, BentoMLEmbeddings

PERSIST_DIR = "./storage"
openai.api_key = "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"


class RAGService:
    embedding_service = bentoml.depends(SentenceTransformers)

    def __init__(self):
        self.embed_model = BentoMLEmbeddings(self.embedding_service)

        from llama_index.core import ServiceContext
        self.node_parser = SimpleNodeParser(chunk_size=1024, chunk_overlap=20)
        ServiceContext.node_parser = self.node_parser
        ServiceContext.embed_model = self.embed_model

        from transformers import AutoTokenizer
        ServiceContext.num_output = 256
        ServiceContext.context_window = 4096
        ServiceContext.tokenizer = AutoTokenizer.from_pretrained("gpt2")

        index = VectorStoreIndex.from_documents([])
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        self.index = load_index_from_storage(storage_context)

    def ingest_pdf(self, pdf: Annotated[Path, File()]) -> str:
        import pypdf
        reader = pypdf.PdfReader(pdf)
        texts = []
        for page in reader.pages:
            text = page.extract_text()
            texts.append(text)
        all_text = "".join(texts)
        doc = Document(text=all_text)
        self.index.insert(doc)
        self.index.storage_context.persist(persist_dir=PERSIST_DIR)
        return "Successfully Loaded Document"

    def ingest_text(self, txt: Annotated[Path, File()]) -> str:
        with open(txt) as f:
            text = f.read()
        doc = Document(text=text)
        self.index.insert(doc)
        self.index.storage_context.persist(persist_dir=PERSIST_DIR)
        return "Successfully Loaded Document"

    def query(self, query: str) -> str:
        from llama_index.core import ServiceContext

        base_url = "https://api.openai.com/v1"
        llm = openai.Completion.create(
            model="gpt2",
            prompt=query,
            max_tokens=150
        )
        ServiceContext.llm = llm

        query_engine = self.index.as_query_engine()
        response = query_engine.query(query)
        return str(response)

svc = bentoml.Service("rag_service")

svc.api(input=File(), output=Text())(RAGService().ingest_pdf)
svc.api(input=File(), output=Text())(RAGService().ingest_text)
svc.api(input=Text(), output=Text())(RAGService().query)