import hashlib
import math
import re

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage

from rag_langchain.config import Settings
from rag_langchain.service import RAGService


class LocalEmbeddings(Embeddings):
    """Deterministic bag-of-words test vectors; no network or downloaded model."""

    def __init__(self):
        self.calls = 0
        self.fail = False

    def vector(self, text):
        vector = [0.0] * 128
        for token in re.findall(r"\w+", text.lower()):
            position = int(hashlib.sha256(token.encode()).hexdigest(), 16) % len(vector)
            vector[position] += 1
        norm = math.sqrt(sum(v * v for v in vector)) or 1
        return [v / norm for v in vector]

    def embed_documents(self, texts):
        self.calls += len(texts)
        if self.fail:
            raise RuntimeError("Simulated provider outage")
        return [self.vector(text) for text in texts]

    def embed_query(self, text):
        return self.vector(text)


class ChatStub:
    def __init__(self):
        self.messages = []
        self.content = "Alice follows the rabbit. [1]"

    def invoke(self, messages):
        self.messages = messages
        return AIMessage(content=self.content)


@pytest.fixture
def setup(tmp_path):
    data = tmp_path / "books"
    data.mkdir()
    (data / "alice.md").write_text("Alice follows a white rabbit into Wonderland.")
    (data / "physics.txt").write_text("Electric current is measured in amperes.")
    settings = Settings(
        data_path=data, db_path=tmp_path / "index", chunk_size=120, chunk_overlap=20
    )
    embeddings, chat = LocalEmbeddings(), ChatStub()
    service = RAGService(settings, embeddings=embeddings, chat=chat)
    return service, embeddings, chat, data
