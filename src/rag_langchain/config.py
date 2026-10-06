"""Validated settings shared by the CLI and web interface."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    data_path: Path = Path("data/books")
    db_path: Path = Path("data/chroma_db")
    embedding_model: str = "gemini-embedding-001"
    chat_model: str = "gemini-2.5-flash"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    max_context_chars: int = 16000

    def __post_init__(self):
        if self.chunk_size < 1 or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError(
                "Chunk size must be positive; overlap must be smaller than chunk size."
            )
        if not 1 <= self.top_k <= 50:
            raise ValueError("top_k must be between 1 and 50.")
        if self.max_context_chars < 100:
            raise ValueError("Context budget must be at least 100 characters.")
        data, db = self.data_path.resolve(), self.db_path.resolve()
        if db == data or db in data.parents:
            raise ValueError("The database directory cannot contain the document source.")

    @classmethod
    def from_env(cls):
        load_dotenv()
        return cls(
            data_path=Path(os.getenv("RAG_DATA_PATH", "data/books")),
            db_path=Path(os.getenv("RAG_DB_PATH", "data/chroma_db")),
            embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
            chat_model=os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash"),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "200")),
            top_k=int(os.getenv("RAG_TOP_K", "5")),
            max_context_chars=int(os.getenv("RAG_MAX_CONTEXT_CHARS", "16000")),
        )


def api_key():
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("Set GOOGLE_API_KEY in .env before indexing or asking questions.")
    return key


def make_embeddings(settings):
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model, google_api_key=api_key(), vertexai=False
    )


def make_chat(settings):
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.chat_model,
        google_api_key=api_key(),
        vertexai=False,
        temperature=0,
        max_retries=2,
        timeout=60,
    )
