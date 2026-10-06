"""Portable PDF, Markdown, and UTF-8 text ingestion."""

import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

SUPPORTED = {".pdf", ".md", ".txt"}


def discover(path: Path, excluded: Path):
    if not path.exists():
        raise ValueError(f"Document path does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED:
            raise ValueError("Supported document types: .pdf, .md, .txt")
        return [path]
    files = [
        p
        for p in path.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED
        and excluded.resolve() not in p.resolve().parents
        and not any(part.startswith(".") for part in p.relative_to(path).parts)
        and not p.is_symlink()
    ]
    if not files:
        raise ValueError(f"No PDF, Markdown, or text documents found in {path}")
    return sorted(files)


def load(path: Path, source: str):
    try:
        if path.suffix.lower() == ".pdf":
            reader = PdfReader(path)
            docs = [
                Document(
                    page_content=page.extract_text() or "",
                    metadata={"source": source, "page": i + 1},
                )
                for i, page in enumerate(reader.pages)
            ]
        else:
            docs = [
                Document(
                    page_content=path.read_text(encoding="utf-8-sig"), metadata={"source": source}
                )
            ]
    except Exception as exc:
        raise ValueError(f"Unable to read {source}. Check encoding or PDF encryption.") from exc
    docs = [doc for doc in docs if doc.page_content.strip()]
    if not docs:
        raise ValueError(f"No extractable text in {source}. Scanned PDFs need OCR first.")
    return docs


def chunks_for(path: Path, source: str, settings):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(load(path, source))
    for i, chunk in enumerate(chunks):
        identity = f"{source}:{chunk.metadata.get('page', 0)}:{i}:{chunk.page_content}"
        chunk.metadata["chunk_id"] = hashlib.sha256(identity.encode()).hexdigest()
    return chunks
