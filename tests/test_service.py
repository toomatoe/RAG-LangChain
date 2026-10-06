from dataclasses import replace

import pytest
from pypdf import PdfWriter

from rag_langchain.documents import chunks_for, discover, load
from rag_langchain.service import RAGService


def test_ingest_persist_search_and_noop(setup):
    service, embeddings, _, _ = setup
    assert not service.status()["indexed"]
    assert service.ingest()["chunks"] == 2
    assert embeddings.calls == 2
    assert service.ingest()["embedded_chunks"] == 0
    assert embeddings.calls == 2
    reopened = RAGService(service.settings, embeddings=embeddings)
    assert reopened.status()["files"] == 2
    hits = reopened.search("Alice rabbit", top_k=1)
    assert hits[0].source == "alice.md"
    assert "rabbit" in hits[0].excerpt


def test_update_and_delete_reuse_vectors(setup):
    service, embeddings, _, data = setup
    service.ingest()
    (data / "alice.md").write_text("Alice drinks tea with the Hatter.")
    result = service.ingest()
    assert result["embedded_chunks"] == 1
    assert embeddings.calls == 3
    (data / "physics.txt").unlink()
    result = service.ingest()
    assert result["removed_files"] == 1
    assert result["chunks"] == 1
    assert embeddings.calls == 3
    assert service.search("amperes")[0].source == "alice.md"
    assert len(service.client.list_collections()) == 1


def test_failed_update_keeps_previous_index(setup):
    service, embeddings, _, data = setup
    service.ingest()
    previous = service.manifest_path.read_text()
    (data / "alice.md").write_text("Changed content.")
    embeddings.fail = True
    with pytest.raises(RuntimeError, match="outage"):
        service.ingest()
    assert service.manifest_path.read_text() == previous
    assert len(service.client.list_collections()) == 1
    assert "white rabbit" in service.search("Alice", top_k=1)[0].excerpt


def test_failed_rebuild_keeps_previous_index(setup):
    service, embeddings, _, _ = setup
    service.ingest()
    changed = RAGService(
        replace(service.settings, embedding_model="different"), embeddings=embeddings
    )
    with pytest.raises(ValueError, match="settings changed"):
        changed.ingest()
    embeddings.fail = True
    with pytest.raises(RuntimeError):
        changed.ingest(rebuild=True)
    assert service.status()["chunks"] == 2


def test_rebuild_switches_settings_and_source(setup, tmp_path):
    service, embeddings, _, _ = setup
    service.ingest()
    other = tmp_path / "other.txt"
    other.write_text("The moon orbits Earth.")
    changed = RAGService(
        replace(service.settings, data_path=other, chunk_size=100), embeddings=embeddings
    )
    with pytest.raises(ValueError):
        changed.ingest()
    result = changed.ingest(rebuild=True)
    assert result["chunks"] == 1
    assert changed.status()["sources"] == ["other.txt"]


def test_answer_uses_sources_and_budget(setup):
    service, _, chat, _ = setup
    service.ingest()
    service.settings = replace(service.settings, max_context_chars=100)
    result = service.ask("What does Alice follow?")
    assert result.answer.endswith("[1]")
    assert result.sources[0].source == "alice.md"
    assert "untrusted data" in chat.messages[0].content
    context = chat.messages[1].content.split("Document excerpts:\n")[1].split("\n\nQuestion:")[0]
    assert len(context) <= 100
    assert result.to_dict()["sources"][0]["chunk_id"]
    chat.content = [{"type": "text", "text": "A rabbit. [1]"}]
    assert service.ask("Alice?").answer == "A rabbit. [1]"


def test_invalid_queries_and_missing_index(setup):
    service, _, _, _ = setup
    with pytest.raises(ValueError, match="No index"):
        service.ask("Hello")
    with pytest.raises(ValueError, match="nonempty"):
        service.search(" ")
    with pytest.raises(ValueError, match="top_k"):
        service.search("Hello", top_k=0)


def test_bad_document_does_not_replace_index(setup):
    service, _, _, data = setup
    service.ingest()
    previous = service.manifest_path.read_text()
    (data / "broken.pdf").write_bytes(b"invalid")
    with pytest.raises(ValueError, match="Unable to read"):
        service.ingest()
    assert service.manifest_path.read_text() == previous


def test_empty_corpus_preserves_previous_index(setup):
    service, _, _, data = setup
    service.ingest()
    for path in data.iterdir():
        path.unlink()
    with pytest.raises(ValueError, match="No PDF"):
        service.ingest()
    assert service.status()["chunks"] == 2


def test_loader_discovery_metadata_and_chunking(tmp_path, setup):
    service, _, _, _ = setup
    path = tmp_path / "notes.TXT"
    path.write_text("word " * 100, encoding="utf-8")
    first = chunks_for(path, "notes.TXT", service.settings)
    second = chunks_for(path, "notes.TXT", service.settings)
    assert len(first) > 1
    assert all(len(c.page_content) <= 120 for c in first)
    assert first[1].metadata["start_index"] > 0
    assert [c.metadata["chunk_id"] for c in first] == [c.metadata["chunk_id"] for c in second]
    (tmp_path / ".hidden.md").write_text("secret")
    assert (tmp_path / ".hidden.md") not in discover(tmp_path, service.settings.db_path)
    blank = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.write(blank)
    with pytest.raises(ValueError, match="OCR"):
        load(blank, "blank.pdf")


def test_pdf_page_numbers(tmp_path):
    # Tiny self-contained PDF with extractable text; no fixture binary required.
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    path = tmp_path / "text.pdf"
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
    )
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 20 250 Td (Hello PDF) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    writer.write(path)
    docs = load(path, "text.pdf")
    assert docs[0].metadata["page"] == 1
    assert "Hello PDF" in docs[0].page_content


def test_no_passage_fits_budget_does_not_call_chat(setup):
    service, _, chat, _ = setup
    service.ingest()
    service.settings = replace(service.settings, max_context_chars=100)
    from rag_langchain.service import Source

    service.search = lambda *args: [Source(1, "a" * 200, None, "id", 0.0, "content")]
    result = service.ask("hello")
    assert result.sources == []
    assert "budget" in result.answer
    assert chat.messages == []
