"""Run with: streamlit run app.py."""

from dataclasses import asdict, replace
from pathlib import Path

import streamlit as st

from rag_langchain.config import Settings
from rag_langchain.service import RAGService

st.set_page_config(page_title="Document Q&A", page_icon="📚", layout="wide")
st.title("📚 Document Q&A")
st.caption(
    "Ask questions about your PDFs, books, and notes. Inspect the passages behind each answer."
)


def show_error(exc):
    if isinstance(exc, ValueError):
        st.error(str(exc))
    else:
        st.error(
            f"Operation failed ({type(exc).__name__}). Check your API key, model access, "
            "network connection, and database path."
        )


try:
    defaults = Settings.from_env()
    with st.sidebar:
        st.header("Your library")
        data = st.text_input("Document file or folder", str(defaults.data_path))
        db = st.text_input("Index folder", str(defaults.db_path))
        top_k = st.slider("Passages to retrieve", 1, 20, min(defaults.top_k, 20))
        st.caption("Configure your Gemini API key in .env before indexing.")
        rebuild = st.checkbox("Rebuild all embeddings", value=False)
        index = st.button("Index documents", type="primary")
        st.caption(
            "Indexing sends document text to Google. Questions use the same embedding model."
        )
    settings = replace(defaults, data_path=Path(data), db_path=Path(db), top_k=top_k)
    service = RAGService(settings)
    if index:
        with st.spinner("Indexing documents…"):
            result = service.ingest(rebuild)
        st.sidebar.success(
            f"{result['files']} files · {result['chunks']} passages · "
            f"{result['embedded_chunks']} newly embedded"
        )
    status = service.status()
    st.sidebar.metric("Indexed passages", status["chunks"])
    with st.sidebar.expander("Indexed files"):
        for source in status.get("sources", []):
            st.text(source)
    # Changing corpus or settings starts a fresh display history.
    identity = (
        str(settings.db_path.resolve()),
        service.manifest_path.read_text() if service.manifest_path.exists() else "",
        settings.chat_model,
    )
    if st.session_state.get("corpus") != identity:
        st.session_state.corpus = identity
        st.session_state.messages = []
    st.caption("Each question is answered independently. Include names and context in follow-ups.")
    for item in st.session_state.messages:
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            st.markdown(item["answer"])
            for source in item["sources"]:
                page = f" · page {source['page']}" if source["page"] else ""
                with st.expander(f"[{source['number']}] {source['source']}{page}"):
                    st.caption(f"Cosine distance: {source['distance']:.4f} (lower is closer)")
                    st.text(source["excerpt"])
    question = st.chat_input("What would you like to know?", disabled=not status["indexed"])
    if not status["indexed"]:
        st.info("Choose a document folder and click Index documents to get started.")
    if question:
        with st.spinner("Retrieving passages and writing an answer…"):
            answer = service.ask(question)
        st.session_state.messages.append(asdict(answer))
        st.rerun()
except Exception as exc:
    show_error(exc)
