"""Run with: streamlit run app.py."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import streamlit as st

from rag_langchain.config import Settings
from rag_langchain.popups import confirm_action
from rag_langchain.service import RAGService

st.set_page_config(page_title="RAG Council · Document workspace", page_icon="◈", layout="wide")
st.markdown(
    """<style>
.block-container{max-width:1120px;padding-top:3rem}
[data-testid="stSidebar"]{border-right:1px solid #2b3a50}
h1{letter-spacing:-.045em!important;font-size:3rem!important}
[data-testid="stChatMessage"]{border:1px solid #2b3a50;border-radius:14px;padding:1.3rem}
[data-testid="stMetric"]{border:1px solid #2b3a50;border-radius:10px;padding:12px 18px}
[data-testid="stExpander"]{border-color:#2b3a50}
[data-testid="stBaseButton-primary"]{color:#0c1422}
.eyebrow{color:#74d6bf;font-size:.75rem;font-weight:700;letter-spacing:.18em}
@media(max-width:640px){h1{font-size:2.2rem!important}.block-container{padding-top:1.5rem}}
</style>""",
    unsafe_allow_html=True,
)


def show_error(exc):
    if isinstance(exc, ValueError):
        st.error(str(exc))
    else:
        st.error(
            f"Operation failed ({type(exc).__name__}). Check your API key, model access, "
            "network connection, and database path."
        )


def show_answer(item):
    with st.chat_message("user"):
        st.write(item["question"])
    with st.chat_message("assistant"):
        st.caption("COUNCIL SYNTHESIS" if item.get("mode") == "council" else "SINGLE ANSWER")
        st.markdown(item["answer"])
        for warning in item.get("warnings", []):
            st.warning(warning)
        answer_tab, council_tab = st.tabs([f"Sources · {len(item['sources'])}", "Council reviews"])
        with answer_tab:
            for source in item["sources"]:
                page = f" · page {source['page']}" if source["page"] else ""
                with st.expander(f"[{source['number']}] {source['source']}{page}"):
                    st.caption(f"Cosine distance: {source['distance']:.4f} (lower is closer)")
                    st.text(source["excerpt"])
        with council_tab:
            if not item.get("reviews"):
                st.caption("Choose Council for three independent evidence reviews.")
            for review in item.get("reviews", []):
                with st.expander(f"{review['role']} · {review['status']}"):
                    st.caption(f"{review['elapsed_seconds']:.1f}s · Model-generated review")
                    if review["status"] == "failed":
                        st.warning(
                            "Reviewer unavailable. No findings were used from this reviewer."
                        )
                    else:
                        st.markdown(review["findings"])
        st.download_button(
            "Export answer",
            data=json.dumps(item, indent=2),
            file_name="rag-answer.json",
            mime="application/json",
            key=f"export-{item['id']}",
        )


try:
    defaults = Settings.from_env()
    with st.sidebar:
        st.markdown("### ◈ RAG Council")
        st.caption("YOUR DOCUMENT WORKSPACE")
        st.divider()
        st.subheader("Library")
        data = st.text_input("Document file or folder", str(defaults.data_path))
        db = st.text_input("Index folder", str(defaults.db_path))
        top_k = st.slider("Passages to retrieve", 1, 20, min(defaults.top_k, 20))
        rebuild = st.checkbox("Rebuild all embeddings", value=False)
        index = st.button("Index documents", type="primary", use_container_width=True)
        st.caption("Document text and questions are sent to Google. Set your API key in .env.")
    settings = replace(defaults, data_path=Path(data), db_path=Path(db), top_k=top_k)
    service = RAGService(settings)
    if index and rebuild:
        st.session_state.pending_action = ("rebuild", data, db)
        st.session_state.confirm_sequence = st.session_state.get("confirm_sequence", 0) + 1
    confirmed_rebuild = st.session_state.pop("confirmed_rebuild", None)
    if (index and not rebuild) or confirmed_rebuild == (data, db):
        with st.spinner("Indexing documents…"):
            result = service.ingest(rebuild=confirmed_rebuild == (data, db))
        st.toast(f"Indexed {result['files']} files · {result['chunks']} passages", icon="✅")
        st.sidebar.success(f"{result['embedded_chunks']} passages newly embedded")
    status = service.status()
    with st.sidebar:
        st.metric("Indexed passages", status["chunks"])
        with st.expander(f"Indexed files · {status['files']}"):
            for source in status.get("sources", []):
                st.text(source)
    identity = (
        str(settings.db_path.resolve()),
        service.manifest_path.read_text() if service.manifest_path.exists() else "",
        settings.chat_model,
        top_k,
        settings.max_context_chars,
    )
    if st.session_state.get("corpus") != identity:
        st.session_state.corpus = identity
        st.session_state.messages = []
    with st.sidebar:
        if st.session_state.messages and st.button("Clear conversation…"):
            st.session_state.pending_action = ("clear", data, db)
            st.session_state.confirm_sequence = st.session_state.get("confirm_sequence", 0) + 1
        st.divider()
        st.caption("Evidence first. Every answer starts with your documents.")
    st.markdown('<p class="eyebrow">RETRIEVE · REVIEW · UNDERSTAND</p>', unsafe_allow_html=True)
    st.title("Give your documents a voice.")
    st.write("Ask a question. Follow the evidence. See where reviewers disagree.")
    mode_label = st.radio("Answer approach", ["Single", "Council"], horizontal=True)
    mode = mode_label.lower()
    st.caption(
        "Single: one model call. Council: three parallel reviews + a chair, up to four calls. "
        "Council takes longer and uses more tokens; agreement does not guarantee correctness."
        if mode == "council"
        else "A direct, source-grounded answer with one model call."
    )
    if not st.session_state.messages:
        with st.container(border=True):
            st.subheader("Your next insight starts with a question.")
            st.write(
                "Compare claims across notes, find a definition, "
                "or ask what the evidence leaves open."
            )
            st.caption("Try: What are the main claims, and which passages support them?")
    st.caption("Questions are independent. Include names and context in follow-ups.")
    for item in st.session_state.messages:
        show_answer(item)
    ready = status["indexed"] and status["chunks"] > 0
    pending = st.session_state.get("pending_action")
    if pending:
        # Bind confirmation to the selected paths, rather than a later changed corpus.
        if pending[1:] != (data, db):
            st.session_state.pending_action = None
        else:

            @st.dialog("Review this action", dismissible=False)
            def action_dialog():
                is_rebuild = pending[0] == "rebuild"
                result = confirm_action(
                    "Rebuild all embeddings?" if is_rebuild else "Clear this conversation?",
                    "This re-embeds every document using Google and may incur API costs. "
                    "The previous index stays active if indexing fails."
                    if is_rebuild
                    else "Answers in this session will be removed. Your indexed documents stay.",
                    key=f"confirm-{st.session_state.confirm_sequence}",
                )
                if result is not None:
                    st.session_state.pending_action = None
                    if result.get("confirmed"):
                        if is_rebuild:
                            st.session_state.confirmed_rebuild = (data, db)
                        else:
                            st.session_state.messages = []
                    st.rerun()

            action_dialog()
    question = st.chat_input("Ask your documents a question…", disabled=not ready)
    if not ready:
        st.info("Choose a document folder and click Index documents to get started.")
    if question:
        with st.status("Working with your documents…", expanded=True) as activity:
            answer = service.ask(question, mode=mode, progress=st.write)
            activity.update(label="Answer ready", state="complete", expanded=False)
        item = asdict(answer)
        item["id"] = st.session_state.get("answer_sequence", 0)
        st.session_state.answer_sequence = item["id"] + 1
        st.session_state.messages.append(item)
        st.rerun()
except Exception as exc:
    show_error(exc)
