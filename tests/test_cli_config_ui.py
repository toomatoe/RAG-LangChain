import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from rag_langchain.cli import main
from rag_langchain.config import Settings, api_key, make_chat, make_embeddings


def test_config_validation(tmp_path):
    with pytest.raises(ValueError, match="overlap"):
        Settings(chunk_size=10, chunk_overlap=10)
    with pytest.raises(ValueError, match="top_k"):
        Settings(top_k=0)
    with pytest.raises(ValueError, match="contain"):
        Settings(data_path=tmp_path / "books", db_path=tmp_path)


def test_key_alias_and_provider_construction(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        api_key()
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    assert api_key() == "test-key-not-real"
    # These constructors make no requests; isolate them from runner proxy settings.
    for name in list(os.environ):
        if name.lower().endswith("_proxy"):
            monkeypatch.delenv(name, raising=False)
    assert make_embeddings(Settings()).model == "gemini-embedding-001"
    assert make_chat(Settings()).model == "gemini-2.5-flash"


def test_cli_status_and_missing_key(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "notes.txt"
    source.write_text("Some notes.")
    args = ["--db", str(tmp_path / "index"), "--data", str(source)]
    assert main([*args, "status"]) == 0
    assert '"indexed": false' in capsys.readouterr().out
    assert main([*args, "ingest"]) == 1
    assert "GOOGLE_API_KEY" in capsys.readouterr().err
    assert main([*args, "ask", "hello"]) == 1
    assert "No index" in capsys.readouterr().err


def test_web_app_empty_index(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_DB_PATH", str(tmp_path / "index"))
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=30)
    assert not app.exception
    assert app.chat_input[0].disabled
    assert app.info


def test_web_app_index_and_answer(setup, monkeypatch):
    service, _, _, _ = setup
    monkeypatch.setattr("rag_langchain.service.RAGService", lambda settings: service)
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=30)
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert not app.chat_input[0].disabled
    app.chat_input[0].set_value("What does Alice follow?").run(timeout=30)
    assert not app.exception
    assert len(app.chat_message) == 2
    assert "rabbit" in app.chat_message[1].markdown[0].value
