# RAG-LangChain

Ask questions about local PDFs, Markdown files, and text documents with **Gemini + LangChain + Chroma**. Answers include numbered citations, and the web interface shows the exact passages sent to the model.

The original document-loading experiment is now an installable application with a CLI, Streamlit interface, incremental indexing, automated tests, and Docker support. The included *Alice in Wonderland* text is a ready-to-index example.

## Quick start

Requires Python 3.11 or newer and a Gemini Developer API key with access to your selected chat and embedding models.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install and configure:

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` (`cp .env.example .env` on macOS/Linux, `Copy-Item .env.example .env` in PowerShell), then set:

```dotenv
GOOGLE_API_KEY=your-key
```

Get your key through [Google AI Studio](https://aistudio.google.com/apikey). This project uses the Gemini Developer API directly; you do not need to configure Vertex AI application credentials. Model availability and API quotas depend on your account. Change the model names in `.env` if needed.

```bash
rag ingest
rag ask "Who does Alice follow at the beginning of the book?"
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`. Choose your document folder, index it, and ask a question. Each question is independent; the displayed conversation is not sent as model memory.

## Commands

| Command | Purpose |
| --- | --- |
| `rag ingest` | Index new/changed files and remove deleted files from the corpus |
| `rag ingest --rebuild` | Re-embed every document, replacing the current corpus after success |
| `rag status` | Show indexed files, chunk count, source root, and embedding settings |
| `rag search "question" --top-k 3` | Inspect retrieved excerpts without calling the chat model |
| `rag ask "question" --top-k 5` | Generate an answer with numbered source citations |
| `rag ask "question" --json` | Return the question, answer, and source records as JSON |
| `rag search "question" --json` | Return source excerpts, page numbers, IDs, and distances as JSON |

Global path options go **before** the subcommand:

```bash
rag --data "C:/Users/you/Documents/notes" --db data/notes-index ingest
rag --db data/notes-index ask "What are the main findings?"
python -m rag_langchain status
```

The original command still works: `python loadData.py`. Use `python queryData.py "your question"` for the companion query entry point.

PDF pages use **one-based** page numbers. Markdown and text citations use file paths relative to the selected folder. Retrieval returns cosine distance, where lower values indicate closer vectors; it is not a calibrated confidence score.

## Configuration

Environment variables take precedence over `.env`. Run commands from the repository root for the default relative paths.

| Variable | Default | Meaning |
| --- | --- | --- |
| `GOOGLE_API_KEY` | Required for indexing/search/answers | Gemini API key; `GEMINI_API_KEY` is an accepted fallback |
| `GEMINI_CHAT_MODEL` | `gemini-2.5-flash` | Chat generation model |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` | Shared document and query embedding model |
| `RAG_DATA_PATH` | `data/books` | Single document or recursively scanned folder |
| `RAG_DB_PATH` | `data/chroma_db` | Persistent Chroma database and index manifest |
| `RAG_CHUNK_SIZE` | `1000` | Maximum chunk length in characters |
| `RAG_CHUNK_OVERLAP` | `200` | Overlap in characters; must be smaller than chunk size |
| `RAG_TOP_K` | `5` | Retrieved passage count, from 1 to 50 |
| `RAG_MAX_CONTEXT_CHARS` | `16000` | Character budget for excerpts and source labels |

Changing the embedding model, chunk size, overlap, or source root requires `rag ingest --rebuild`. Changing the chat model does not require re-indexing. Use separate DB paths to keep separate document libraries.

Supported inputs are UTF-8 `.txt` and `.md` files and PDFs with extractable text. Hidden paths, unsupported files, and symlinked files in folders are skipped. Scanned PDFs need OCR first. A malformed, empty, or unreadable document fails indexing with its filename, preserving the previously completed index.

## How it works

1. Discover supported documents and compare file hashes with the index manifest.
2. Extract text and split changed files into overlapping passages with stable chunk IDs.
3. Embed new passages using Google; copy stored vectors for unchanged documents.
4. Build a new Chroma collection and atomically switch the manifest after it is complete.
5. Embed the question, retrieve the nearest passages, and send bounded excerpts to Gemini.
6. Ask the model to answer from those excerpts and cite them; return the excerpt records for inspection.

An interprocess file lock serializes local indexing and retrieval. If embedding fails, the last completed collection remains available. Deleting every source file is treated as an empty-source error and preserves the old index; use a fresh DB directory when intentionally starting an empty library. A process killed mid-update can leave an unused collection, but the manifest still points to the previous completed version.

The source list records passages supplied to the model. It does **not** independently verify every generated claim or citation. Retrieval always returns nearest passages, even for unrelated questions; the prompt requests an explicit abstention when the excerpts do not answer the question. Inspect the passages before relying on the answer. Document content is treated as data in the system prompt, though this is not a complete defense against prompt injection.

## Web interface

The sidebar lets you select paths, adjust passage count, rebuild the index, and inspect indexed filenames. The conversation shows expandable passages, PDF pages, and retrieval distances. Changing the index resets display history to avoid mixing corpora.

The app works with local files; add documents to your folder before indexing. It does not provide browser uploads or user authentication. Docker binds the interface to localhost by default.

## Docker

Create `.env` and set your API key first:

```bash
docker compose up --build
```

Open `http://localhost:8501`, then click **Index documents**. The source folder is mounted read-only, and the index persists in the `rag-index` volume. The container runs as an unprivileged user. For Docker, retain the default `RAG_DATA_PATH` and `RAG_DB_PATH` or use paths inside `/app` that match your mounts.

```bash
docker compose down
```

Stopping the container preserves the index. Removing the named volume removes stored embeddings.

## Development and tests

```bash
pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest -q
```

Tests use deterministic local embeddings and a fake chat model; no key, API calls, or model downloads are needed. They exercise real Chroma persistence and retrieval, unchanged-file reuse, edits and deletions, failed-update rollback, model configuration changes, PDF metadata, context limits, CLI errors, and Streamlit indexing and answers. CI runs these checks on Python 3.11 and 3.12.

Tests do not establish real Gemini answer quality or account/model access. Validate those with the sample book after configuring your key. Keep credentials out of source control. The local index stores document text and vectors; indexing and query embedding send text to Google, and answering sends the question plus retrieved passages to Google. Review the provider's policies before using private documents. Chroma telemetry is disabled by this application's client configuration.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Missing API key | Set `GOOGLE_API_KEY` in `.env` in your working directory |
| Invalid key, permission, model, or quota error | Check your Gemini account, enabled models, API limits, and configured model names |
| No index found | Run `rag ingest` against the same `--db` path |
| Index settings/source path changed | Run `rag ingest --rebuild` |
| No extractable text | OCR the PDF or replace the empty file |
| Text decoding fails | Save the text file as UTF-8 |
| Poor retrieval | Inspect `rag search`, improve source text, or adjust chunks and rebuild |
| Concurrent operation times out | Wait for the other index/search process to finish, then retry |

## Project layout

```text
src/rag_langchain/
  config.py       Environment settings and Gemini providers
  documents.py    Discovery, PDF/text extraction, and chunk IDs
  service.py      Versioned Chroma indexing, retrieval, and answers
  cli.py          ingest / status / search / ask commands
app.py            Streamlit document Q&A
loadData.py       Original ingestion command compatibility
queryData.py      Query command compatibility
tests/            Offline integration and interface tests
```

References: [LangChain Gemini chat](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai), [Gemini embeddings](https://docs.langchain.com/oss/python/integrations/embeddings/google_generative_ai), and [Chroma integration](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma).

GPL-3.0 license. See [LICENSE](LICENSE).
