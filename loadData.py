from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import VertexAIEmbeddings
import shutil
import os

DATA_PATH = "data\\books\\alice_in_the_wonderland.md"
CHROMA_PATH = "data\\chroma_db"


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    # choose loader based on file extension
    ext = os.path.splitext(DATA_PATH)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(DATA_PATH)
    else:
        loader = TextLoader(DATA_PATH, encoding="utf-8")
    documents = loader.load()
    return documents


def split_text(documents: list):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f'Split {len(documents)} documents into {len(chunks)} chunks.')

    # safe sample print (avoid index error)
    if len(chunks) > 10:
        document = chunks[10]
        print(document.page_content)
        print(document.metadata)
    else:
        print("Less than 11 chunks; skipping sample print.")

    return chunks


def save_to_chroma(chunks: list):
    #clear out database
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH) 
def get_vertexai_embeddings():
    
    model_name = os.getenv("GEMINI_EMBEDDING_MODEL")
    try:
        if model_name:
            return VertexAIEmbeddings(model_name=model_name)
        return VertexAIEmbeddings()
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize VertexAIEmbeddings. Ensure Google Vertex AI SDK is installed and configured. "
            + str(e)
        )


def save_to_chroma(chunks: list):
    # clear out database directory then create anew
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    os.makedirs(CHROMA_PATH, exist_ok=True)
        embeddings = get_vertexai_embeddings()
    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
    try:
        db.persist()
    except Exception:
        # some Chroma versions call .persist() automatically or don't support it
        pass
    print(f"Saved {len(chunks)} chunks to Chroma at {CHROMA_PATH}")


if __name__ == "__main__":
    main()

