from xml.dom.minidom import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_PATH = "data\\books\\alice_in_the_wonderland.md"

def load_documents():
    loader = PyPDFLoader(DATA_PATH)
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f'Split {len(documents)} documents into {len(chunks)} chunks.')
   
