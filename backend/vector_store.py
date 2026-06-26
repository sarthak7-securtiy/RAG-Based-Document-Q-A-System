import os
from langchain_community.vectorstores import Chroma
from backend.embeddings import get_embeddings

CHROMA_DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chroma_db"))

def get_vectorstore():
    return Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=get_embeddings()
    )
