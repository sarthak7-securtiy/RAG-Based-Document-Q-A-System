import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.vector_store import get_vectorstore

def ingest_document(pdf_path: str):
    # Extract text
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    
    # Chunk it
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    
    # Store
    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    
    return True
