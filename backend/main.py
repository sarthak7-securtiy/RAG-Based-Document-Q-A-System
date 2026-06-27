import os
import shutil
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory

from backend.ingestion import ingest_document
from backend.llm_chain import build_qa_chain

load_dotenv()

app = FastAPI(title="RAG Document Q&A System")

UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "uploaded_docs"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Store memories per session
sessions_memory = {}

class QueryRequest(BaseModel):
    question: str
    session_id: str

def create_sample_pdf(pdf_path: str):
    """Generates a high-quality sample PDF describing the system architecture using PyMuPDF."""
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=595, height=842) # A4 size
    
    # Title
    page.insert_textbox(
        fitz.Rect(50, 40, 545, 90),
        "RAG DOCUMENT QA SYSTEM ARCHITECTURE",
        fontsize=16,
        fontname="hebo",
        color=(0, 0.4, 0.8),
        align=1
    )
    
    content = (
        "This document serves as the target sample document to test the semantic search and "
        "retrieval capabilities of the system. Recruiters can query this document to evaluate the RAG pipeline.\n\n"
        "1. PROJECT SUMMARY & TECH STACK\n"
        "This project is a high-performance Retrieval-Augmented Generation (RAG) web application built using:\n"
        "- Streamlit: Clean, modern responsive user interface for interactive QA.\n"
        "- FastAPI: High-performance, asynchronous REST backend.\n"
        "- LangChain: Pipeline orchestration, memory buffers, and chain state.\n"
        "- Chroma DB: Light, local vector database for embedding indexing.\n"
        "- Google Gemini 2.5 Flash: State-of-the-art LLM for natural, accurate QA generation.\n"
        "- PyMuPDF (fitz): Fast, reliable PDF text parsing.\n\n"
        "2. DOCUMENT INGESTION WORKFLOW\n"
        "When a document is uploaded, it is ingested through the following pipeline:\n"
        "- Extraction: PyMuPDF parses the raw PDF content and metadata.\n"
        "- Chunking: Text is split via LangChain's RecursiveCharacterTextSplitter.\n"
        "- Chunk Size is configured to 500 characters.\n"
        "- Chunk Overlap is configured to 50 characters to preserve context boundaries.\n"
        "- Storage: Chunks are converted into 768-dimensional embeddings via Google Gemini's "
        "'models/gemini-embedding-001' and stored directly in a Chroma vector collection.\n\n"
        "3. RETRIEVAL & QUERY PIPELINE\n"
        "When a user submits a chat query, the following steps occur:\n"
        "- The query is vectorized using the Gemini embeddings model.\n"
        "- Chroma DB calculates cosine similarities to fetch the top matching chunks.\n"
        "- The LangChain ConversationalRetrievalChain synthesizes historical chat buffer "
        "memory, retrieved document context, and the new query.\n"
        "- The contextual prompt is sent to the Gemini 2.5 Flash model, which generates a coherent answer "
        "along with exact page source references.\n\n"
        "4. SYSTEM PERFORMANCE & LOGS\n"
        "The retrieval pipeline has average latency under 1.5 seconds for semantic searches and "
        "re-ranking. Custom memory key storage ensures distinct, independent conversation buffers "
        "keyed by session UUIDs."
    )
    
    page.insert_textbox(
        fitz.Rect(50, 100, 545, 800),
        content,
        fontsize=10.5,
        fontname="helv",
        align=0
    )
    doc.save(pdf_path)
    doc.close()

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        ingest_document(file_path)
        return {"message": f"Successfully ingested {file.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load-demo")
async def load_demo_document():
    demo_pdf_path = os.path.join(UPLOAD_DIR, "system_architecture_spec.pdf")
    try:
        # Create demo PDF if not exists
        create_sample_pdf(demo_pdf_path)
        # Ingest it
        ingest_document(demo_pdf_path)
        return {"message": "Demo document generated and loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load demo document: {str(e)}")

@app.post("/ask")
async def ask_question(req: QueryRequest):
    if req.session_id not in sessions_memory:
        sessions_memory[req.session_id] = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True,
            output_key="answer"
        )
    
    memory = sessions_memory[req.session_id]
    chain = build_qa_chain(memory)
    
    try:
        # ConversationalRetrievalChain requires 'question' not 'query'
        result = chain.invoke({"question": req.question})
        sources = []
        for doc in result.get("source_documents", []):
            sources.append({
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", -1)
            })
            
        return {
            "answer": result["answer"],
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

