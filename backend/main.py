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
