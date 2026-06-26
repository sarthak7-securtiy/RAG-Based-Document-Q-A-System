import os
import sys
import shutil
import fitz  # PyMuPDF
from dotenv import load_dotenv

# Ensure we can import from backend
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Load env
load_dotenv()

# We will create a test PDF, run ingestion, retrieval, and LLM query.
TEST_PDF_PATH = "test_sample.pdf"
TEST_CHROMA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "chroma_db_test"))

def create_sample_pdf(path):
    print(f"Creating sample PDF at: {path}")
    doc = fitz.open()
    page = doc.new_page()
    text = (
        "RAG Document QA System Test Document\n\n"
        "Antigravity is a powerful agentic AI coding assistant designed by Google DeepMind.\n"
        "The project is built using FastAPI backend and Streamlit frontend.\n"
        "It uses langchain, Chroma DB, and Google Gemini API for embeddings and generation.\n"
        "This is a specific secret code for verification: ANTIGRAVITY-9988-VALID.\n"
    )
    page.insert_text((50, 50), text)
    doc.save(path)
    doc.close()

def run_tests():
    # 1. Check API Key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[-] ERROR: GOOGLE_API_KEY not found in environment or .env file.")
        return False
    print(f"[+] GOOGLE_API_KEY is configured (starts with {api_key[:10]}...)")

    # 2. Test Embeddings
    print("\n--- Testing Embeddings Component ---")
    try:
        from backend.embeddings import get_embeddings
        embeddings = get_embeddings()
        print(f"[+] Successfully loaded embedding class: {embeddings.__class__.__name__}")
        print(f"[+] Model configured: {getattr(embeddings, 'model', 'unknown')}")
        
        # Try embedding a test query
        res = embeddings.embed_query("Test embedding generation")
        print(f"[+] Embeddings generation success. Vector dimension: {len(res)}")
    except Exception as e:
        print(f"[-] Embeddings test failed: {e}")
        # Continue to see other errors

    # 3. Create Sample PDF & Test Ingestion
    print("\n--- Testing Ingestion Component ---")
    create_sample_pdf(TEST_PDF_PATH)
    
    # Temporarily override CHROMA_DB_DIR in backend.vector_store to avoid messing up main database
    import backend.vector_store
    original_chroma_dir = backend.vector_store.CHROMA_DB_DIR
    backend.vector_store.CHROMA_DB_DIR = TEST_CHROMA_DIR
    
    try:
        from backend.ingestion import ingest_document
        print("Ingesting document...")
        success = ingest_document(TEST_PDF_PATH)
        if success:
            print("[+] Document ingestion reported success.")
        else:
            print("[-] Document ingestion returned False.")
    except Exception as e:
        print(f"[-] Ingestion test failed: {e}")

    # 4. Test Retriever
    print("\n--- Testing Retriever Component ---")
    try:
        from backend.retriever import get_retriever
        retriever = get_retriever()
        print("[+] Got retriever from backend.")
        docs = retriever.invoke("What is the secret code?")
        print(f"[+] Retrieved {len(docs)} documents:")
        for idx, doc in enumerate(docs):
            print(f"    Document {idx+1}: {doc.page_content.strip()}")
    except Exception as e:
        print(f"[-] Retriever test failed: {e}")

    # 5. Test LLM and QA Chain
    print("\n--- Testing LLM QA Chain Component ---")
    try:
        from langchain.memory import ConversationBufferMemory
        from backend.llm_chain import build_qa_chain
        
        memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True,
            output_key="answer"
        )
        chain = build_qa_chain(memory)
        print("[+] Successfully built QA chain.")
        
        print("Running query: 'What is the secret code for verification?'")
        result = chain.invoke({"question": "What is the secret code for verification?"})
        print(f"[+] QA Chain Answer: {result['answer']}")
        print("[+] Source documents:")
        for doc in result.get("source_documents", []):
            print(f"    - {doc.metadata.get('source')} (page {doc.metadata.get('page')})")
    except Exception as e:
        print(f"[-] QA Chain test failed: {e}")

    # Clean up
    print("\n--- Cleaning up test artifacts ---")
    if os.path.exists(TEST_PDF_PATH):
        os.remove(TEST_PDF_PATH)
        print(f"Removed {TEST_PDF_PATH}")
    if os.path.exists(TEST_CHROMA_DIR):
        # Retry deletion if locked
        try:
            shutil.rmtree(TEST_CHROMA_DIR)
            print(f"Removed temporary vector store at {TEST_CHROMA_DIR}")
        except Exception as e:
            print(f"Warning: Could not remove temporary vector store: {e}")

    # Restore vector store dir
    backend.vector_store.CHROMA_DB_DIR = original_chroma_dir

if __name__ == "__main__":
    run_tests()
