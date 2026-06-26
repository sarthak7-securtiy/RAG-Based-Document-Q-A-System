import streamlit as st
import requests
import uuid
import os

st.set_page_config(page_title="RAG Document Q&A", layout="wide")

# Backend API URL
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Initialize session state for chat history and session ID
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("📚 RAG-Based Document Q&A System")

# Sidebar for PDF Upload
with st.sidebar:
    st.header("Document Upload")
    uploaded_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)
    
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing..."):
                for uploaded_file in uploaded_files:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    response = requests.post(f"{API_URL}/upload", files=files)
                    
                    if response.status_code == 200:
                        st.success(f"Successfully processed {uploaded_file.name}")
                    else:
                        st.error(f"Error processing {uploaded_file.name}: {response.text}")
        else:
            st.warning("Please upload at least one PDF.")

# Chat Interface
st.header("Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for source in msg["sources"]:
                    st.write(f"- {source['source']} (Page {source['page']})")

if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Get response from backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"question": prompt, "session_id": st.session_state.session_id}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])
                    
                    st.markdown(answer)
                    if sources:
                        with st.expander("Sources"):
                            for source in sources:
                                st.write(f"- {source['source']} (Page {source['page']})")
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")
