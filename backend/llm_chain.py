from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.retriever import get_retriever

def build_qa_chain(memory):
    retriever = get_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )
    return chain
