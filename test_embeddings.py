import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
print(f"Loaded key starting with: {api_key[:10]}...")

genai.configure(api_key=api_key)

print("Listing models:")
try:
    for m in genai.list_models():
        print(f"Name: {m.name}, Supported Methods: {m.supported_generation_methods}")
except Exception as e:
    print(f"Error listing models: {e}")

print("\nTrying to embed using langchain_google_genai:")
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    # Try different models
    for model_name in ["models/gemini-embedding-001", "models/gemini-embedding-2", "models/gemini-embedding-2-preview"]:
        try:
            print(f"Trying {model_name}...")
            embeddings = GoogleGenerativeAIEmbeddings(model=model_name)
            res = embeddings.embed_query("Hello world")
            print(f"Success with {model_name}! Embedding length: {len(res)}")
            break
        except Exception as ex:
            print(f"Failed with {model_name}: {ex}")
except Exception as e:
    print(f"Error importing or setting up langchain embeddings: {e}")

print("\nTrying ChatGoogleGenerativeAI:")
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    for model_name in ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-2.5-flash", "gemini-3.5-flash"]:
        try:
            print(f"Trying LLM model: {model_name}...")
            llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
            res = llm.invoke("Hello, say 'Test successful'")
            print(f"Success with LLM {model_name}: {res.content}")
        except Exception as ex:
            print(f"Failed with LLM {model_name}: {ex}")
except Exception as e:
    print(f"Error importing or setting up ChatGoogleGenerativeAI: {e}")

