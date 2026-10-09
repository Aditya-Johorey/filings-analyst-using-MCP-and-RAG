import os
from dotenv import load_dotenv
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

load_dotenv()
COLLECTION = "filings"

def get_embeddings():
    # Runs on CPU inside the app, 384 dimensions.
    # Ingest and query MUST use the same model.
    return FastEmbedEmbeddings(model_name = "BAAI/bge-small-en-v1.5")

def get_llm():
    if os.getenv("LLM_PROVIDER", "groq") == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    from langchain_ollama import ChatOllama
    return ChatOllama(model = "llama3.1:8b", temperature=0)

def get_client():
    return QdrantClient(url = os.environ["QDRANT_URL"],
                        api_key = os.getenv("QDRANT_API_KEY"))

def get_store():
    return QdrantVectorStore(client = get_client(),
                             collection_name = COLLECTION,
                             embedding=get_embeddings())