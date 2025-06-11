import time

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.core.settings import settings

_INDEX = None  # cache


def get_vectorstore() -> PineconeVectorStore:
    global _INDEX
    if _INDEX:
        return _INDEX

    pc = Pinecone(api_key=settings.pinecone_api_key)
    name = "llama-2-rag"
    if name not in pc.list_indexes().names():
        pc.create_index(
            name=name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(name).status["ready"]:
            time.sleep(1)

    index = pc.Index(name)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    _INDEX = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")
    return _INDEX


def get_raw_pinecone_index():
    """Return the *underlying* Pinecone Index object (no LangChain wrapper)."""
    _ = get_vectorstore()  # ensures the singleton & index exist
    return _INDEX.index  # type: ignore[attr-defined]
