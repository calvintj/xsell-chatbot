import logging
import time
from typing import Dict, List, Optional
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from core.settings import settings

"""
Vector store utilities for the Jenius FX chatbot.
Handles Pinecone index management, vector store caching, and document retrieval.
"""

# cache one store per namespace
_VECTORSTORES: Dict[str, PineconeVectorStore] = {}

logger = logging.getLogger(__name__)

def _ensure_index():
    """
    Create Pinecone index if it doesn't exist, and return the Index client.
    Waits until the index is ready.
    """
    try:
        pc = Pinecone(api_key=settings.pinecone_api_key)
        name = settings.pinecone_index
        if name not in pc.list_indexes().names():
            logger.info(f"Creating Pinecone index: {name}")
            pc.create_index(
                name=name,
                dimension=settings.embed_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            # wait until ready
            while not pc.describe_index(name).status["ready"]:
                logger.info(f"Waiting for Pinecone index '{name}' to be ready...")
                time.sleep(1)
        return pc.Index(name)
    except Exception as e:
        logger.error(f"Error ensuring Pinecone index: {e}")
        raise


def get_vectorstore(namespace: str = "") -> PineconeVectorStore:
    """
    Return a LangChain PineconeVectorStore scoped to `namespace`.
    Caches a separate store for each namespace.
    """
    if namespace in _VECTORSTORES:
        return _VECTORSTORES[namespace]
    try:
        # ensure the index exists
        index = _ensure_index()
        # create embeddings instance
        embed = OpenAIEmbeddings(model=settings.embed_model)
        store = PineconeVectorStore(
            index=index,
            embedding=embed,
            text_key="text",
            namespace=namespace,
        )
        _VECTORSTORES[namespace] = store
        logger.info(f"Created PineconeVectorStore for namespace '{namespace}'")
        return store
    except Exception as e:
        logger.error(f"Error creating vector store for namespace '{namespace}': {e}")
        raise


def get_raw_pinecone_index():
    """
    Return the underlying Pinecone Index client for direct operations.
    Ensures the index exists before returning.
    """
    try:
        idx = _ensure_index()
        return idx
    except Exception as e:
        logger.error(f"Error getting raw Pinecone index: {e}")
        raise


def retrieve_docs(query: str, lang: Optional[str] = None, k: int = 3) -> List[Document]:
    """
    Do a similarity search in the `lang` namespace (if provided),
    filter by metadata {'lang': lang}, and fallback to unfiltered if no hits.
    Args:
        query: The query string.
        lang: Optional language code for namespace and filtering.
        k: Number of documents to retrieve.
    Returns:
        List of matching Document objects.
    """
    ns = lang or ""
    try:
        vs = get_vectorstore(namespace=ns)
        # primary search: metadata filter
        docs = vs.similarity_search(query, k=k, filter={"lang": lang} if lang else None)
        if not docs:
            # fallback: same namespace, no filter
            docs = vs.similarity_search(query, k=k)
        logger.info(
            f"Retrieved {len(docs)} docs for query '{query}' in namespace '{ns}'"
        )
        return docs
    except Exception as e:
        logger.error(
            f"Error retrieving docs for query '{query}' in namespace '{ns}': {e}"
        )
        return []
