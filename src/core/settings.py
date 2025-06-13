import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()  # .env is parsed exactly once

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or .env file.
    - bati_openai_api_key: API key for OpenAI
    - pinecone_api_key: API key for Pinecone
    - pinecone_index: Pinecone index name
    - embed_model: Embedding model name
    - embed_dim: Embedding dimension
    """

    bati_openai_api_key: str
    pinecone_api_key: str
    pinecone_index: str = "xsell-chatbot"
    embed_model: str = "text-embedding-3-large"
    embed_dim: int = 3072

    class Config:  # allow BATI_OPENAI_API_KEY in .env
        env_prefix = ""
        case_sensitive = False

settings = Settings()

# Bridge → libraries that look for OPENAI_API_KEY
os.environ.setdefault("OPENAI_API_KEY", settings.bati_openai_api_key)
