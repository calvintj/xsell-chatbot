from dotenv import load_dotenv
from pydantic_settings import BaseSettings
import os

load_dotenv()                      # .env is parsed exactly once

class Settings(BaseSettings):
    bati_openai_api_key: str
    pinecone_api_key: str

    class Config:                  # allow BATI_OPENAI_API_KEY in .env
        env_prefix = ""
        case_sensitive = False

settings = Settings()

# Bridge → libraries that look for OPENAI_API_KEY
os.environ.setdefault("OPENAI_API_KEY", settings.bati_openai_api_key)
