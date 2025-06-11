from langchain_openai import ChatOpenAI
from core.settings import settings

def chat_model(temperature: float = 0):
    return ChatOpenAI(
        openai_api_key=settings.bati_openai_api_key,
        model="gpt-4o",
        temperature=temperature,
    )
