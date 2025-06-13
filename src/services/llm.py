import logging
from langchain_openai import ChatOpenAI
from core.settings import settings

"""
LLM (Large Language Model) service for the Jenius FX chatbot.
Provides a factory for creating a streaming ChatOpenAI instance.
"""

logger = logging.getLogger(__name__)

def chat_model(temperature: float = 0):
    """
    Create a streaming ChatOpenAI model instance for chat completion.
    Args:
        temperature: Sampling temperature for the model (default 0).
    Returns:
        A ChatOpenAI instance configured for streaming.
    """
    try:
        return ChatOpenAI(
            openai_api_key=settings.bati_openai_api_key,
            model="gpt-4o",
            temperature=temperature,
            streaming=True,
        )
    except Exception as e:
        logger.error(f"Failed to instantiate ChatOpenAI: {e}")
        raise
