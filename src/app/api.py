"""
FastAPI app for the Jenius FX chatbot API.
Provides a /chat-stream endpoint for streaming chat responses.
"""

# PYTHONPATH=src uvicorn app.api:app

import logging
from typing import List, Literal, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain.schema import HumanMessage
from pydantic import BaseModel

from services.rag_services import stream_chat_with_memory

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)


class ChatStreamRequest(BaseModel):
    """
    Request model for the /chat-stream endpoint.
    history: List of message dicts (role/content)
    user_input: The user's input string
    lang: Optional language code ("en" or "id")
    """

    history: List[dict]  # role: "user"/"assistant", content: str
    user_input: str
    lang: Optional[Literal["en", "id"]] = None


@app.post("/chat-stream")
async def chat_stream(req: ChatStreamRequest):
    """
    Stream chat responses for a given user input and chat history.
    Converts incoming history to HumanMessage objects and streams the response chunks.
    """
    try:
        logger.info(
            f"Received /chat-stream request: user_input='{req.user_input}', lang='{req.lang}'"
        )
        history = [HumanMessage(**m) for m in req.history]

        def event_gen():
            for chunk in stream_chat_with_memory(history, req.user_input, req.lang):
                yield f"data:{chunk}\n\n"

        return StreamingResponse(event_gen(), media_type="text/event-stream")
    except Exception as e:
        logger.error(f"Error in /chat-stream: {e}")
        return {"error": "An error occurred while processing your request."}
