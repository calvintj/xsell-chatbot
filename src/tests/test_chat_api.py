from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


@patch("services.rag_services.stream_chat_with_memory")
def test_chat_stream_endpoint(mock_stream):
    """
    Test the /chat-stream endpoint returns a streaming response.
    Mocks the RAG service to avoid external dependencies.
    """
    # Simulate streaming chunks
    mock_stream.return_value = iter(
        [MagicMock(content="Hello!"), MagicMock(content="How can I help?")]
    )
    payload = {
        "history": [{"role": "user", "content": "Hi"}],
        "user_input": "What is FX?",
        "lang": "en",
    }
    response = client.post("/chat-stream", json=payload)
    assert response.status_code == 200
    # The response is a streaming event, so we check the content
    body = b"".join(response.iter_bytes())
    assert b"Hello!" in body
    assert b"How can I help?" in body
