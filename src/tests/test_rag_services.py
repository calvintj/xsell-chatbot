from unittest.mock import MagicMock, patch

import pytest

from services import rag_services


def test_augment_prompt_returns_context_and_query():
    """
    Test that augment_prompt returns a string containing context and the user query.
    Mocks retrieve_docs to avoid external dependencies.
    """
    fake_docs = [MagicMock(page_content="This is a relevant context.")]
    with patch("retrieval.vector_store.retrieve_docs", return_value=fake_docs):
        result = rag_services.augment_prompt("What is FX?", "en")
        assert "Context:" in result
        assert "This is a relevant context." in result
        assert "Query:" in result
        assert "What is FX?" in result
