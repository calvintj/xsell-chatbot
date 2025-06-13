"""
Terminal runner for the language-aware Jenius FX chatbot.

Run:
    PYTHONPATH=src python3 -m app.main
"""

import logging
import sys
from langchain.schema import AIMessage, HumanMessage

from services.rag_services import stream_chat_with_memory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run_cli() -> None:
    """
    Run the chatbot in a terminal-based CLI loop.
    Prompts the user for language selection and user input, maintains chat history,
    and streams responses from the RAG pipeline until the user exits.
    """
    try:
        lang = input("Choose language [en/id] › ").strip().lower() or "en"
        if lang not in ("en", "id"):
            lang = "en"

        history = []  # ← start EMPTY; chat_with_memory adds SystemMessage itself
        logger.info(f"Starting CLI chatbot in language: {lang}")
        print("\nJenius FCY Chatbot - type 'exit' to quit.\n")

        while True:
            user = input("You: ").strip()
            if user.lower() == "exit":
                logger.info("User exited the CLI chatbot.")
                break
            logger.info(f"User input: {user}")
            try:
                reply_chunks = stream_chat_with_memory(history, user, lang=lang)
                reply_text = "".join(reply_chunks)
                reply_msg = AIMessage(content=reply_text)
                history.extend([HumanMessage(content=user), reply_msg])
                logger.info(f"AI reply: {reply_text}")
                print("AI:", reply_text, "\n")
            except Exception as e:
                logger.error(f"Error during chat: {e}")
                print("AI: [Sorry, there was an error generating a response.]\n")
    except Exception as e:
        logger.error(f"Fatal error in CLI: {e}")
        print("[Fatal error: unable to start chatbot]")


if __name__ == "__main__":
    """
    Entry point for running the CLI chatbot.
    """
    run_cli()
    # uvicorn.run(app, host="0.0.0.0", port=8000)
