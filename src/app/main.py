# src/main.py
"""
Entry-point for a simple terminal RAG chat.
Run with:  PYTHONPATH=src python3 -m app.main
"""

from langchain.schema import SystemMessage, HumanMessage
from services.rag_services import chat_with_memory, SYS_MSG

def run_cli() -> None:
    history = [SYS_MSG]
    print("🔮  Jenius FCY Chatbot – type 'exit' to quit.\n")
    while True:
        user = input("You: ")
        if user.lower().strip() == "exit":
            break
        reply = chat_with_memory(history, user)
        history.extend([HumanMessage(content=user), reply])
        print(f"AI: {reply.content}\n")

if __name__ == "__main__":
    run_cli()
