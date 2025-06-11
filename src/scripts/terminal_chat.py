from langchain.schema import HumanMessage
from services.rag_services import SYS_MSG, chat_with_memory

history = [SYS_MSG]
print("Type 'exit' to quit.")
while True:
    q = input("You: ")
    if q.strip().lower() == "exit":
        break
    reply = chat_with_memory(history, q)
    history.extend([HumanMessage(content=q), reply])
    print("AI:", reply.content[:500], "\n")
