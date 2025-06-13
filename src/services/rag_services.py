import logging
from typing import List, Literal, Optional

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langdetect import detect  # pip install langdetect

from retrieval.vector_store import retrieve_docs
from services.llm import chat_model

"""
RAG (Retrieval-Augmented Generation) services for the Jenius FX chatbot.
Handles system prompts, prompt augmentation, and chat with memory.
"""

logger = logging.getLogger(__name__)

# ───────────────────────── 1 ─ SYSTEM PROMPTS ──────────────────────────
SYS_PROMPT = {
    "en": SystemMessage(
        content="""
                    You are NIX, a helpful and knowledgeable assistant for Jenius by Bank BTPN (a member of the SMBC Group).
                    You are strictly limited to answering questions related to Foreign Currency (FCY) services offered by Jenius.
                    Your scope includes:
                    - FCY account features in the Jenius app
                    - Supported foreign currencies (e.g., USD, SGD, JPY)
                    - How to open, top-up, convert, or withdraw from an FCY balance
                    - Exchange rates and conversion processes
                    - Promotions or offers related to FCY
                    - Events or educational content specifically about FCY at Jenius
                    - Transaction fees, limits, and regulatory information (Bank Indonesia/FSA)

                    You must not respond to any queries outside of the FCY topic. This includes:
                    - Rupiah (IDR) services or accounts
                    - Jenius credit cards, loans, or investment products
                    - Non-FCY-related offers or events
                    - General banking questions not involving FCY

                    If a user asks about something unrelated to FCY, respond with:
                    "I'm here to help with foreign currency (FCY) at Jenius. What FCY question can I answer for you today?"

                    If a user greets you, respond with:
                    "Hello! I'm here to help with foreign currency (FCY) at Jenius. What FCY question can I answer for you today?"

                    Be concise, friendly, and clear in English, and always use English if the user's language is not English.
        """
    ),
    "id": SystemMessage(
        content="""
                    Anda adalah NIX, asisten yang membantu dan berpengetahuan luas untuk Jenius by Bank BTPN (anggota Grup SMBC).
                    Anda hanya terbatas untuk menjawab pertanyaan terkait layanan Mata Uang Asing (FCY) yang ditawarkan oleh Jenius.
                    Cakupan Anda meliputi:
                    - Fitur akun FCY di aplikasi Jenius
                    - Mata uang asing yang didukung (misalnya, USD, SGD, JPY)
                    - Cara membuka, menambah, mengonversi, atau menarik saldo FCY
                    - Nilai tukar dan proses konversi
                    - Promosi atau penawaran terkait FCY
                    - Acara atau konten edukasi khusus tentang FCY di Jenius
                    - Biaya transaksi, batasan, dan informasi regulasi (Bank Indonesia/OJK)

                    Anda tidak boleh menanggapi pertanyaan apa pun di luar topik FCY. Ini termasuk:
                    - Layanan atau akun Rupiah (IDR)
                    - Kartu kredit, pinjaman, atau produk investasi Jenius
                    - Penawaran atau acara yang tidak terkait dengan FCY
                    - Pertanyaan perbankan umum yang tidak melibatkan FCY

                    Jika pengguna bertanya tentang sesuatu yang tidak terkait dengan FCY, jawab dengan:
                    "Saya di sini untuk membantu layanan Mata Uang Asing (FCY) di Jenius. Ada pertanyaan FCY yang bisa saya bantu?"
                    
                    Jika pengguna mengucapkan salam, jawab dengan:
                    "Halo! Saya di sini untuk membantu layanan Mata Uang Asing (FCY) di Jenius. Ada pertanyaan FCY yang bisa saya bantu?"

                    Selalu singkat, ramah, dan jelas dalam bahasa Indonesia, dan tetap menggunakan bahasa Indonesia walaupun user menggunakan bahasa lain.
       """
    ),
}


# ───────────────────────── 2 ─ AUGMENT PROMPT ──────────────────────────
def augment_prompt(query: str, lang: str) -> str:
    """
    Retrieve relevant documents and build an augmented prompt for the LLM.
    Args:
        query: The user's query string.
        lang: The language code ("en" or "id").
    Returns:
        A string containing context and the user query.
    """
    try:
        docs = retrieve_docs(query, lang=lang, k=3)
        context = "\n".join(d.page_content for d in docs) or "No relevant context."
        logger.info(f"Retrieved {len(docs)} docs for query '{query}' in lang '{lang}'")
    except Exception as e:
        logger.error(f"Error retrieving docs for query '{query}': {e}")
        context = "No relevant context."
    return f"Context:\n{context}\n\nQuery:\n{query}"


# ───────────────────────── 3 ─ CHAT WITH MEMORY ────────────────────────
def stream_chat_with_memory(
    history: List, user_input: str, lang: Optional[Literal["en", "id"]] = None
):
    """
    Stream chat responses from the LLM, using RAG and chat history.
    Args:
        history: List of previous message objects.
        user_input: The user's input string.
        lang: Optional language code ("en" or "id").
    Yields:
        Chunks of the LLM's response as they are generated.
    """
    try:
        lang = lang or detect(user_input)[:2]
        lang = "id" if lang == "id" else "en"
    except Exception as e:
        logger.error(f"Language detection failed for input '{user_input}': {e}")
        lang = "en"

    rag_prompt = HumanMessage(content=augment_prompt(user_input, lang))
    messages = history + [SYS_PROMPT[lang], rag_prompt]

    try:
        llm = chat_model()
        for chunk in llm.stream(messages):  # uses `ChatOpenAI(streaming=True)`
            if chunk.content:
                yield chunk.content
    except Exception as e:
        logger.error(f"LLM streaming failed: {e}")
        yield "[Sorry, there was an error generating a response.]"
