"""
Ingest Jenius FX FAQ (English) with Playwright – optimized:
 • single Chromium instance for all pages
 • batch embeddings (100 chunks per call)
"""

import asyncio, hashlib, re, math
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

from langchain_openai import OpenAIEmbeddings
from src.retrieval.vector_store import get_raw_pinecone_index

BASE   = "https://www.jenius.com/faq/mata-uang-asing"
PAGES  = [
    "tentang-mata-uang-asing",
    "aktivasi-menabung",
    "transaksi-dengan-m-card",
    "kirim-mata-uang-asing",
]
LOCALE = "en"
BATCH  = 100                         # embedding batch size

embedder = OpenAIEmbeddings(model="text-embedding-3-small")
index    = get_raw_pinecone_index()

# …imports and constants stay the same …

def extract_qna(html: str, page_url: str):
    """
    Yield (chunk_text, metadata) tuples for every accordion item.
    Works even if Jenius changes <button> → <div>.
    """
    soup = BeautifulSoup(html, "lxml")

    for acc in soup.select(".accordion-item"):
        # 1 · question ─────────────────────────────
        q_el = acc.select_one(
            "button, .accordion__title, .faq__question, h3, h4"
        )
        if not q_el:
            # skip if we can’t find a question node
            continue
        question = q_el.get_text(" ", strip=True)

        # 2 · answer ───────────────────────────────
        a_el = acc.select_one(
            ".accordion-body, .accordion__body, .faq__answer, .accordion-content"
        )
        if not a_el:
            continue
        answer = a_el.get_text(" ", strip=True)

        chunk = f"{question}\n{answer}"
        meta  = {"source": page_url}
        yield chunk, meta


async def render_all(playwright):
    browser = await playwright.chromium.launch(headless=True)
    page    = await browser.new_page()
    html_pages = {}
    for path in PAGES:
        url = f"{BASE}/{path}?locale={LOCALE}"
        print(f"⏳  Fetching {url}")
        await page.goto(url, timeout=45_000)
        await page.wait_for_selector(".accordion-item")
        html_pages[path] = await page.content()
    await browser.close()
    return html_pages

async def main():
    async with async_playwright() as pw:
        html_map = await render_all(pw)

    all_texts, all_ids, all_meta = [], [], []
    for path, html in html_map.items():
        page_url = f"{BASE}/{path}?locale={LOCALE}"
        for chunk, meta in extract_qna(html, page_url):
            all_texts.append(chunk)
            all_ids.append(hashlib.sha1(chunk.encode()).hexdigest())
            all_meta.append({**meta, "text": chunk})

    print(f"➜ Total chunks to embed: {len(all_texts)}")

    # ── batch embed ───────────────────────────────────────────────
    for i in range(0, len(all_texts), BATCH):
        part_texts = all_texts[i : i + BATCH]
        part_emb   = embedder.embed_documents(part_texts)
        part_ids   = all_ids[i : i + BATCH]
        part_meta  = all_meta[i : i + BATCH]
        index.upsert(zip(part_ids, part_emb, part_meta))
        print(f"   • Upserted batch {i//BATCH + 1}/{math.ceil(len(all_texts)/BATCH)}")

    print("✅  Finished ingesting English FAQ.")

if __name__ == "__main__":
    asyncio.run(main())
