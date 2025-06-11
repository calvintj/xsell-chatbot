import ssl

import nltk

try:
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except AttributeError:
    pass
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

import asyncio
import re
import time
from urllib.parse import urljoin

import aiohttp
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from unstructured.partition.html import partition_html

from src.retrieval.vector_store import get_raw_pinecone_index

BASE = "https://www.jenius.com/faq/mata-uang-asing"
URL_LIST = [
    f"{BASE}/tentang-mata-uang-asing",
    f"{BASE}/aktivasi-menabung",
    f"{BASE}/transaksi-dengan-m-card",
    f"{BASE}/kirim-mata-uang-asing",
]


async def fetch(session, url):
    async with session.get(url, timeout=30) as r:
        return url, await r.text()


async def gather_html():
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(ssl=False)
    ) as session:
        tasks = [fetch(session, u) for u in URL_LIST]
        return await asyncio.gather(*tasks)


def html_to_chunks(html, url):
    # 1) strip boilerplate
    elements = partition_html(text=html)
    body = "\n".join(e.text for e in elements)
    body = re.sub(r"\s+", " ", body).strip()

    # 2) split
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    for i, chunk in enumerate(splitter.split_text(body)):
        yield {
            "id": f"{url}#p{i}",
            "text": chunk,
            "metadata": {"source": url},
        }


async def main():
    raw_index = get_raw_pinecone_index()
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    html_pages = await gather_html()

    for url, html in html_pages:
        vectors, ids, meta = [], [], []
        for piece in html_to_chunks(html, url):
            ids.append(piece["id"])
            vectors.append(embedder.embed_query(piece["text"]))
            meta.append({"text": piece["text"], **piece["metadata"]})
        raw_index.upsert(zip(ids, vectors, meta))
        print(f"✅  indexed {url}")


async def delete_all():
    index = get_raw_pinecone_index()
    # Delete all vectors in the index
    index.delete(delete_all=True)
    print("✅ All vectors deleted from the index.")


if __name__ == "__main__":
    asyncio.run(delete_all())
