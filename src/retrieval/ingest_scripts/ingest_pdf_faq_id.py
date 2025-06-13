# PYTHONPATH=. python3 -m src.scripts.ingest_pdf_faq_id
import hashlib
import logging
import math
import re
import sys

import fitz
import pinecone
import tqdm
from langchain_openai import OpenAIEmbeddings

from src.retrieval.vector_store import get_raw_pinecone_index  # your helper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PDF_PATH = "src/data/raw/FAQ_FCY_Jenius_id.pdf"
SOURCE_URL = (
    "https://drive.google.com/uc?export=download"
    "&id=1mFmcDTmzeSwso-apDS8rLAKcozNvdmjJ"
)

QUESTION_MIN = 13.5  # ≥14-pt is a question
ANSWER_MAX = 13.49  # ≤13.5-pt is body

embedder = OpenAIEmbeddings(model="text-embedding-3-large")
index = get_raw_pinecone_index()
BATCH = 100


def iter_qna_blocks(doc, question_min=13.5):
    """
    Yield (question, answer, page_num) tuples.
    Any line whose average font-size >= question_min is considered part of the question.
    We only yield when the question ends with '?' and we've collected its answers.
    """
    pending_q, pending_a, page_num = None, [], None

    for page_idx in range(doc.page_count):
        page = doc.load_page(page_idx)
        blocks = page.get_text("dict")["blocks"]

        for b in blocks:
            for line in b["lines"]:
                # 1) collapse all spans on this line
                spans = [s for s in line["spans"] if s["text"].strip()]
                line_text = " ".join(s["text"].strip() for s in spans)
                if not line_text:
                    continue
                # 2) compute an average font-size for the line
                avg_size = sum(s["size"] for s in spans) / len(spans)

                if avg_size >= question_min:
                    # it's a question line
                    # if we already had a complete Q (ending in '?'), flush it
                    if pending_q and pending_q.strip().endswith("?"):
                        yield pending_q, " ".join(pending_a), page_num
                        pending_a = []
                        # start brand-new question
                        pending_q = line_text
                        page_num = page_idx + 1
                    else:
                        # accumulate multi-line question
                        pending_q = (
                            (pending_q + " " + line_text) if pending_q else line_text
                        )
                        page_num = page_idx + 1
                else:
                    # everything smaller is answer content
                    if pending_q is not None:
                        pending_a.append(line_text)

    # flush the very last Q&A if it ended in '?'
    if pending_q and pending_q.strip().endswith("?"):
        yield pending_q, " ".join(pending_a), page_num


def build_vectors(pdf_path):
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Failed to open PDF file: {pdf_path}. Error: {e}")
        return
    for q, a, pg in iter_qna_blocks(doc):
        chunk = a
        uid = hashlib.sha1(chunk.encode()).hexdigest()
        meta = {
            "source": f"{SOURCE_URL}#page={pg}",
            "lang": "id",  # this PDF is Bahasa
            "question": q,
            "text": chunk,
        }
        yield uid, chunk, meta
    doc.close()


def main():
    ids, texts, metas = [], [], []
    for uid, txt, meta in build_vectors(PDF_PATH):
        ids.append(uid)
        texts.append(txt)
        metas.append(meta)

    logger.info(f"Total Q&A chunks: {len(ids)}")
    for i in tqdm.tqdm(range(0, len(texts), BATCH)):
        try:
            vecs = embedder.embed_documents(texts[i : i + BATCH])
            index.upsert(
                zip(ids[i : i + BATCH], vecs, metas[i : i + BATCH]), namespace="id"
            )
        except Exception as e:
            logger.error(f"Failed to embed or upsert batch {i}-{i+BATCH}: {e}")


def delete_all():
    try:
        index.delete(delete_all=True, namespace="id")
        logger.info("Deleted all vectors")
    except Exception as e:
        logger.error(f"Failed to delete vectors: {e}")


if __name__ == "__main__":
    main()
