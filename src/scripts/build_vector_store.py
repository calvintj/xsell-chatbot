# src/scripts/build_vector_store.py
"""
Embed a CSV / DataFrame and upsert to Pinecone.
Run once after you add new docs:  python -m src.scripts.build_vector_store
"""

import os
from dotenv import load_dotenv
import pandas as pd
from tqdm.auto import tqdm
from langchain_openai import OpenAIEmbeddings
from core.settings import settings
from retrieval.vector_store import get_raw_pinecone_index   # helper shown below

load_dotenv()  # in case you run this outside the main app

EMBED_MODEL = OpenAIEmbeddings(model="text-embedding-ada-002")
BATCH_SIZE = 100

def main() -> None:
    index = get_raw_pinecone_index()            # same index the app queries
    if index.describe_index_stats().total_vector_count:
        print("✅  Pinecone already populated – nothing to do.")
        return

    # ── 1. Load your raw data ──────────────────────────────────────────────────
    df: pd.DataFrame = pd.read_csv("data/raw/llama2_chunks.csv")
    print(f"Embedding {len(df):,} chunks …")

    # ── 2. Batch-embed & upsert ────────────────────────────────────────────────
    for i in tqdm(range(0, len(df), BATCH_SIZE)):
        part = df.iloc[i : i + BATCH_SIZE]
        ids      = [f"{row.doi}-{row['chunk-id']}" for _, row in part.iterrows()]
        texts    = part["chunk"].tolist()
        embeds   = EMBED_MODEL.embed_documents(texts)
        metadata = (
            part[["chunk", "source", "title"]]
            .rename(columns={"chunk": "text"})
            .to_dict(orient="records")
        )
        index.upsert(vectors=zip(ids, embeds, metadata))

    print("🎉  Finished uploading vectors!")

if __name__ == "__main__":
    main()
