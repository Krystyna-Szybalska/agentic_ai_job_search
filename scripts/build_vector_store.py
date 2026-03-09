"""
Run once to build the FAISS vector store from jobs_dataset.json.
Usage: python scripts/build_vector_store.py
"""
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings.embedder import get_embedder, embed_batch
from embeddings.vector_store import VectorStore
from config.settings import JOBS_DATA_PATH, VECTOR_STORE_PATH


def build():
    print("Loading jobs...")
    with open(JOBS_DATA_PATH, encoding="utf-8") as f:
        raw_jobs = json.load(f)

    # Add a stable id and build text_for_embedding
    jobs = []
    texts = []
    for i, job in enumerate(raw_jobs):
        job["id"] = str(i)
        jobs.append(job)
        text = f"{job.get('positionName', '')}. {job.get('description', '')}"
        texts.append(text)

    print(f"Embedding {len(jobs)} jobs (this may take a minute)...")
    embedder = get_embedder()
    embeddings = embed_batch(texts, embedder)

    print("Building FAISS index...")
    store = VectorStore(dimension=384)
    store.build(jobs, embeddings)

    print(f"Saving to {VECTOR_STORE_PATH}...")
    store.save(VECTOR_STORE_PATH)
    print("Done! Vector store built successfully.")


if __name__ == "__main__":
    build()
