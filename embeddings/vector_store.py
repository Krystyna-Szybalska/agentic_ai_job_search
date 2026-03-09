import json
import os
import faiss
import numpy as np


class VectorStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.jobs: list[dict] = []

    def build(self, jobs: list[dict], embeddings: list[list[float]]) -> None:
        """Build FAISS index from job embeddings."""
        self.jobs = jobs
        vectors = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(vectors)
        self.index = faiss.IndexFlatIP(self.dimension)  # inner product = cosine after normalization
        self.index.add(vectors)

    def query(self, embedding: list[float], k: int = 10) -> list[dict]:
        """Return top-K jobs for a query embedding."""
        vector = np.array([embedding], dtype="float32")
        faiss.normalize_L2(vector)
        scores, indices = self.index.search(vector, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            job = dict(self.jobs[idx])
            job["vector_score"] = float(score)
            results.append(job)
        return results

    def save(self, path: str) -> None:
        """Save index and job metadata to disk."""
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "jobs.json"), "w", encoding="utf-8") as f:
            json.dump(self.jobs, f, ensure_ascii=False)

    def load(self, path: str) -> None:
        """Load index and job metadata from disk."""
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "jobs.json"), encoding="utf-8") as f:
            self.jobs = json.load(f)
