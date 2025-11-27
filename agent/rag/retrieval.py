import os
from typing import List, Dict, Tuple
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np


class Retriever:
    def __init__(self, docs_path: str = "docs", min_chunk_len: int = 20):
        self.docs_path = Path(docs_path)
        self.min_chunk_len = min_chunk_len

        self.chunks: List[str] = []
        self.chunk_meta: List[Dict] = []
        self.vectorizer: TfidfVectorizer = None
        self.tfidf_matrix = None

    def _chunk_file(self, filename: Path) -> List[str]:
        text = filename.read_text(encoding="utf-8")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        raw_chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = [c for c in raw_chunks if len(c) >= self.min_chunk_len]
        return chunks

    def load_corpus(self):
        if not self.docs_path.exists():
            raise FileNotFoundError(f"Docs folder not found at {self.docs_path}")
        self.chunks = []
        self.chunk_meta = []

        md_files = sorted([p for p in self.docs_path.glob("*.md")])
        chunk_counter = 0
        for md in md_files:
            chunks = self._chunk_file(md)
            for i, c in enumerate(chunks):
                chunk_id = f"{md.name}::chunk_{i}"
                self.chunks.append(c)
                self.chunk_meta.append({
                    "chunk_id": chunk_id,
                    "source": md.name,
                    "index_in_file": i,
                })
                chunk_counter += 1

        if len(self.chunks) == 0:
            raise ValueError(f"No chunks found in {self.docs_path} - check files and min_chunk_len")

        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=8000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.tfidf_matrix is None:
            raise RuntimeError("Corpus not loaded. Call load_corpus() first.")

        q_vec = self.vectorizer.transform([query])
        sims = linear_kernel(q_vec, self.tfidf_matrix).flatten()
        if np.all(sims == 0):
            return []

        top_idx = np.argsort(-sims)[:top_k]
        results = []
        for idx in top_idx:
            results.append({
                "chunk_id": self.chunk_meta[idx]["chunk_id"],
                "source": self.chunk_meta[idx]["source"],
                "content": self.chunks[idx],
                "score": float(sims[idx])
            })
        return results

    def get_chunk(self, chunk_id: str) -> Dict:
        for meta, content in zip(self.chunk_meta, self.chunks):
            if meta["chunk_id"] == chunk_id:
                return {"chunk_id": meta["chunk_id"], "source": meta["source"], "content": content}
        raise KeyError(f"Chunk id not found: {chunk_id}")
