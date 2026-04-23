from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import sys

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval.base_retriever import BaseRetriever

class DenseRetriever(BaseRetriever):
    """
    Dense Retrieval thực hiện vector hóa và tìm kiếm ngữ nghĩa FAISS.
    Sử dụng SentenceTransformer và IndexFlatIP (Inner Product = Cosine cho vectors đã chuẩn hóa L2).
    """
    def __init__(self, model_path_or_name: str = 'all-MiniLM-L6-v2', device: str = 'cuda'):
        self.model = SentenceTransformer(model_path_or_name, device=device)
        self.index: faiss.IndexFlatIP | None = None
        self.doc_ids: list[int] = []

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2 normalize vectors để tính Cosine Similarity trên FAISS FlatIP."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vectors / norms

    def build_index(self, corpus: list[dict[str, Any]], batch_size: int = 64) -> np.ndarray:
        """Mã hóa văn bản và xây dựng FAISS Index."""
        self.doc_ids = [int(doc['doc_id']) for doc in corpus]
        
        # Combine title and text for embedding
        texts = []
        for doc in corpus:
            title = str(doc.get('title', '')).strip()
            text = str(doc.get('text', '')).strip()
            # Giống như baseline, nếu có title thì ghép với text
            source_text = f"{title}. {text}" if title and text else (title or text)
            texts.append(source_text)

        print(f"Bắt đầu encode {len(texts)} văn bản, batch_size={batch_size}...")
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        embeddings = self._normalize(embeddings)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        print(f"Đã xây dựng xong FAISS Index, tổng số vector: {self.index.ntotal}")
        
        return embeddings

    def retrieve(self, query: str, top_k: int = 100) -> list[tuple[int, float]]:
        """Truy hồi top-k documents gần nghĩa nhất với query."""
        if self.index is None:
            raise ValueError("Bạn chưa build_index() hoặc tải index sẵn.")

        query_embedding = self.model.encode([query])
        query_embedding = self._normalize(np.array(query_embedding).astype('float32'))

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, doc_idx in enumerate(indices[0]):
            if doc_idx >= 0 and doc_idx < len(self.doc_ids):
                results.append((self.doc_ids[doc_idx], float(scores[0][i])))

        return results

    def save_index(self, faiss_path: str | Path, doc_ids_path: str | Path):
        faiss.write_index(self.index, str(faiss_path))
        with open(doc_ids_path, 'w', encoding='utf-8') as f:
            json.dump(self.doc_ids, f)

    def load_index(self, faiss_path: str | Path, doc_ids_path: str | Path):
        self.index = faiss.read_index(str(faiss_path))
        with open(doc_ids_path, 'r', encoding='utf-8') as f:
            self.doc_ids = json.load(f)
