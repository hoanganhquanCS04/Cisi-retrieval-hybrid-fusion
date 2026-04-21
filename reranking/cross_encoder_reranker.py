import os
import json
import torch
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name_or_path: str = 'hf_cache/cross_encoder_minilm', device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        # Load from local cache, fallback to online if not found
        if not os.path.exists(model_name_or_path):
            model_name_or_path = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.model = CrossEncoder(model_name_or_path, device=device)

    def build_pairs(self, query_text: str, doc_texts: list[str]) -> list[tuple[str, str]]:
        return [(query_text, doc) for doc in doc_texts]

    def rerank(self, query_text: str, candidate_docs: list[dict], batch_size: int = 32) -> list[dict]:
        """
        candidate_docs: list of dict, each dict has 'doc_id' and 'text'.
        return: list of candidate_docs sorted by CE score in descending order.
        """
        doc_texts = [doc['text'] for doc in candidate_docs]
        pairs = self.build_pairs(query_text, doc_texts)
        scores = self.model.predict(pairs, batch_size=batch_size)
        
        # Merge scores and sort
        scored_docs = []
        for i, doc in enumerate(candidate_docs):
            scored_doc = doc.copy()
            scored_doc['ce_score'] = float(scores[i])
            scored_docs.append(scored_doc)
            
        scored_docs.sort(key=lambda x: x['ce_score'], reverse=True)
        return scored_docs
