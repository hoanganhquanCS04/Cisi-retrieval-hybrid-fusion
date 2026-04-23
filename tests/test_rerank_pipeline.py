import sys
import os
import json
import time

# Add root project dir to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from reranking.cross_encoder_reranker import CrossEncoderReranker
from reranking.monot5_reranker import MonoT5Reranker

def load_data():
    with open('data/corpus.json', 'r', encoding='utf-8') as f:
        corpus = {d['doc_id']: d for d in json.load(f)}
    
    with open('data/queries.json', 'r', encoding='utf-8') as f:
        queries = {q['query_id']: q['text'] for q in json.load(f)}
        
    if os.path.exists('data/hybrid_top100.json'):
        print("Found hybrid_top100.json! Using Hybrid ranking as candidate list.")
        with open('data/hybrid_top100.json', 'r', encoding='utf-8') as f:
            candidate_results = json.load(f)
    elif os.path.exists('data/dense_top100.json'):
        print("No hybrid_top100.json found. Found dense_top100.json! Using Dense ranking as candidate list.")
        with open('data/dense_top100.json', 'r', encoding='utf-8') as f:
            candidate_results = json.load(f)
    else:
        print("No hybrid/dense config found. Falling back to bm25_top100.json.")
        with open('data/bm25_top100.json', 'r', encoding='utf-8') as f:
            candidate_results = json.load(f)
        
    return corpus, queries, candidate_results

def main():
    print("Loading data...")
    corpus, queries, candidate_results = load_data()
    
    print("Initializing Rerankers...")
    ce_reranker = CrossEncoderReranker()
    monot5_reranker = MonoT5Reranker()
    
    # Test on first 5 queries only for quick testing
    test_queries = list(candidate_results.keys())[:5]
    print(f"Testing on {len(test_queries)} queries...")
    
    ce_total_time = 0
    t5_total_time = 0
    
    for qid_str in test_queries:
        qid = int(qid_str) if qid_str.isdigit() else qid_str
        if qid not in queries and str(qid) not in queries:
            continue
            
        q_text = queries.get(qid) or queries.get(str(qid))
        top_docs = candidate_results[qid_str]
        
        # Prepare candidate docs (take top 10 for faster testing)
        candidates = []
        for d in top_docs[:10]:  # Only top 10 candidates
            doc_id = d['doc_id']
            if doc_id in corpus:
                candidates.append({
                    "doc_id": doc_id,
                    "text": corpus[doc_id]['text'],
                    "bm25_score": d['score']
                })
        
        if not candidates:
            continue
            
        print(f"Testing query {qid}...")
        
        # Cross-Encoder test
        start_time = time.time()
        ce_ranked = ce_reranker.rerank(q_text, candidates, batch_size=256)
        ce_total_time += (time.time() - start_time)
        
        # MonoT5 test
        start_time = time.time()
        t5_ranked = monot5_reranker.rerank(q_text, candidates, batch_size=64)
        t5_total_time += (time.time() - start_time)
        
        # Basic assertions
        assert len(ce_ranked) == len(candidates), f"CE output length mismatch for query {qid}"
        assert len(t5_ranked) == len(candidates), f"MonoT5 output length mismatch for query {qid}"
        assert all('ce_score' in d for d in ce_ranked), f"CE missing scores for query {qid}"
        assert all('monot5_score' in d for d in t5_ranked), f"MonoT5 missing scores for query {qid}"
    
    # Calculate and display latency
    num_tested = len(test_queries)
    ce_latency_ms = (ce_total_time / max(num_tested, 1)) * 1000
    t5_latency_ms = (t5_total_time / max(num_tested, 1)) * 1000
    
    print("\nTest Results:")
    print(f"Queries tested: {num_tested}")
    print(f"Cross-Encoder avg latency: {ce_latency_ms:.2f} ms/query")
    print(f"MonoT5 avg latency: {t5_latency_ms:.2f} ms/query")
    print("All tests passed! ✅")

if __name__ == "__main__":
    main()
