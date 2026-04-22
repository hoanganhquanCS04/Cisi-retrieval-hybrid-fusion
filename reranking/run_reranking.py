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
    else:
        print("No hybrid_top100.json found. Falling back to bm25_top100.json.")
        with open('data/bm25_top100.json', 'r', encoding='utf-8') as f:
            candidate_results = json.load(f)
        
    return corpus, queries, candidate_results

def main():
    print("Loading data...")
    corpus, queries, candidate_results = load_data()
    
    print("Initializing Rerankers...")
    ce_reranker = CrossEncoderReranker()
    monot5_reranker = MonoT5Reranker()
    
    ce_final_results = {}
    monot5_final_results = {}
    
    ce_total_time = 0
    t5_total_time = 0
    
    from tqdm import tqdm
    
    print("Running Re-ranking...")
    for qid_str, top_docs in tqdm(candidate_results.items(), desc="Re-ranking Queries"):
        qid = int(qid_str) if qid_str.isdigit() else qid_str
        # Some query ids might not be in queries due to lack of text, skip if so
        if qid not in queries and str(qid) not in queries:
            continue
            
        q_text = queries.get(qid) or queries.get(str(qid))
        
        # Prepare candidate docs
        candidates = []
        for d in top_docs:
            doc_id = d['doc_id']
            if doc_id in corpus:
                candidates.append({
                    "doc_id": doc_id,
                    "text": corpus[doc_id]['text'],
                    "bm25_score": d['score']
                })
        
        if not candidates:
            continue
            
        # Cross-Encoder (tăng tốc bằng batch_size lớn cho RTX 4090)
        start_time = time.time()
        ce_ranked = ce_reranker.rerank(q_text, candidates, batch_size=256)
        ce_total_time += (time.time() - start_time)
        ce_final_results[str(qid)] = [{"doc_id": d["doc_id"], "score": d["ce_score"]} for d in ce_ranked]
        
        # MonoT5 (tăng tốc bằng batch_size lớn cho RTX 4090)
        start_time = time.time()
        t5_ranked = monot5_reranker.rerank(q_text, candidates, batch_size=64)
        t5_total_time += (time.time() - start_time)
        monot5_final_results[str(qid)] = [{"doc_id": d["doc_id"], "score": d["monot5_score"]} for d in t5_ranked]

    # Save results
    print("Saving results...")
    with open('data/ce_reranked.json', 'w', encoding='utf-8') as f:
        json.dump(ce_final_results, f, indent=2)
        
    with open('data/monot5_reranked.json', 'w', encoding='utf-8') as f:
        json.dump(monot5_final_results, f, indent=2)
        
    # Build reports
    num_queries = len(candidate_results)
    ce_latency_ms = (ce_total_time / max(num_queries, 1)) * 1000
    t5_latency_ms = (t5_total_time / max(num_queries, 1)) * 1000
    
    with open('reports/ce_latency.md', 'w') as f:
        f.write(f"# Cross-Encoder Latency Report\n\nTotal time: {ce_total_time:.2f}s\nAverage latency per query: {ce_latency_ms:.2f} ms")
        
    with open('reports/reranker_latency_comparison.md', 'w') as f:
        f.write(f"# Reranker Latency Comparison\n\n| Model | Avg Latency/Query (ms) |\n|---|---|\n| Cross-Encoder | {ce_latency_ms:.2f} |\n| MonoT5 | {t5_latency_ms:.2f} |")

if __name__ == "__main__":
    main()
