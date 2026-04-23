import json
import sys
from pathlib import Path
import argparse

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval.rrf import reciprocal_rank_fusion, weighted_fusion

def run_hybrid_pipeline(
    bm25_top100_path: str | Path,
    dense_top100_path: str | Path,
    output_path: str | Path,
    method: str = "rrf",
    k: int = 60,
    bm25_weight: float = 0.5,
    top_k: int = 100
):
    """
    Đọc output của BM25 và Dense, apply RRF để tạo ra danh sách Hybrid top_k.
    Schema Output chuẩn của team:
    { "query_id": [ {"doc_id": id, "score": rrf_score}, ... ], ... }
    """
    print(f"Loading BM25 results from: {bm25_top100_path}")
    with open(bm25_top100_path, 'r', encoding='utf-8') as f:
        bm25_results = json.load(f)
        
    print(f"Loading Dense results from: {dense_top100_path}")
    with open(dense_top100_path, 'r', encoding='utf-8') as f:
        dense_results = json.load(f)
        
    hybrid_dict = {}
    common_queries = sorted(list(set(bm25_results.keys()).intersection(set(dense_results.keys()))), key=int)
    print(f"Xử lý {len(common_queries)} queries...")

    for query_id in common_queries:
        # Giải nén list dict thành list tuple (doc_id, score) cho RRF
        bm25_list = [(doc["doc_id"], doc["score"]) for doc in bm25_results[query_id]]
        dense_list = [(doc["doc_id"], doc["score"]) for doc in dense_results[query_id]]
        
        if method == "rrf":
            fused = reciprocal_rank_fusion([bm25_list, dense_list], k=k)
        elif method == "weighted":
            fused = weighted_fusion(bm25_list, dense_list, bm25_weight=bm25_weight)
        else:
            raise ValueError("Method must be 'rrf' or 'weighted'")
        
        # Chỉ lấy top_k và trả về chuẩn Schema dict
        fused_top = fused[:top_k]
        hybrid_dict[str(query_id)] = [{"doc_id": doc_id, "score": score} for doc_id, score in fused_top]
        
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(hybrid_dict, f, indent=2)
    print(f"Đã lưu kết quả Hybrid vào: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply RRF or Weighted Fusion on BM25 and Dense output.")
    parser.add_argument("--bm25", default="data/bm25_top100.json")
    parser.add_argument("--dense", default="data/dense_top100.json")
    parser.add_argument("--output", default="data/hybrid_top100.json")
    parser.add_argument("--method", choices=["rrf", "weighted"], default="rrf")
    parser.add_argument("--k", type=int, default=60, help="Hằng số cho thuật toán RRF")
    parser.add_argument("--weight", type=float, default=0.5, help="Trọng số BM25 cho thuật toán Weighted Fusion (0.0 tới 1.0)")
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parents[1]
    run_hybrid_pipeline(
        bm25_top100_path=project_root / args.bm25,
        dense_top100_path=project_root / args.dense,
        output_path=project_root / args.output,
        method=args.method,
        k=args.k,
        bm25_weight=args.weight
    )
