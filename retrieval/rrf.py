from __future__ import annotations

def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[int, float]]], 
    k: int = 60
) -> list[tuple[int, float]]:
    """
    Thuật toán Reciprocal Rank Fusion (RRF) dùng để lai ghép N danh sách hạng.
    Lấy ý tưởng cấu trúc trực tiếp từ knowledge_baseline.md.
    
    Công thức: RRF_Score = sum( 1 / (k + rank) ) cho tất cả các hệ thống retrieve.
    
    Args:
        ranked_lists: Danh sách các mảng xếp hạng. Mỗi mảng xếp hạng là list các tuple (doc_id, điểm).
                      Giả định các list đã được sắp xếp giảm dần theo điểm (Top 1 đứng đầu).
        k: Hằng số smoothing của thuật toán (mặc định k=60 là chuẩn nhất theo bài báo RRF).
        
    Returns:
        Một danh sách hợp nhất `[(doc_id, rrf_score), ...]` đã được sắp xếp giảm dần theo rrf_score.
    """
    rrf_scores: dict[int, float] = {}
    
    for ranked_list in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked_list, start=1):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (k + rank)
            
    # Sort descending
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results

def weighted_fusion(
    bm25_list: list[tuple[int, float]],
    dense_list: list[tuple[int, float]],
    bm25_weight: float = 0.5
) -> list[tuple[int, float]]:
    """
    Lai ghép bằng tổng có trọng số (Weighted Sum) sau khi Min-Max Normalization.
    
    Args:
        bm25_list: Danh sách kết quả BM25 [(doc_id, score), ...]
        dense_list: Danh sách kết quả Dense [(doc_id, score), ...]
        bm25_weight: Tỷ trọng điểm cho BM25 (0.0 đến 1.0). Dense sẽ là 1 - bm25_weight.
        
    Returns:
        Danh sách hợp nhất [(doc_id, combined_score), ...]
    """
    bm25_scores = {doc_id: score for doc_id, score in bm25_list}
    dense_scores = {doc_id: score for doc_id, score in dense_list}
    
    # Min-Max Normalization Function
    def normalize(scores: dict[int, float]) -> dict[int, float]:
        if not scores: return {}
        min_s, max_s = min(scores.values()), max(scores.values())
        if max_s == min_s: return {k: 1.0 for k in scores}
        return {k: (v - min_s) / (max_s - min_s) for k, v in scores.items()}
        
    bm25_norm = normalize(bm25_scores)
    dense_norm = normalize(dense_scores)
    
    dense_weight = 1.0 - bm25_weight
    all_docs = set(bm25_norm.keys()) | set(dense_norm.keys())
    
    fused_scores = {}
    for doc_id in all_docs:
        s_bm25 = bm25_norm.get(doc_id, 0.0)
        s_dense = dense_norm.get(doc_id, 0.0)
        fused_scores[doc_id] = bm25_weight * s_bm25 + dense_weight * s_dense
        
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
