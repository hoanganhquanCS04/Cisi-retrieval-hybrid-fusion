import json
from pathlib import Path

def compute_rr(ranked_list: list[dict], relevant_set: set[int]) -> float:
    for rank, item in enumerate(ranked_list, start=1):
        if item["doc_id"] in relevant_set:
            return 1.0 / rank
    return 0.0

def main():
    project_root = Path(__file__).resolve().parent
    
    # Load data
    with open(project_root / "data" / "queries.json", "r", encoding="utf-8") as f:
        queries = {int(q["query_id"]): q["text"] for q in json.load(f)}
        
    with open(project_root / "data" / "corpus.json", "r", encoding="utf-8") as f:
        corpus = {int(d["doc_id"]): d["text"] for d in json.load(f)}
        
    with open(project_root / "data" / "qrels.json", "r", encoding="utf-8") as f:
        raw_qrels = json.load(f)
        qrels = {int(qid): set(doc_ids) for qid, doc_ids in raw_qrels.items()}
        
    with open(project_root / "data" / "bm25_top100.json", "r", encoding="utf-8") as f:
        bm25_results = json.load(f)
        
    # Tính toán MRR cho từng query
    query_metrics = []
    for qid_str, ranked_list in bm25_results.items():
        qid = int(qid_str)
        if qid not in qrels:
            continue  # Bỏ qua các query không có ground truth
            
        relevant_set = qrels[qid]
        rr = compute_rr(ranked_list, relevant_set)
        
        query_metrics.append({
            "query_id": qid,
            "query_text": queries.get(qid, ""),
            "rr": rr,
            "relevant_docs": relevant_set,
            "retrieved_list": [doc["doc_id"] for doc in ranked_list]
        })
        
    # Sắp xếp List tăng dần theo RR (Càng thấp quy lỗi càng nặng)
    query_metrics.sort(key=lambda x: x["rr"])
    
    # Lấy 10 lỗi tệ nhất
    worst_10 = query_metrics[:10]
    
    print("=== TOP 10 QUERIES BỊ BM25 TRẢ VỀ TỆ NHẤT ===")
    for i, meta in enumerate(worst_10, 1):
        print(f"\n[{i}] Query ID: {meta['query_id']} | Điểm Reciprocal Rank (RR): {meta['rr']:.4f}")
        print(f"Câu hỏi: {meta['query_text']}")
        print(f"Danh sách đáp án chuẩn (Ground Truth Doc IDs): {list(meta['relevant_docs'])}")
        
        # In tóm tắt đoạn văn mô tả của đáp án đúng đầu tiên để đánh giá bằng mắt
        if meta['relevant_docs']:
            first_truth_doc = list(meta['relevant_docs'])[0]
            truth_text = corpus.get(first_truth_doc, "")
            print(f"==> Trích xuất đáp án chuẩn (Doc {first_truth_doc}): {truth_text[:200]}...")

if __name__ == "__main__":
    main()
