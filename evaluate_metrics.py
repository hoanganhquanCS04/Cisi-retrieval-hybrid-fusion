import json
from pathlib import Path
import argparse

def compute_metrics(ranked_list: list[dict], relevant_set: set[int]) -> dict:
    """Tính các metric MRR, P@10, Recall@100 cho một query."""
    if not relevant_set:
        return {"rr": 0.0, "p_10": 0.0, "recall": 0.0}

    # Tính Reciprocal Rank (RR)
    rr = 0.0
    for rank, item in enumerate(ranked_list, start=1):
        if item["doc_id"] in relevant_set:
            rr = 1.0 / rank
            break

    # Tính Precision@10
    top_10 = [item["doc_id"] for item in ranked_list[:10]]
    hits_in_10 = sum(1 for doc_id in top_10 if doc_id in relevant_set)
    p_10 = hits_in_10 / 10.0

    # Tính Recall@100 (vì ranked_list đã là top 100)
    top_100 = [item["doc_id"] for item in ranked_list]
    hits_in_100 = sum(1 for doc_id in top_100 if doc_id in relevant_set)
    recall = hits_in_100 / len(relevant_set)

    return {"rr": rr, "p_10": p_10, "recall": recall}

def evaluate_system(qrels_path: str, results_path: str, output_report_path: str):
    """Đánh giá toàn bộ hệ thống và xuất report JSON."""
    with open(qrels_path, "r", encoding="utf-8") as f:
        raw_qrels = json.load(f)
        qrels = {int(qid): set(doc_ids) for qid, doc_ids in raw_qrels.items()}
        
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    total_rr = 0.0
    total_p10 = 0.0
    total_recall = 0.0
    valid_queries = 0

    for qid_str, ranked_list in results.items():
        qid = int(qid_str)
        if qid not in qrels or not qrels[qid]:
            continue

        metrics = compute_metrics(ranked_list, qrels[qid])
        total_rr += metrics["rr"]
        total_p10 += metrics["p_10"]
        total_recall += metrics["recall"]
        valid_queries += 1

    if valid_queries == 0:
        print("Không tìm thấy query nào hợp lệ trong qrels!")
        return

    mrr = total_rr / valid_queries
    avg_p10 = total_p10 / valid_queries
    avg_recall = total_recall / valid_queries

    report = {
        "MRR": round(mrr, 4),
        "P@10": round(avg_p10, 4),
        "Recall@100": round(avg_recall, 4),
        "evaluated_queries": valid_queries
    }

    print(f"--- Đánh giá File {results_path} ---")
    print(f"MRR:        {report['MRR']}")
    print(f"P@10:       {report['P@10']}")
    print(f"Recall@100: {report['Recall@100']}")
    
    Path(output_report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    print(f"Đã lưu báo cáo tại {output_report_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrels", default="data/qrels.json")
    parser.add_argument("--results", required=True, help="Đường dẫn đến file JSON của BM25, Dense hoặc Hybrid")
    parser.add_argument("--output", required=True, help="Đường dẫn file lưu report (.json)")
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parent
    evaluate_system(
        qrels_path=project_root / args.qrels,
        results_path=project_root / args.results,
        output_report_path=project_root / args.output
    )
