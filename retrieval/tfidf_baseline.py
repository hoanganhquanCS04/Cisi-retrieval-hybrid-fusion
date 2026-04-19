from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def _build_doc_text(doc: dict[str, Any]) -> str:
    """Ghép nội dung 1 document thành chuỗi đầu vào cho TF-IDF.

    Chức năng:
        - Kết hợp `title` và `text` để tăng tín hiệu truy hồi.
        - Nếu thiếu 1 trong 2 trường thì dùng trường còn lại.

    Input:
        doc (dict): 1 phần tử từ `corpus.json`, tối thiểu nên có:
            - doc_id (int)
            - title (str)
            - text (str)

    Output:
        str: Chuỗi văn bản đã ghép, ví dụ "title. text".
    """
    title = str(doc.get("title", "")).strip()
    text = str(doc.get("text", "")).strip()
    if title and text:
        return f"{title}. {text}"
    return title or text


def load_dataset(
    corpus_path: str | Path, queries_path: str | Path, qrels_path: str | Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, set[int]]]:
    """Đọc toàn bộ dữ liệu đầu vào cho baseline.

    Chức năng:
        - Load 3 file JSON chuẩn của M1: corpus, queries, qrels.
        - Chuẩn hóa qrels từ JSON object key-string sang `dict[int, set[int]]`.

    Input:
        corpus_path: Đường dẫn `data/corpus.json` (List[Dict]).
        queries_path: Đường dẫn `data/queries.json` (List[Dict]).
        qrels_path: Đường dẫn `data/qrels.json` (Dict[str, List[int]]).

    Yêu cầu dữ liệu đầu vào:
        - corpus[i] có các key: doc_id, title, author, text
        - queries[i] có các key: query_id, text
        - qrels có dạng {"query_id": [doc_id1, doc_id2, ...]}

    Output:
        tuple gồm:
            1) corpus: list[dict[str, Any]]
            2) queries: list[dict[str, Any]]
            3) qrels: dict[int, set[int]]
    """
    corpus = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    queries = json.loads(Path(queries_path).read_text(encoding="utf-8"))
    raw_qrels = json.loads(Path(qrels_path).read_text(encoding="utf-8"))
    qrels = {int(qid): {int(doc_id) for doc_id in doc_ids} for qid, doc_ids in raw_qrels.items()}
    return corpus, queries, qrels


def build_tfidf_index(corpus: list[dict[str, Any]]) -> tuple[TfidfVectorizer, Any, list[int]]:
    """Xây dựng TF-IDF index từ corpus.

    Chức năng:
        - Vector hóa toàn bộ tập document bằng `TfidfVectorizer`.
        - Trả về matrix để dùng cho truy hồi cosine similarity.

    Input:
        corpus: Danh sách document đã parse từ `corpus.json`.

    Yêu cầu dữ liệu đầu vào:
        - Mỗi phần tử có `doc_id` (int), `title`/`text` (str).

    Output:
        (vectorizer, doc_matrix, doc_ids)
        - vectorizer: TfidfVectorizer đã fit
        - doc_matrix: sparse matrix [num_docs, vocab_size]
        - doc_ids: list[int] ánh xạ index hàng -> doc_id thật
    """
    doc_ids = [int(doc["doc_id"]) for doc in corpus]
    documents = [_build_doc_text(doc) for doc in corpus]

    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
    doc_matrix = vectorizer.fit_transform(documents)
    return vectorizer, doc_matrix, doc_ids


def tfidf_retrieve(
    query_text: str,
    vectorizer: TfidfVectorizer,
    doc_matrix: Any,
    doc_ids: list[int],
    top_k: int = 100,
) -> list[tuple[int, float]]:
    """Truy hồi top-k tài liệu cho 1 query bằng TF-IDF + cosine.

    Chức năng:
        - Vector hóa query.
        - Tính điểm tương đồng cosine giữa query và toàn bộ document.
        - Sắp hạng giảm dần và lấy top_k.

    Input:
        query_text: Nội dung query (str).
        vectorizer: TfidfVectorizer đã fit trên corpus.
        doc_matrix: TF-IDF matrix của corpus.
        doc_ids: Danh sách doc_id tương ứng từng hàng trong doc_matrix.
        top_k: Số lượng kết quả trả về (mặc định 100).

    Output:
        list[tuple[int, float]]:
            Danh sách [(doc_id, score)] đã sort giảm dần theo score.
    """
    query_vector = vectorizer.transform([query_text])
    scores = linear_kernel(query_vector, doc_matrix).ravel()
    ranked_indices = scores.argsort()[::-1][:top_k]
    return [(doc_ids[i], float(scores[i])) for i in ranked_indices]


def _reciprocal_rank(ranked_doc_ids: list[int], relevant: set[int]) -> float:
    """Tính Reciprocal Rank (RR) cho 1 query.

    Input:
        ranked_doc_ids: Danh sách doc_id theo thứ hạng dự đoán.
        relevant: Tập doc_id đúng (ground-truth) cho query.

    Output:
        float:
            - 1/rank của doc relevant đầu tiên nếu có hit.
            - 0.0 nếu không có doc relevant nào trong danh sách xếp hạng.
    """
    if not relevant:
        return 0.0
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def _precision_at_k(ranked_doc_ids: list[int], relevant: set[int], k: int = 10) -> float:
    """Tính Precision@k cho 1 query.

    Input:
        ranked_doc_ids: Danh sách doc_id theo thứ hạng dự đoán.
        relevant: Tập doc_id đúng cho query.
        k: Ngưỡng cutoff, mặc định 10.

    Output:
        float: số lượng tài liệu đúng trong top-k chia cho k.
    """
    if k <= 0:
        return 0.0
    if not ranked_doc_ids:
        return 0.0
    hits = sum(1 for doc_id in ranked_doc_ids[:k] if doc_id in relevant)
    return hits / k


def evaluate_tfidf(
    queries: list[dict[str, Any]],
    qrels: dict[int, set[int]],
    vectorizer: TfidfVectorizer,
    doc_matrix: Any,
    doc_ids: list[int],
    top_k: int = 100,
) -> dict[str, Any]:
    """Đánh giá TF-IDF baseline trên toàn bộ query set.

    Chức năng:
        - Chạy retrieve cho từng query.
        - Tính RR và P@10 per-query.
        - Tổng hợp MRR và mean P@10 toàn tập.

    Input:
        queries: List query dict (ít nhất có query_id, text).
        qrels: Dict[int, set[int]] ground-truth relevance.
        vectorizer/doc_matrix/doc_ids: output từ `build_tfidf_index`.
        top_k: số lượng document retrieve cho mỗi query.

    Output:
        dict metrics với schema:
            {
              "num_queries": int,
              "mrr": float,
              "p@10": float,
              "top_k": int,
              "per_query": [{"query_id": int, "rr": float, "p@10": float}, ...]
            }
    """
    rr_scores: list[float] = []
    p10_scores: list[float] = []
    per_query: list[dict[str, float | int]] = []

    for q in queries:
        query_id = int(q["query_id"])
        query_text = str(q["text"])
        ranked = tfidf_retrieve(query_text, vectorizer, doc_matrix, doc_ids, top_k=top_k)
        ranked_doc_ids = [doc_id for doc_id, _ in ranked]
        relevant = qrels.get(query_id, set())

        rr = _reciprocal_rank(ranked_doc_ids, relevant)
        p10 = _precision_at_k(ranked_doc_ids, relevant, k=10)

        rr_scores.append(rr)
        p10_scores.append(p10)
        per_query.append({"query_id": query_id, "rr": rr, "p@10": p10})

    mrr = sum(rr_scores) / len(rr_scores) if rr_scores else 0.0
    p10_mean = sum(p10_scores) / len(p10_scores) if p10_scores else 0.0
    return {
        "num_queries": len(queries),
        "mrr": mrr,
        "p@10": p10_mean,
        "top_k": top_k,
        "per_query": per_query,
    }


def run_tfidf_baseline(
    corpus_path: str | Path,
    queries_path: str | Path,
    qrels_path: str | Path,
    output_path: str | Path,
    top_k: int = 100,
) -> dict[str, Any]:
    """Chạy full pipeline baseline và ghi metrics ra file.

    Chức năng:
        1) Load dữ liệu từ JSON.
        2) Build TF-IDF index.
        3) Evaluate MRR/P@10.
        4) Save kết quả vào `reports/baseline_metrics.json` (hoặc path truyền vào).

    Input:
        corpus_path: file `corpus.json`.
        queries_path: file `queries.json`.
        qrels_path: file `qrels.json`.
        output_path: file metrics JSON đầu ra.
        top_k: số lượng doc retrieve cho mỗi query.

    Output:
        dict metrics cùng schema như `evaluate_tfidf`.
    """
    corpus, queries, qrels = load_dataset(corpus_path, queries_path, qrels_path)
    vectorizer, doc_matrix, doc_ids = build_tfidf_index(corpus)
    metrics = evaluate_tfidf(queries, qrels, vectorizer, doc_matrix, doc_ids, top_k=top_k)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    """Điểm vào CLI cho Task 2.1 + 2.2.

    Input (từ command line):
        --corpus-path, --queries-path, --qrels-path, --output-path, --top-k

    Output:
        - File JSON metrics tại `output-path`
        - Log tóm tắt trên terminal: num_queries, MRR, P@10
    """
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run TF-IDF baseline and save MRR/P@10 metrics")
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=project_root / "data" / "corpus.json",
    )
    parser.add_argument(
        "--queries-path",
        type=Path,
        default=project_root / "data" / "queries.json",
    )
    parser.add_argument(
        "--qrels-path",
        type=Path,
        default=project_root / "data" / "qrels.json",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=project_root / "reports" / "baseline_metrics.json",
    )
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    metrics = run_tfidf_baseline(
        corpus_path=args.corpus_path,
        queries_path=args.queries_path,
        qrels_path=args.qrels_path,
        output_path=args.output_path,
        top_k=args.top_k,
    )
    print(f"[OK] Saved metrics: {args.output_path}")
    print(f"[OK] Queries: {metrics['num_queries']}")
    print(f"[OK] MRR: {metrics['mrr']:.6f}")
    print(f"[OK] P@10: {metrics['p@10']:.6f}")


if __name__ == "__main__":
    main()
