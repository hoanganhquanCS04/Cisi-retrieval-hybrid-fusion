import argparse
import json
from pathlib import Path
import sys
from typing import Any
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval.tokenizer import build_stopword_set, tokenize_corpus, tokenize_text


@dataclass
class BM25Runtime:
    """Runtime object giữ trạng thái BM25 để gọi retrieve nhiều lần.

    Fields:
        bm25: BM25Okapi index đã build.
        doc_ids: Ánh xạ index nội bộ -> doc_id thật.
        apply_stemming: Cấu hình stemming dùng khi tokenize query.
        use_stopwords: Cấu hình stopword removal dùng khi tokenize query.
    """

    bm25: BM25Okapi
    doc_ids: list[int]
    apply_stemming: bool = False
    use_stopwords: bool = True


_DEFAULT_RUNTIME: BM25Runtime | None = None


def load_corpus(corpus_path: str | Path) -> list[dict[str, Any]]:
    """Đọc corpus JSON làm đầu vào cho BM25.

    Chức năng:
        - Load file `data/corpus.json` (đầu ra Task 1.6).

    Input:
        corpus_path: Đường dẫn tới file corpus.json.

    Yêu cầu dữ liệu đầu vào:
        - File phải là JSON array.
        - Mỗi phần tử nên có các key: `doc_id`, `title`, `author`, `text`.

    Output:
        list[dict[str, Any]]: Danh sách document đã parse từ JSON.
    """
    return json.loads(Path(corpus_path).read_text(encoding="utf-8"))


def build_bm25_index(
    corpus: list[dict[str, Any]],
    *,
    apply_stemming: bool = False,
    include_title: bool = True,
    use_stopwords: bool = True,
    k1: float = 1.5,
    b: float = 1.0,
) -> tuple[BM25Okapi, list[int], list[list[str]]]:
    """Khởi tạo BM25Okapi index trên toàn bộ corpus.

    Chức năng:
        - Chạy pipeline tokenization (lowercase + stopword removal + stemming tùy chọn).
        - Dùng tokenized corpus để tạo `BM25Okapi`.

    Input:
        corpus: Danh sách document từ `corpus.json`.
        apply_stemming: Bật/tắt stemming.
        include_title: Nếu True thì ghép title vào text trước khi tokenize.
        use_stopwords: Nếu True thì loại stopwords tiếng Anh (NLTK).
        k1, b: Siêu tham số BM25.

    Yêu cầu dữ liệu đầu vào:
        - corpus chứa đủ tài liệu cần index (CISI chuẩn là 1460 docs).
        - Mỗi doc có `doc_id` hợp lệ.

    Output:
        tuple:
            - bm25: BM25Okapi index đã fit
            - doc_ids: list[int], ánh xạ index nội bộ -> doc_id
            - tokenized_docs: list[list[str]], tokenized corpus dùng để xây index
    """
    stopword_set = build_stopword_set(download_if_missing=True) if use_stopwords else None
    doc_ids, tokenized_docs = tokenize_corpus(
        corpus=corpus,
        stopword_set=stopword_set,
        apply_stemming=apply_stemming,
        include_title=include_title,
    )

    bm25 = BM25Okapi(tokenized_docs, k1=k1, b=b)
    return bm25, doc_ids, tokenized_docs


def init_default_bm25(
    *,
    corpus_path: str | Path,
    apply_stemming: bool = False,
    include_title: bool = True,
    use_stopwords: bool = True,
    k1: float = 1.5,
    b: float = 1.0,
) -> BM25Runtime:
    """Khởi tạo default BM25 runtime cho API `bm25_retrieve(query, top_k=100)`.

    Chức năng:
        - Load corpus từ file.
        - Build BM25 index 1 lần.
        - Lưu vào biến module-level để các lần retrieve sau gọi nhanh.

    Input:
        corpus_path: Đường dẫn `data/corpus.json`.
        apply_stemming/include_title/use_stopwords: cấu hình tokenize.
        k1, b: tham số BM25.

    Output:
        BM25Runtime: runtime vừa khởi tạo và được set làm default runtime.
    """
    global _DEFAULT_RUNTIME
    corpus = load_corpus(corpus_path)
    bm25, doc_ids, _ = build_bm25_index(
        corpus=corpus,
        apply_stemming=apply_stemming,
        include_title=include_title,
        use_stopwords=use_stopwords,
        k1=k1,
        b=b,
    )
    _DEFAULT_RUNTIME = BM25Runtime(
        bm25=bm25,
        doc_ids=doc_ids,
        apply_stemming=apply_stemming,
        use_stopwords=use_stopwords,
    )
    return _DEFAULT_RUNTIME


def bm25_retrieve(
    query: str,
    top_k: int = 100,
    *,
    runtime: BM25Runtime | None = None,
    bm25: BM25Okapi | None = None,
    doc_ids: list[int] | None = None,
    apply_stemming: bool | None = None,
    use_stopwords: bool | None = None,
) -> list[tuple[int, float]]:
    """Truy hồi top-k document cho 1 query bằng BM25 (Task 2.7).

    Chức năng:
        - Tokenize query theo cùng cấu hình đã dùng khi build index.
        - Tính BM25 score trên toàn bộ documents.
        - Trả về danh sách doc_id đã sắp xếp giảm dần theo score.

    Input:
        query: Nội dung query raw text.
        top_k: Số lượng kết quả cần lấy.
        runtime: BM25Runtime đã init (khuyên dùng).
        bm25/doc_ids: Tương thích ngược với cách gọi cũ nếu không truyền runtime.
        apply_stemming/use_stopwords:
            - Nếu truyền runtime: mặc định lấy từ runtime.
            - Nếu truyền bm25/doc_ids trực tiếp: phải truyền rõ hoặc để mặc định False/True.

    Yêu cầu trước khi gọi:
        1) Cách chuẩn:
            runtime = init_default_bm25(corpus_path='data/corpus.json', k1=1.5, b=1.0)
            bm25_retrieve("your query", top_k=100)  # dùng default runtime
        2) Hoặc truyền runtime trực tiếp:
            bm25_retrieve("your query", top_k=100, runtime=runtime)

    Output:
        list[tuple[int, float]]: [(doc_id, bm25_score), ...] dài tối đa top_k.
    """
    active_runtime = runtime
    if active_runtime is None:
        if bm25 is not None and doc_ids is not None:
            active_runtime = BM25Runtime(
                bm25=bm25,
                doc_ids=doc_ids,
                apply_stemming=False if apply_stemming is None else apply_stemming,
                use_stopwords=True if use_stopwords is None else use_stopwords,
            )
        else:
            active_runtime = _DEFAULT_RUNTIME

    if active_runtime is None:
        raise ValueError(
            "BM25 runtime chưa được khởi tạo. "
            "Hãy gọi init_default_bm25(...) trước, hoặc truyền runtime=..., "
            "hoặc truyền bm25=... và doc_ids=...."
        )

    stem_flag = active_runtime.apply_stemming if apply_stemming is None else apply_stemming
    stop_flag = active_runtime.use_stopwords if use_stopwords is None else use_stopwords

    stopword_set = build_stopword_set(download_if_missing=True) if stop_flag else None
    query_tokens = tokenize_text(
        text=query,
        stopword_set=stopword_set,
        apply_stemming=stem_flag,
    )
    scores = active_runtime.bm25.get_scores(query_tokens)
    ranked_idx = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:top_k]
    return [(active_runtime.doc_ids[i], float(scores[i])) for i in ranked_idx]


def main() -> None:
    """CLI chạy nhanh Task 2.5: build BM25 index và in thống kê.
    Thêm vào đó thực thi Task 2.8 (serialize_index) và Task 2.9 (query retrieval).

    Input (CLI args):
        --corpus-path: đường dẫn `data/corpus.json`
        --k1, --b: siêu tham số BM25
        --stemming: bật stemming
        --no-stopwords: tắt stopword removal

    Output:
        - Không ghi file ở Task 2.5.
        - In ra màn hình: số docs đã index, token stats, cấu hình index.
    """
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build BM25 index over CISI corpus")
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=project_root / "data" / "corpus.json",
    )
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=1.0)
    parser.add_argument("--stemming", action="store_true")
    parser.add_argument("--no-stopwords", action="store_true")
    args = parser.parse_args()

    corpus = load_corpus(args.corpus_path)
    bm25, doc_ids, tokenized_docs = build_bm25_index(
        corpus=corpus,
        apply_stemming=args.stemming,
        include_title=True,
        use_stopwords=not args.no_stopwords,
        k1=args.k1,
        b=args.b,
    )
    avg_doc_len = sum(len(toks) for toks in tokenized_docs) / len(tokenized_docs) if tokenized_docs else 0.0

    print(f"[OK] Built BM25 index for {len(doc_ids)} documents")
    print(f"[OK] k1={args.k1}, b={args.b}, stemming={args.stemming}, stopwords={not args.no_stopwords}")
    print(f"[OK] Avg tokenized doc length: {avg_doc_len:.2f}")
    print(f"[OK] BM25 object: {type(bm25).__name__}")


if __name__ == "__main__":
    main()
