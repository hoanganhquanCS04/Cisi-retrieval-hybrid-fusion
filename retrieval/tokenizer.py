from __future__ import annotations

import re
from typing import Any

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def build_stopword_set(
    language: str = "english",
    extra_stopwords: set[str] | None = None,
    download_if_missing: bool = False,
) -> set[str]:
    """Tạo stopword set dùng cho pipeline tokenization.

    Chức năng:
        - Lấy stopwords từ `nltk.corpus.stopwords`.
        - Cho phép thêm stopwords tùy chỉnh của project.

    Input:
        language: Ngôn ngữ stopwords trong NLTK (mặc định "english").
        extra_stopwords: Tập stopword bổ sung do người dùng tự định nghĩa.
        download_if_missing: Nếu True thì tự tải resource `stopwords` khi chưa có.

    Yêu cầu dữ liệu đầu vào:
        - language phải là ngôn ngữ có trong NLTK stopwords corpus.
        - extra_stopwords (nếu có) là `set[str]`.

    Output:
        set[str]: Tập stopword đã hợp nhất (NLTK + extra_stopwords).
    """
    try:
        base = set(stopwords.words(language))
    except LookupError:
        if not download_if_missing:
            raise LookupError(
                "NLTK stopwords resource not found. "
                "Run: nltk.download('stopwords') or call with download_if_missing=True."
            )
        nltk.download("stopwords", quiet=True)
        base = set(stopwords.words(language))

    if extra_stopwords:
        base = base.union({w.strip().lower() for w in extra_stopwords if w.strip()})
    return base


def normalize_text(text: str) -> str:
    """Chuẩn hóa text trước khi tách token.

    Chức năng:
        - lowercase toàn bộ chuỗi.
        - thay nhiều khoảng trắng/newline liên tiếp thành 1 khoảng trắng.
        - trim khoảng trắng đầu/cuối.

    Input:
        text: Chuỗi đầu vào cần chuẩn hóa.

    Output:
        str: Chuỗi đã chuẩn hóa.
    """
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def tokenize_text(
    text: str,
    stopword_set: set[str] | None = None,
    apply_stemming: bool = False,
    stemmer: PorterStemmer | None = None,
) -> list[str]:
    """Tokenize 1 chuỗi theo chuẩn BM25 pipeline (lowercase + stopword removal + stemming tùy chọn).

    Chức năng:
        - chuẩn hóa text.
        - tách token chữ/số bằng regex `[a-z0-9]+`.
        - loại stopwords (nếu cung cấp stopword_set).
        - stemming (nếu apply_stemming=True).

    Input:
        text: Chuỗi văn bản cần tokenize.
        stopword_set: Tập stopwords để loại bỏ; có thể truyền None để giữ nguyên.
        apply_stemming: Bật/tắt stemming.
        stemmer: Stemmer tùy chọn. Nếu None và apply_stemming=True thì dùng PorterStemmer mặc định.

    Output:
        list[str]: Danh sách token sau tiền xử lý.
    """
    normalized = normalize_text(text)
    tokens = re.findall(r"[a-z0-9]+", normalized)

    if stopword_set is not None:
        tokens = [tok for tok in tokens if tok not in stopword_set]

    if apply_stemming:
        stem = stemmer or PorterStemmer()
        tokens = [stem.stem(tok) for tok in tokens]

    return tokens


def tokenize_corpus(
    corpus: list[dict[str, Any]],
    stopword_set: set[str] | None = None,
    apply_stemming: bool = False,
    include_title: bool = True,
) -> tuple[list[int], list[list[str]]]:
    """Tokenize toàn bộ corpus để làm đầu vào cho BM25.

    Chức năng:
        - nhận `corpus.json` đã parse.
        - với mỗi document: ghép `title` + `text` (nếu include_title=True) rồi tokenize.
        - trả về tokenized_docs theo đúng thứ tự doc_ids.

    Input:
        corpus: Danh sách document dict. Mỗi phần tử tối thiểu cần:
            - doc_id (int)
            - title (str, có thể rỗng)
            - text (str, có thể rỗng)
        stopword_set: Tập stopwords dùng để loại bỏ token.
        apply_stemming: Bật/tắt stemming.
        include_title: Nếu True thì ghép title vào trước text.

    Yêu cầu dữ liệu đầu vào:
        - `corpus` phải có doc_id hợp lệ và giữ thứ tự ổn định nếu muốn ánh xạ index->doc_id.

    Output:
        tuple[list[int], list[list[str]]]:
            - doc_ids: danh sách doc_id theo thứ tự token hóa.
            - tokenized_docs: danh sách token list tương ứng từng doc_id.
    """
    doc_ids: list[int] = []
    tokenized_docs: list[list[str]] = []

    for doc in corpus:
        doc_id = int(doc["doc_id"])
        title = str(doc.get("title", "")).strip()
        text = str(doc.get("text", "")).strip()
        source_text = f"{title}. {text}" if include_title and title and text else (title or text)
        tokens = tokenize_text(
            text=source_text,
            stopword_set=stopword_set,
            apply_stemming=apply_stemming,
        )
        doc_ids.append(doc_id)
        tokenized_docs.append(tokens)

    return doc_ids, tokenized_docs


def tokenize_queries(
    queries: list[dict[str, Any]],
    stopword_set: set[str] | None = None,
    apply_stemming: bool = False,
) -> dict[int, list[str]]:
    """Tokenize toàn bộ queries theo cùng pipeline với corpus.

    Chức năng:
        - chuẩn hóa và token hóa tất cả query trong `queries.json`.
        - đảm bảo query và document dùng cùng logic preprocessing.

    Input:
        queries: Danh sách query dict, mỗi phần tử tối thiểu có:
            - query_id (int)
            - text (str)
        stopword_set: Tập stopwords dùng chung với corpus.
        apply_stemming: Bật/tắt stemming.

    Output:
        dict[int, list[str]]:
            Mapping `query_id -> token_list`.
    """
    tokenized: dict[int, list[str]] = {}
    for q in queries:
        query_id = int(q["query_id"])
        query_text = str(q.get("text", ""))
        tokenized[query_id] = tokenize_text(
            text=query_text,
            stopword_set=stopword_set,
            apply_stemming=apply_stemming,
        )
    return tokenized
