"""Utilities for parsing CISI-format files.

Module này tập trung vào bài toán parse file dạng CISI (ví dụ: CISI.ALL, CISI.QRY)
thành dữ liệu có cấu trúc để dùng cho IR pipeline (BM25, Dense Retrieval, Re-ranking).

Mục tiêu chính:
1. Tách từng document/query bằng marker `.I <id>`.
2. Trích xuất nội dung từng thẻ bằng regex nâng cao:
   - `.T`: title
   - `.A`: author
   - `.W`: main text
3. Chịu được dữ liệu thiếu/không đồng nhất (thiếu `.A` hoặc `.W`) mà không crash.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Any


_DOC_SPLIT_PATTERN = re.compile(r"\.I\s+(\d+)")
_TAG_PATTERN = re.compile(r"\.([A-Z])\s*(.*?)(?=\n\.[A-Z]\s|$)", re.DOTALL)


def _normalize_text(value: str) -> str:
    """Chuẩn hóa text trích xuất từ tag.

    - Loại bỏ khoảng trắng thừa ở đầu/cuối.
    - Co cụm nhiều khoảng trắng/newline thành 1 space để dữ liệu đồng nhất.
    """
    return re.sub(r"\s+", " ", value).strip()


def parse_cisi_text(raw_text: str) -> list[dict[str, Any]]:
    """Parse nội dung thô CISI thành danh sách document đã chuẩn hóa.

    Tác dụng:
    - Đây là hàm lõi xử lý chuỗi CISI trong bộ nhớ.
    - Dùng khi bạn đã có sẵn nội dung file (đọc từ nơi khác, test unit, notebook).

    Input:
    - ``raw_text`` (str):
      Toàn bộ nội dung của file CISI theo format:
      `.I <id>` + các thẻ `.T`, `.A`, `.W`, ...
      Ví dụ tối thiểu:
      ```.text
      .I 1
      .T
      Title 1
      .A
      Author 1
      .W
      Body 1
      .I 2
      .T
      Title 2
      .W
      Body 2
      ```

    Output:
    - ``list[dict[str, Any]]``:
      Mỗi phần tử là một document với schema:
      - ``doc_id`` (int): ID từ `.I`
      - ``title`` (str): nội dung tag `.T` (rỗng nếu thiếu)
      - ``author`` (str): nội dung tag `.A` (rỗng nếu thiếu)
      - ``text`` (str): nội dung tag `.W` (rỗng nếu thiếu)
      - ``raw_tags`` (dict[str, str]): toàn bộ tag đã parse được để debug/mở rộng

    Cơ chế xử lý dữ liệu thiếu:
    - Dùng ``defaultdict(dict)`` làm vùng đệm tài liệu.
    - Dùng ``try-except KeyError`` khi đọc tag quan trọng (`.T`, `.A`, `.W`) để
      đảm bảo không lỗi nếu dữ liệu không có đủ tag.

    Lưu ý:
    - Regex tag dùng ``re.DOTALL`` để bắt nội dung nhiều dòng.
    - Các khoảng trắng/newline trong nội dung tag được chuẩn hóa về 1 space.
    """
    if not isinstance(raw_text, str):
        raise TypeError(f"raw_text must be str, got: {type(raw_text).__name__}")

    parsed_docs: defaultdict[int, dict[str, Any]] = defaultdict(dict)
    chunks = _DOC_SPLIT_PATTERN.split(raw_text)
    # Với pattern có capture group, split sẽ cho dạng:
    # [prefix, doc_id_1, body_1, doc_id_2, body_2, ...]

    for idx in range(1, len(chunks), 2):
        doc_id_str = chunks[idx]
        body = chunks[idx + 1] if (idx + 1) < len(chunks) else ""

        if not doc_id_str.isdigit():
            # Bỏ qua block bất thường nhưng không làm hỏng toàn bộ pipeline.
            continue

        doc_id = int(doc_id_str)
        tag_map = {tag: _normalize_text(content) for tag, content in _TAG_PATTERN.findall(body)}

        parsed_docs[doc_id]["doc_id"] = doc_id

        try:
            parsed_docs[doc_id]["title"] = tag_map["T"]
        except KeyError:
            parsed_docs[doc_id]["title"] = ""

        try:
            parsed_docs[doc_id]["author"] = tag_map["A"]
        except KeyError:
            parsed_docs[doc_id]["author"] = ""

        try:
            parsed_docs[doc_id]["text"] = tag_map["W"]
        except KeyError:
            parsed_docs[doc_id]["text"] = ""

        parsed_docs[doc_id]["raw_tags"] = tag_map

    return [parsed_docs[doc_id] for doc_id in sorted(parsed_docs.keys())]


def parse_cisi_file(file_path: str | Path, encoding: str = "utf-8") -> list[dict[str, Any]]:
    """Đọc file CISI từ đĩa và trả về dữ liệu đã parse.

    Tác dụng:
    - Hàm wrapper thuận tiện cho pipeline chính (chỉ cần truyền đường dẫn file).
    - Nội bộ sẽ đọc file rồi gọi ``parse_cisi_text``.

    Input:
    - ``file_path`` (str | pathlib.Path):
      Đường dẫn tới file CISI cần parse (ví dụ: `CISI.ALL`, `CISI.QRY`).
    - ``encoding`` (str, mặc định ``"utf-8"``):
      Encoding dùng để đọc file.

    Output:
    - ``list[dict[str, Any]]``: cùng schema như `parse_cisi_text`.

    Ngoại lệ:
    - ``FileNotFoundError``: file không tồn tại.
    - ``IsADirectoryError``: `file_path` trỏ tới thư mục, không phải file.
    - ``UnicodeDecodeError``: encoding không phù hợp.
    """
    path = Path(file_path)
    raw_text = path.read_text(encoding=encoding)
    return parse_cisi_text(raw_text)


def parse_cisi_rel_file(file_path: str | Path, encoding: str = "utf-8") -> dict[int, list[int]]:
    """Parse file CISI.REL thành qrels dạng Dict[int, List[int]].

    Tác dụng:
    - Đọc ground-truth relevance từ file `CISI.REL`.
    - Trả về mapping: query_id -> danh sách doc_id liên quan.

    Input:
    - ``file_path`` (str | Path): đường dẫn tới file CISI.REL.
    - ``encoding`` (str): encoding đọc file, mặc định ``utf-8``.

    Format CISI.REL (mỗi dòng):
    - Cột 1: query_id
    - Cột 2: doc_id
    - Cột 3, 4: metadata (không dùng cho qrels cơ bản)
    """
    path = Path(file_path)
    qrels: defaultdict[int, set[int]] = defaultdict(set)

    for raw_line in path.read_text(encoding=encoding).splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        try:
            query_id = int(parts[0])
            doc_id = int(parts[1])
        except ValueError:
            continue

        qrels[query_id].add(doc_id)

    return {qid: sorted(doc_ids) for qid, doc_ids in sorted(qrels.items())}


def export_qrels_json(
    rel_file_path: str | Path, output_path: str | Path, encoding: str = "utf-8"
) -> dict[int, list[int]]:
    """Parse `CISI.REL` và lưu qrels ra JSON.

    Output JSON có key là chuỗi (đúng chuẩn JSON object), ví dụ:
    {
      "1": [28, 35, 38],
      "2": [12, 77]
    }
    """
    qrels = parse_cisi_rel_file(rel_file_path, encoding=encoding)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(qrels, ensure_ascii=False, indent=2), encoding="utf-8")
    return qrels

