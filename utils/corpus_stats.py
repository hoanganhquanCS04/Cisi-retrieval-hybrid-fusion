from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median


def _token_count(text: str) -> int:
    """Đếm số token đơn giản bằng cách split theo khoảng trắng.

    Args:
        text: Chuỗi đầu vào (thường là trường `text` của 1 document).

    Returns:
        Số token (int).
    """
    return len(text.split())


def _percentile(sorted_values: list[int], p: float) -> float:
    """Tính percentile theo nội suy tuyến tính.

    Args:
        sorted_values: Danh sách số đã sắp xếp tăng dần.
        p: Percentile cần tính trong khoảng [0, 100].

    Returns:
        Giá trị percentile dạng float. Trả 0.0 nếu danh sách rỗng.
    """
    if not sorted_values:
        return 0.0
    if p <= 0:
        return float(sorted_values[0])
    if p >= 100:
        return float(sorted_values[-1])
    k = (len(sorted_values) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return float(sorted_values[f])
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def build_corpus_stats(corpus: list[dict]) -> dict:
    """Tính thống kê tổng quan cho corpus CISI.

    Input:
        corpus: List[dict], mỗi phần tử nên có các key:
            - doc_id (int)
            - title (str)
            - author (str)
            - text (str)

    Output:
        dict gồm:
            - num_docs: tổng số văn bản
            - missing_tags: số văn bản thiếu title/author/text
            - text_length_chars: min/max/mean/median/p90/p95 và histogram bins theo độ dài ký tự của `text`
            - token_count: avg/median/p90/p95 theo token count (`text.split()`)
    """
    lengths = [len(str(d.get("text", ""))) for d in corpus]
    token_counts = [_token_count(str(d.get("text", ""))) for d in corpus]
    sorted_lengths = sorted(lengths)
    sorted_tokens = sorted(token_counts)

    missing_title = sum(1 for d in corpus if not str(d.get("title", "")).strip())
    missing_author = sum(1 for d in corpus if not str(d.get("author", "")).strip())
    missing_text = sum(1 for d in corpus if not str(d.get("text", "")).strip())

    bins = {
        "0-199": 0,
        "200-499": 0,
        "500-999": 0,
        "1000-1999": 0,
        "2000+": 0,
    }
    for l in lengths:
        if l <= 199:
            bins["0-199"] += 1
        elif l <= 499:
            bins["200-499"] += 1
        elif l <= 999:
            bins["500-999"] += 1
        elif l <= 1999:
            bins["1000-1999"] += 1
        else:
            bins["2000+"] += 1

    return {
        "num_docs": len(corpus),
        "missing_tags": {
            "title": missing_title,
            "author": missing_author,
            "text": missing_text,
        },
        "text_length_chars": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "mean": round(mean(lengths), 2) if lengths else 0,
            "median": round(median(lengths), 2) if lengths else 0,
            "p90": round(_percentile(sorted_lengths, 90), 2),
            "p95": round(_percentile(sorted_lengths, 95), 2),
            "distribution_bins": bins,
        },
        "token_count": {
            "avg_tokens": round(mean(token_counts), 2) if token_counts else 0,
            "median_tokens": round(median(token_counts), 2) if token_counts else 0,
            "p90_tokens": round(_percentile(sorted_tokens, 90), 2),
            "p95_tokens": round(_percentile(sorted_tokens, 95), 2),
        },
    }


def _format_report_md(stats: dict) -> str:
    """Chuyển dict thống kê thành nội dung Markdown report.

    Args:
        stats: Dict thống kê theo schema trả về từ `build_corpus_stats`.

    Returns:
        Chuỗi markdown hoàn chỉnh để ghi ra `corpus_stats.md`.
    """
    bins = stats["text_length_chars"]["distribution_bins"]
    return (
        "# Corpus Statistics\n\n"
        f"- Total documents: **{stats['num_docs']}**\n"
        f"- Missing `.T` (title): **{stats['missing_tags']['title']}**\n"
        f"- Missing `.A` (author): **{stats['missing_tags']['author']}**\n"
        f"- Missing `.W` (text): **{stats['missing_tags']['text']}**\n\n"
        "## `.W` length (characters)\n\n"
        f"- min: **{stats['text_length_chars']['min']}**\n"
        f"- max: **{stats['text_length_chars']['max']}**\n"
        f"- mean: **{stats['text_length_chars']['mean']}**\n"
        f"- median: **{stats['text_length_chars']['median']}**\n"
        f"- p90: **{stats['text_length_chars']['p90']}**\n"
        f"- p95: **{stats['text_length_chars']['p95']}**\n\n"
        "### Distribution bins\n\n"
        f"- 0-199: **{bins['0-199']}**\n"
        f"- 200-499: **{bins['200-499']}**\n"
        f"- 500-999: **{bins['500-999']}**\n"
        f"- 1000-1999: **{bins['1000-1999']}**\n"
        f"- 2000+: **{bins['2000+']}**\n\n"
        "## Token count (`text.split()`)\n\n"
        f"- average: **{stats['token_count']['avg_tokens']}**\n"
        f"- median: **{stats['token_count']['median_tokens']}**\n"
        f"- p90: **{stats['token_count']['p90_tokens']}**\n"
        f"- p95: **{stats['token_count']['p95_tokens']}**\n"
    )


def main() -> None:
    """CLI entrypoint để chạy thống kê corpus từ terminal.

    Đầu vào qua tham số dòng lệnh:
        --corpus-path: đường dẫn file corpus.json
        --report-path: nơi lưu báo cáo markdown
        --json-path: nơi lưu thống kê JSON

    Đầu ra:
        1) File markdown thống kê (`corpus_stats.md`)
        2) File JSON thống kê (`corpus_stats.json`)
        3) Log tóm tắt in ra terminal
    """
    project_root = Path(__file__).resolve().parents[1]
    default_corpus = project_root / "data" / "corpus.json"
    default_report = project_root / "reports" / "corpus_stats.md"
    default_json = project_root / "reports" / "corpus_stats.json"

    parser = argparse.ArgumentParser(description="Compute corpus statistics for CISI corpus.json")
    parser.add_argument("--corpus-path", type=Path, default=default_corpus, help="Path to corpus.json")
    parser.add_argument("--report-path", type=Path, default=default_report, help="Output markdown report path")
    parser.add_argument("--json-path", type=Path, default=default_json, help="Output JSON stats path")
    args = parser.parse_args()

    corpus = json.loads(args.corpus_path.read_text(encoding="utf-8"))
    stats = build_corpus_stats(corpus)

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(_format_report_md(stats), encoding="utf-8")
    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Loaded corpus: {args.corpus_path}")
    print(f"[OK] Documents: {stats['num_docs']}")
    print(f"[OK] Missing tags: {stats['missing_tags']}")
    print(f"[OK] Saved report: {args.report_path}")
    print(f"[OK] Saved json: {args.json_path}")


if __name__ == "__main__":
    main()
