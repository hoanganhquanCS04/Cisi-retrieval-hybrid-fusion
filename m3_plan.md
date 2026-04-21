# Kế hoạch thực hiện Task của Member 3 (Neural Re-ranking & Evaluation)

Dựa trên tài liệu `TEAM_ASSIGNMENTS.md` và `TEAM_GUIDE.md`, dưới đây là kế hoạch chi tiết để hoàn thành nhiệm vụ của Member 3.

## Tình trạng hiện tại (Dependencies Check)
- **Đã có:** `corpus.json`, `queries.json`, `qrels.json` và kết quả của BM25 `bm25_top100.json` (từ M1).
- **Thiếu:** `dense_top100.json` và `hybrid_top100.json` (từ M2 chưa hoàn thành).
- **Giải pháp (Workaround):** Căn cứ theo đúng Bảng Hard Dependencies trong tài liệu dự án, do M2 chưa Handoff, M3 sẽ dùng `bm25_top100.json` trực tiếp từ M1 làm danh sách ứng viên (candidate list) để bắt đầu test Cross-Encoder và MonoT5.

---

## Chi tiết Kế Hoạch Thực Hiện

### Phase 1: Môi trường & Thiết lập Model
1. **Thiết lập thư mục & Script tải Model:** Tạo thư mục `hf_cache/` và viết script `download_models.py` dùng `snapshot_download` để tải các model `cross-encoder/ms-marco-MiniLM-L-6-v2` và `castorini/monot5-base-msmarco`.
2. **Cài đặt thư viện:** Bổ sung `requirements_m3.txt` (`sentence-transformers`, `transformers`, `torch`, `ranx`, `accelerate`).
3. **Thử nghiệm khói (Smoke Test):** Tạo script/notebook hoặc check load model cơ bản.

### Phase 2: Triển khai Cross-Encoder Re-ranking
1. **Xây dựng Reranker:** Viết module `reranking/cross_encoder_reranker.py` chứa class tự động load từ `hf_cache` offline local.
2. **Chạy Re-rank & Lưu file:** Mở file `bm25_top100.json`, nối nội dung `corpus.json` vào ứng với 112 queries, sau đó re-rank toàn bộ 100 docs / query.
3. **Report:** Xuất file kết quả `data/ce_reranked.json` và ghi nhận độ trễ (latency) vào `reports/ce_latency.md`.

### Phase 3: Triển khai MonoT5 Re-ranking
1. **Xây dựng Reranker:** Viết module `reranking/monot5_reranker.py` xây dựng prompt (`Query: ... Document: ... Relevant:`) và dùng log-prob của token " true" (dựa theo decoder).
2. **Chạy Re-rank & Lưu file:** Re-rank tương tự Phase 2 và xuất ra `data/monot5_reranked.json`.
3. **Report:** So sánh tốc độ giữa CE vs MonoT5 trong `reports/reranker_latency_comparison.md`.

### Phase 4: Đánh giá Toàn Diện & Báo cáo với RANX
1. **Eval Pipeline:** Viết `evaluation/eval_pipeline.py` sử dụng thư viện `ranx`. Convert kết quả BM25, CE-Reranked, và MonoT5-Reranked sang `Run` và đánh giá với metrics: `mrr` và `hit_rate@10` / `precision@10`.
2. **Hình ảnh & Báo cáo:** Xuất json báo cáo (`reports/final_metrics_table.json`), log phân tích lỗi, phân tích rank shift (`reports/rank_shift_analysis.md`) và final report.

---

## Hướng dẫn chạy Test (Workaround Pipeline)

Để test pipeline của Member 3 ngay cả khi Member 2 chưa hoàn thiện, bạn có thể thực hiện theo các bước sau trong terminal từ thư mục gốc của project:

```bash
# 1. Tải hai mô hình (Cross-Encoder và MonoT5) về thư mục hf_cache/ để chạy offline local
uv run python models/download_models.py

# 2. Re-ranking: Chạy evaluation trên data của Member 1 (dùng tạm top 100 từ BM25)
uv run python reranking/run_reranking.py

# 3. Đánh giá (Evaluation): Tổng hợp so sánh MRR và P@10 bằng thư viện RANX
uv run python evaluation/eval_pipeline.py
```
*Kết quả cuối cùng sẽ ghi nhận vào thư mục `reports/final_metrics_table.json` và xuất file `data/ce_reranked.json`, `data/monot5_reranked.json`.*
