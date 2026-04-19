# TEAM ASSIGNMENTS — CISI Information Retrieval Pipeline
## Enterprise-Grade Search Funnel: Hybrid BM25 + Dense Retrieval + Neural Re-ranking
### Môi trường thực thi: Kaggle Notebooks (GPU T4 / P100) — Local Model Inference

---

## Tổng quan phân công

| | Member 1 | Member 2 | Member 3 |
|---|---|---|---|
| **Role** | Data Engineer & BM25 Architect | Dense Retrieval & Hybrid Fusion Engineer | Neural Re-ranking & Evaluation Engineer |
| **Layer** | Data Ingestion → First-Stage Sparse Retrieval | Dense Embedding → Hybrid RRF Pipeline | Cross-Encoder / MonoT5 Re-ranking → Metrics |
| **Primary** | Regex Parser + BM25Okapi/bm25s + TF-IDF Baseline | SentenceTransformer Embedding + FAISS + RRF Fusion | Cross-Encoder (MiniLM) + MonoT5 + MRR/P@10 Evaluation |
| **Kaggle Notebook** | `01_data_bm25.ipynb` | `02_dense_hybrid.ipynb` | `03_reranking_eval.ipynb` |

> **Lưu ý môi trường Kaggle:**
> - Tất cả model phải chạy **offline / local** trong Kaggle Session (không gọi API ngoài).
> - Download model weights qua **Kaggle Datasets** hoặc `huggingface_hub` snapshot trước khi dùng.
> - GPU T4 (15GB VRAM) đủ cho MiniLM-L6 và MonoT5-220M. Nếu dùng MonoT5-3B cần bật P100 hoặc dùng `bfloat16`.
> - Lưu toàn bộ intermediate output (`top100_candidates.json`, checkpoint BM25) vào `/kaggle/working/` để notebook sau đọc lại.

---

## Chi tiết theo thành viên

---

### 👤 Member 1 — Data Engineer & BM25 Architect

**Trách nhiệm chính:** Xây dựng toàn bộ pipeline đọc và làm sạch tập dữ liệu CISI, triển khai TF-IDF baseline và BM25 làm First-Stage Sparse Retrieval. Output của M1 là nền tảng mà M2 và M3 đều phụ thuộc vào.

#### Nhiệm vụ theo Phase

**Phase 1: Phân tích cú pháp & Tiền xử lý dữ liệu CISI**

| Task | Mô tả | Output (trong `/kaggle/working/`) |
|---|---|---|
| 1.1 | Tải 3 file CISI (`CISI.ALL`, `CISI.QRY`, `CISI.REL`) lên Kaggle Dataset private | Dataset `cisi-ir-dataset` |
| 1.2 | Viết hàm `parse_cisi_file()` dùng `re.split(r'\.I\s+\d+')` để tách tài liệu | `utils/parser.py` |
| 1.3 | Triển khai Regex nâng cao `r'\.([A-Z])\s*(.*?)(?=\n\.[A-Z]\|$)'` với cờ `re.DOTALL` để trích xuất thẻ `.T`, `.A`, `.W` | `utils/parser.py` |
| 1.4 | Xử lý bất đồng nhất dữ liệu: dùng `defaultdict(dict)` + `try-except` cho các doc thiếu thẻ `.A` hoặc `.W` | `utils/parser.py` |
| 1.5 | Parse `CISI.REL` → `Dict[int, List[int]]` làm Ground Truth (qrels) | `data/qrels.json` |
| 1.6 | Export corpus thành `List[Dict]` với các keys: `doc_id`, `title`, `author`, `text` | `data/corpus.json` |
| 1.7 | Export queries thành `List[Dict]` với keys: `query_id`, `text` | `data/queries.json` |
| 1.8 | Thống kê corpus: phân bố độ dài `.W`, số doc thiếu thẻ, avg token count | `reports/corpus_stats.md` |

**Phase 2: Xây dựng TF-IDF Baseline & BM25 Retrieval**

| Task | Mô tả | Output |
|---|---|---|
| 2.1 | Triển khai TF-IDF baseline bằng `sklearn.TfidfVectorizer` + cosine similarity | `retrieval/tfidf_baseline.py` |
| 2.2 | Đánh giá TF-IDF: tính MRR và P@10 trên 112 queries → làm mốc so sánh | `reports/baseline_metrics.json` |
| 2.3 | Cài `rank-bm25` trong Kaggle (`!pip install rank-bm25`) | — |
| 2.4 | Viết pipeline tokenization: lowercase + stopword removal (`nltk.stopwords`) + giữ stemming tùy chọn | `retrieval/tokenizer.py` |
| 2.5 | Khởi tạo `BM25Okapi` index trên toàn bộ 1.460 tài liệu | `retrieval/bm25_retriever.py` |
| 2.6 | Tune siêu tham số: grid search `k1` ∈ {1.2, 1.5, 2.0} × `b` ∈ {0.5, 0.75, 1.0} trên 30 queries dev | `notebooks/bm25_tuning.ipynb` |
| 2.7 | Hàm `bm25_retrieve(query, top_k=100)` → trả về `List[(doc_id, bm25_score)]` | `retrieval/bm25_retriever.py` |
| 2.8 | Serialize BM25 index ra file `bm25_index.pkl` (dùng `pickle`) để M2 tải lại | `data/bm25_index.pkl` |
| 2.9 | Chạy BM25 trên toàn bộ 112 queries, lưu kết quả top-100 mỗi query | `data/bm25_top100.json` |

**Phase 3: Tích hợp & Bàn giao**

| Task | Mô tả | Output |
|---|---|---|
| 3.1 | Viết `retrieval/base_retriever.py`: abstract class `BaseRetriever` với method `retrieve(query, top_k)` | `retrieval/base_retriever.py` |
| 3.2 | Hỗ trợ M2 debug lỗi tokenization nếu BM25 và Dense Retrieval dùng khác nhau | Code review |
| 3.3 | Hỗ trợ M3 chuẩn bị input format `(query_text, doc_text)` cho Cross-Encoder | Schema doc |
| 3.4 | Kiểm tra lại `bm25_top100.json` — đảm bảo không có `doc_id` nằm ngoài phạm vi corpus | `tests/test_bm25_output.py` |

**Phase 4: Báo cáo & Hoàn thiện**

| Task | Mô tả | Output |
|---|---|---|
| 4.1 | Phân tích lỗi BM25: liệt kê 10 queries có MRR thấp nhất, giải thích nguyên nhân (vocabulary mismatch) | `reports/bm25_error_analysis.md` |
| 4.2 | Viết phần báo cáo: Data Ingestion, Regex Parsing Strategy, TF-IDF vs BM25 Analysis | Báo cáo cuối kỳ |

**Deliverables:**
- [ ] Parser hoạt động đúng: 1.460 docs + 112 queries được trích xuất đầy đủ
- [ ] `corpus.json`, `queries.json`, `qrels.json` — schema chuẩn cho cả nhóm dùng
- [ ] TF-IDF baseline với MRR và P@10 đã tính
- [ ] BM25 index (`bm25_index.pkl`) + kết quả top-100 (`bm25_top100.json`)
- [ ] Unit tests cho parser và BM25 output

---

### 👤 Member 2 — Dense Retrieval & Hybrid RRF Fusion Engineer

**Trách nhiệm chính:** Triển khai Dense Retrieval bằng Bi-encoder (chạy local trên Kaggle GPU), xây dựng FAISS index, và hợp nhất BM25 + Dense qua thuật toán RRF để tạo ra danh sách top-100 ứng viên chất lượng cao nhất.

> **Kaggle-specific:** Tải model `all-MiniLM-L6-v2` từ Kaggle Models hoặc dùng `snapshot_download()` để cache vào `/kaggle/working/hf_cache/` trước khi encode.

#### Nhiệm vụ theo Phase

**Phase 1: Nghiên cứu & Chuẩn bị môi trường Kaggle**

| Task | Mô tả | Output |
|---|---|---|
| 1.1 | Cài dependencies: `!pip install sentence-transformers faiss-cpu` trong Kaggle Notebook | `requirements_kaggle.txt` |
| 1.2 | Tải model `sentence-transformers/all-MiniLM-L6-v2` về local Kaggle session: `snapshot_download(repo_id=..., local_dir='/kaggle/working/hf_cache/minilm')` | `hf_cache/minilm/` |
| 1.3 | Thử nghiệm encode tốc độ: đo thời gian encode 1.460 docs trên T4 GPU (batch_size=64) | `reports/encode_benchmark.md` |
| 1.4 | Đọc `corpus.json` và `bm25_top100.json` từ M1 | — |

**Phase 2: Xây dựng Dense Retrieval với FAISS**

| Task | Mô tả | Output |
|---|---|---|
| 2.1 | Encode toàn bộ 1.460 docs thành dense vectors (dim=384) bằng `all-MiniLM-L6-v2` với `device='cuda'` | `data/doc_embeddings.npy` |
| 2.2 | Xây dựng FAISS `IndexFlatIP` (Inner Product / Cosine) trên toàn bộ doc embeddings | `data/faiss_index.bin` |
| 2.3 | Hàm `dense_retrieve(query, top_k=100)` → encode query → FAISS search → `List[(doc_id, cosine_score)]` | `retrieval/dense_retriever.py` |
| 2.4 | Chạy Dense Retrieval trên 112 queries, lưu kết quả top-100 | `data/dense_top100.json` |
| 2.5 | Đánh giá độc lập Dense Retrieval: tính MRR và P@10 | `reports/dense_metrics.json` |

**Phase 3: Xây dựng Hybrid Search Pipeline (BM25 + Dense + RRF)**

| Task | Mô tả | Output |
|---|---|---|
| 3.1 | Triển khai hàm `reciprocal_rank_fusion(lists, k=60)` — nhận nhiều ranked list, trả về merged list theo công thức RRF | `retrieval/rrf.py` |
| 3.2 | Pipeline Hybrid: load `bm25_top100.json` (từ M1) + `dense_top100.json` → apply RRF → output `hybrid_top100.json` | `retrieval/hybrid_pipeline.py` |
| 3.3 | Thực nghiệm weight RRF: so sánh kết quả khi α=0.5/0.5 vs α=0.7 BM25 / 0.3 Dense cho corpus kỹ thuật | `notebooks/rrf_tuning.ipynb` |
| 3.4 | Đánh giá Hybrid: tính MRR và P@10 → so sánh với BM25-only và Dense-only | `reports/hybrid_metrics.json` |
| 3.5 | Xác nhận Recall@100 của Hybrid ≥ Recall@100 của BM25 đơn lẻ (mục tiêu: tăng ≥ 10%) | `reports/recall_comparison.md` |
| 3.6 | Export `hybrid_top100.json` — input chính thức cho M3 Re-ranking | `data/hybrid_top100.json` |

**Phase 4: Báo cáo & Hoàn thiện**

| Task | Mô tả | Output |
|---|---|---|
| 4.1 | Vẽ biểu đồ so sánh: BM25 vs Dense vs Hybrid (MRR bar chart, P@10 line plot) | `reports/retrieval_comparison.png` |
| 4.2 | Phân tích case study: chọn 5 queries mà Hybrid vượt BM25 rõ rệt (vocabulary mismatch đã được giải quyết) | `reports/hybrid_case_study.md` |
| 4.3 | Viết phần báo cáo: Dense Retrieval Architecture, RRF Fusion, Hybrid Search Analysis | Báo cáo cuối kỳ |

**Deliverables:**
- [ ] `doc_embeddings.npy` (1460 × 384) — encode bởi MiniLM-L6-v2 trên Kaggle GPU
- [ ] `faiss_index.bin` — FAISS IndexFlatIP sẵn sàng query
- [ ] `dense_top100.json` — kết quả Dense Retrieval 112 queries
- [ ] `rrf.py` — hàm RRF tái sử dụng được, có unit test
- [ ] `hybrid_top100.json` — output chính thức cho M3
- [ ] Bảng so sánh metric: TF-IDF vs BM25 vs Dense vs Hybrid

---

### 👤 Member 3 — Neural Re-ranking & Evaluation Engineer

**Trách nhiệm chính:** Triển khai Cross-Encoder và MonoT5 để re-rank danh sách top-100 từ Hybrid Pipeline của M2, tính toán MRR và P@10 đầy đủ, phân tích và so sánh toàn bộ pipeline.

> **Kaggle-specific:** Tải `cross-encoder/ms-marco-MiniLM-L-6-v2` và `castorini/monot5-base-msmarco` về local. MonoT5-base (~250MB) chạy ổn trên T4 với `torch_dtype=torch.bfloat16`. Nếu muốn MonoT5-3B cần enable P100 Kaggle Accelerator.

#### Nhiệm vụ theo Phase

**Phase 1: Nghiên cứu & Setup Kaggle cho Re-ranking**

| Task | Mô tả | Output |
|---|---|---|
| 1.1 | Cài dependencies: `!pip install sentence-transformers transformers torch ranx` | `requirements_kaggle.txt` |
| 1.2 | Tải Cross-Encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` về local session (`snapshot_download`) | `hf_cache/cross_encoder_minilm/` |
| 1.3 | Tải MonoT5 `castorini/monot5-base-msmarco` về local session | `hf_cache/monot5_base/` |
| 1.4 | Đọc `hybrid_top100.json` từ M2 và `corpus.json`, `queries.json` từ M1 | — |
| 1.5 | Thử nghiệm nhanh: chạy Cross-Encoder trên 10 cặp (query, doc) để kiểm tra GPU inference hoạt động | `notebooks/ce_smoke_test.ipynb` |

**Phase 2: Triển khai Cross-Encoder Re-ranking**

| Task | Mô tả | Output |
|---|---|---|
| 2.1 | Load `CrossEncoder('hf_cache/cross_encoder_minilm', device='cuda')` từ path local | `reranking/cross_encoder_reranker.py` |
| 2.2 | Xây dựng hàm `build_pairs(query_text, top100_docs)` → `List[Tuple[str, str]]` — nối query + doc text | `reranking/cross_encoder_reranker.py` |
| 2.3 | Hàm `cross_encoder_rerank(query_text, top100_docs, batch_size=32)` → chấm điểm logit + argsort giảm dần | `reranking/cross_encoder_reranker.py` |
| 2.4 | Chạy Cross-Encoder re-rank trên toàn bộ 112 queries × 100 docs/query | `data/ce_reranked.json` |
| 2.5 | Đo thời gian xử lý trung bình mỗi query trên T4 GPU (mục tiêu: < 100ms/query) | `reports/ce_latency.md` |

**Phase 3: Triển khai MonoT5 Re-ranking**

| Task | Mô tả | Output |
|---|---|---|
| 3.1 | Load MonoT5 từ path local với `T5ForConditionalGeneration.from_pretrained('hf_cache/monot5_base')` và `AutoTokenizer` | `reranking/monot5_reranker.py` |
| 3.2 | Xây dựng prompt template: `f"Query: {query} Document: {doc_text} Relevant:"` cho từng cặp | `reranking/monot5_reranker.py` |
| 3.3 | Hàm `monot5_score(query, doc_text)` → tính `log_prob("true")` bằng cách lấy logits tại token "▁true" trong vocabulary của T5 | `reranking/monot5_reranker.py` |
| 3.4 | Chạy MonoT5 re-rank trên 112 queries × 100 docs — lưu kết quả | `data/monot5_reranked.json` |
| 3.5 | So sánh latency MonoT5 vs Cross-Encoder trên cùng T4 GPU | `reports/reranker_latency_comparison.md` |

**Phase 4: Đánh giá toàn diện & Báo cáo**

| Task | Mô tả | Output |
|---|---|---|
| 4.1 | Cài `ranx` (`!pip install ranx`), khởi tạo `Qrels` từ `qrels.json` và `Run` từ mỗi pipeline | `evaluation/eval_pipeline.py` |
| 4.2 | Tính **MRR** và **P@10** cho tất cả 5 hệ thống: TF-IDF, BM25, Dense, Hybrid, CE-Reranked, MonoT5-Reranked | `reports/final_metrics_table.json` |
| 4.3 | Tạo bảng so sánh tổng hợp 6 hệ thống (MRR ↑, P@10 ↑) và vẽ bar chart | `reports/full_pipeline_comparison.png` |
| 4.4 | Phân tích "dịch chuyển thứ hạng": với 10 queries đại diện, trực quan hóa vị trí của correct doc trước và sau re-ranking | `reports/rank_shift_analysis.md` |
| 4.5 | Phân tích lỗi Re-ranking: 5 queries mà MonoT5 vẫn xếp hạng sai, giải thích nguyên nhân | `reports/reranker_error_cases.md` |
| 4.6 | Kết luận cuối: đề xuất kiến trúc tối ưu (Hybrid BM25+Dense + MonoT5) và giải thích trade-off latency vs accuracy | `reports/architecture_recommendation.md` |
| 4.7 | Viết phần báo cáo: Neural Re-ranking, Cross-Encoder vs MonoT5, Evaluation Results, Conclusion | Báo cáo cuối kỳ |

**Deliverables:**
- [ ] `cross_encoder_reranker.py` — inference local, không cần internet
- [ ] `monot5_reranker.py` — log-prob scoring từ T5 decoder
- [ ] `ce_reranked.json` + `monot5_reranked.json`
- [ ] `eval_pipeline.py` — tính MRR + P@10 bằng `ranx` cho mọi pipeline
- [ ] Bảng so sánh 6 hệ thống với MRR và P@10 đầy đủ
- [ ] Báo cáo phân tích rank shift (minh chứng toán học cho giá trị của Re-ranking)

---

## Ma Trận Phụ Thuộc Công Việc (Dependency Matrix)

### Biểu đồ luồng phụ thuộc tổng thể

```
M1 (Phase 1) ──► corpus.json / queries.json / qrels.json
     │                    │                        │
     ▼                    ▼                        ▼
M1 (Phase 2) ──► bm25_top100.json ──────────────► M2 (Phase 3)
                                                    │
M2 (Phase 2) ──► dense_top100.json ────────────► M2 (Phase 3)
                                                    │
                                                    ▼
                                          hybrid_top100.json ──► M3 (Phase 2, 3)
                                                                    │
M1 ──► qrels.json ──────────────────────────────────────────────► M3 (Phase 4)
```

---

### Bảng Hard Dependencies (Phải có mới làm được)

| Ai cần | Cần từ ai | Artifact cụ thể | Deadline cần có | Workaround nếu chưa xong |
|---|---|---|---|---|
| **M2** | **M1** | `corpus.json` (1460 docs với field `text`) | Cuối Phase 1 | M2 dùng 200 docs mẫu từ CISI raw tự parse tạm để test encode pipeline |
| **M2** | **M1** | `bm25_top100.json` (112 queries × top-100 doc_id + score) | Cuối Phase 2 của M1 | M2 tự chạy BM25 tạm bằng `rank-bm25` với tokenization đơn giản để có dữ liệu test RRF |
| **M3** | **M1** | `corpus.json` (để tra nội dung doc theo doc_id) | Cuối Phase 1 | M3 giữ map `doc_id → text` trong memory từ CISI.ALL raw |
| **M3** | **M1** | `qrels.json` (ground truth để tính MRR, P@10) | Cuối Phase 1 | Không có workaround — đây là blocker cứng. M1 ưu tiên task 1.5 trước |
| **M3** | **M2** | `hybrid_top100.json` (danh sách ứng viên sau RRF) | Cuối Phase 3 của M2 | M3 dùng `bm25_top100.json` trực tiếp từ M1 để bắt đầu test Cross-Encoder. Kết quả sẽ thấp hơn nhưng pipeline chạy được |
| **M3** | **M2** | `dense_top100.json` (để so sánh Dense-only baseline) | Cuối Phase 2 của M2 | M3 bỏ qua cột Dense-only trong bảng so sánh, điền sau |

---

### Bảng Soft Dependencies (Nên có, không có vẫn làm được)

| Ai cần | Cần từ ai | Mô tả | Workaround |
|---|---|---|---|
| M2 | M1 | Biết `avg_doc_length` để chọn `batch_size` encode hợp lý | M2 dùng `batch_size=64` mặc định cho MiniLM |
| M3 | M1 | Biết số lượng queries có ≥ 1 relevant doc trong qrels (có thể < 112) | M3 filter trước khi tính MRR, bỏ qua queries không có qrel |
| M3 | M2 | Biết thời gian encode FAISS để ước tính tổng pipeline latency | M3 ghi thời gian inference riêng của Re-ranking, tổng hợp sau |
| M2 | M3 | Feedback MRR của Hybrid so với BM25 để quyết định có cần tune RRF weight không | M2 tune dựa trên Recall@100 thay vì đợi MRR từ M3 |

---

### Bảng Phụ Thuộc Theo Thứ Tự Thời Gian

```
TUẦN 1-2 (Phase 1 của M1 phải xong trước)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
M1: Parse CISI → corpus.json + queries.json + qrels.json
     ↓ (unblock cho cả M2 và M3)
M2: Bắt đầu setup Kaggle, test encode nhỏ
M3: Bắt đầu setup Kaggle, test load Cross-Encoder

TUẦN 3-4 (Phase 2 song song)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
M1: Build BM25 index + export bm25_top100.json
     ↓ (unblock M2 Phase 3)
M2: Dense encode + FAISS (song song với M1 Phase 2)
M3: Chuẩn bị Cross-Encoder + MonoT5, test với bm25_top100.json tạm

TUẦN 5-6 (Phase 3 — M2 xong trước)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
M2: RRF Fusion → export hybrid_top100.json
     ↓ (unblock M3 Phase 2, 3 chính thức)
M3: Re-rank CE + MonoT5 trên hybrid_top100.json chính thức

TUẦN 7-8 (Phase 4 — tất cả song song)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
M1 + M2 + M3: Báo cáo từng phần → M3 tổng hợp bảng so sánh cuối
```

---

### Thứ tự khởi động bắt buộc (Critical Path)

```
1. M1 Task 1.2 → 1.3 → 1.4 → 1.5 (parser + qrels)  ← PHẢI XONG TRONG TUẦN 1
2. M1 Task 1.6 → 1.7 (corpus.json + queries.json)   ← PHẢI XONG TRƯỚC KHI M2 encode
3. M1 Task 2.9 (bm25_top100.json)                   ← PHẢI XONG TRƯỚC KHI M2 chạy RRF
4. M2 Task 3.6 (hybrid_top100.json)                 ← PHẢI XONG TRƯỚC KHI M3 re-rank chính thức
5. M3 Task 4.2 (final metrics table)                ← Cần output của M1 + M2 + M3 đều xong
```

---

## Giao Thức Handoff (Bàn giao giữa các thành viên)

Mỗi khi hoàn thành artifact để người khác dùng, thực hiện đủ 3 bước:

```
Bước 1: Upload file lên Kaggle Dataset chung của nhóm hoặc share Notebook output
Bước 2: Ghi vào HANDOFF_LOG.md:
         - Tên file và đường dẫn Kaggle
         - Schema / format (ví dụ: JSON với keys gì, dtype của array npy)
         - Ví dụ: {"query_id": 1, "results": [{"doc_id": 23, "score": 14.7}, ...]}
Bước 3: Ping thành viên nhận trên nhóm chat, gửi kèm đoạn code đọc file đó
```

**Các mốc handoff cụ thể:**

| Phase | Từ | Đến | Artifact | Format |
|---|---|---|---|---|
| Cuối Phase 1 | M1 | M2, M3 | `corpus.json` | `List[{doc_id: int, title: str, author: str, text: str}]` |
| Cuối Phase 1 | M1 | M3 | `qrels.json` | `Dict[str, List[int]]` → `{"1": [12, 45, 78], ...}` |
| Cuối Phase 1 | M1 | M2, M3 | `queries.json` | `List[{query_id: int, text: str}]` |
| Cuối Phase 2 M1 | M1 | M2 | `bm25_top100.json` | `Dict[str, List[{doc_id, score}]]` |
| Cuối Phase 2 M1 | M1 | M3 (tạm) | `bm25_top100.json` | Dùng tạm để test CE pipeline |
| Cuối Phase 2 M2 | M2 | M3 | `dense_top100.json` | `Dict[str, List[{doc_id, score}]]` |
| Cuối Phase 3 M2 | M2 | M3 | `hybrid_top100.json` | `Dict[str, List[{doc_id, rrf_score}]]` — input chính thức |
| Cuối Phase 3 M3 | M3 | Báo cáo | `final_metrics_table.json` | `{system: str, MRR: float, P@10: float}` |

---

## Schema Dữ liệu Chung (Data Contract giữa các thành viên)

Mọi file JSON trao đổi giữa thành viên phải tuân theo schema sau:

```python
# corpus.json
[
  {
    "doc_id": 1,            # int, 1-indexed theo CISI
    "title": "...",         # str, nội dung thẻ .T
    "author": "...",        # str, nội dung thẻ .A (rỗng nếu không có)
    "text": "..."           # str, nội dung thẻ .W — dùng để encode và score
  },
  ...
]

# queries.json
[
  {
    "query_id": 1,          # int
    "text": "..."           # str, nội dung thẻ .W trong CISI.QRY
  },
  ...
]

# qrels.json
{
  "1": [12, 45, 78],        # key là query_id dạng str, value là list doc_id relevant
  ...
}

# bm25_top100.json / dense_top100.json / hybrid_top100.json / ce_reranked.json
{
  "1": [                    # key là query_id dạng str
    {"doc_id": 23, "score": 14.72},
    {"doc_id": 5,  "score": 12.31},
    ...                     # đúng 100 entries, sorted giảm dần theo score
  ],
  ...
}
```

---

## Cấu trúc thư mục dự án (Kaggle Notebook Layout)

```
CISI-IR-Pipeline/
├── notebooks/
│   ├── 01_data_bm25.ipynb          # M1: Parser + TF-IDF + BM25
│   ├── 02_dense_hybrid.ipynb       # M2: Dense Encode + FAISS + RRF
│   ├── 03_reranking_eval.ipynb     # M3: CE + MonoT5 + Evaluation
│   ├── bm25_tuning.ipynb           # M1: Hyperparameter grid search
│   └── rrf_tuning.ipynb            # M2: RRF weight experiments
├── retrieval/
│   ├── base_retriever.py           # M1: Abstract class
│   ├── tokenizer.py                # M1: NLP preprocessing
│   ├── tfidf_baseline.py           # M1: sklearn TF-IDF
│   ├── bm25_retriever.py           # M1: BM25Okapi wrapper
│   ├── dense_retriever.py          # M2: SentenceTransformer + FAISS
│   ├── rrf.py                      # M2: Reciprocal Rank Fusion
│   └── hybrid_pipeline.py          # M2: BM25 + Dense → RRF
├── reranking/
│   ├── cross_encoder_reranker.py   # M3: MiniLM Cross-Encoder
│   └── monot5_reranker.py          # M3: MonoT5 log-prob scoring
├── evaluation/
│   └── eval_pipeline.py            # M3: ranx MRR + P@10
├── utils/
│   └── parser.py                   # M1: CISI Regex parser
├── data/                           # Kaggle /kaggle/working/data/
│   ├── corpus.json
│   ├── queries.json
│   ├── qrels.json
│   ├── bm25_index.pkl
│   ├── bm25_top100.json
│   ├── doc_embeddings.npy
│   ├── faiss_index.bin
│   ├── dense_top100.json
│   ├── hybrid_top100.json
│   ├── ce_reranked.json
│   └── monot5_reranked.json
├── hf_cache/                       # Model weights local (không push lên git)
│   ├── minilm/                     # all-MiniLM-L6-v2
│   ├── cross_encoder_minilm/       # ms-marco-MiniLM-L-6-v2
│   └── monot5_base/                # monot5-base-msmarco
├── reports/
│   ├── corpus_stats.md
│   ├── baseline_metrics.json
│   ├── bm25_error_analysis.md
│   ├── dense_metrics.json
│   ├── hybrid_metrics.json
│   ├── recall_comparison.md
│   ├── ce_latency.md
│   ├── reranker_latency_comparison.md
│   ├── final_metrics_table.json
│   ├── full_pipeline_comparison.png
│   ├── rank_shift_analysis.md
│   ├── reranker_error_cases.md
│   └── architecture_recommendation.md
├── tests/
│   ├── test_parser.py
│   ├── test_bm25_output.py
│   └── test_rrf.py
├── HANDOFF_LOG.md
├── requirements_kaggle.txt
└── README.md
```

---

## Checklist Phase Gate

### ✅ Phase 1 Done khi:
- [ ] `corpus.json` có đúng 1.460 entries, mỗi entry có `doc_id`, `title`, `text` (M1)
- [ ] `queries.json` có đúng 112 entries (M1)
- [ ] `qrels.json` parse xong từ CISI.REL, không bị thiếu query (M1)
- [ ] M2 và M3 đã tải được model weights về Kaggle session local thành công
- [ ] Smoke test: Cross-Encoder và MonoT5 chạy được trên 5 cặp mẫu (M3)

### ✅ Phase 2 Done khi:
- [ ] BM25 baseline MRR ≥ 0.25 (nếu thấp hơn: kiểm tra lại tokenizer) (M1)
- [ ] `doc_embeddings.npy` shape = (1460, 384) không chứa NaN (M2)
- [ ] Dense Retrieval Recall@100 ≥ 60% trên 112 queries (M2)
- [ ] `bm25_top100.json` và `dense_top100.json` đã được handoff (M1, M2)
- [ ] M3 đã test Cross-Encoder thành công trên `bm25_top100.json` tạm (M3)

### ✅ Phase 3 Done khi:
- [ ] `hybrid_top100.json` Recall@100 cao hơn `bm25_top100.json` Recall@100 (M2)
- [ ] Cross-Encoder re-ranking MRR ≥ 0.55 (M3)
- [ ] MonoT5 re-ranking MRR ≥ 0.55 (M3)
- [ ] Latency Cross-Encoder < 150ms/query trên T4 GPU (M3)

### ✅ Phase 4 Done khi:
- [ ] Bảng so sánh 6 hệ thống (TF-IDF / BM25 / Dense / Hybrid / CE / MonoT5) hoàn chỉnh với MRR và P@10 (M3)
- [ ] Rank shift analysis: chứng minh được tài liệu chính xác "dịch chuyển" lên Top-5 sau re-ranking (M3)
- [ ] Mỗi thành viên đã viết xong section báo cáo của mình (M1, M2, M3)
- [ ] Toàn bộ notebook chạy được end-to-end từ đầu đến cuối trong Kaggle Session mới (clean run)

---

## Shared Responsibilities (Tất cả thành viên)

| Trách nhiệm chung | Mô tả |
|---|---|
| **Kaggle Notebook tái chạy được** | Mỗi notebook phải `Run All` được trong Kaggle Session mới không có cache |
| **Không hardcode path** | Dùng `BASE_DIR = '/kaggle/working'` hoặc biến môi trường, không dùng path tuyệt đối cứng |
| **Seed cố định** | Tất cả random operation dùng `SEED = 42` để kết quả reproducible |
| **Ghi log rõ ràng** | Mỗi bước xử lý in ra: số docs xử lý, thời gian, kết quả metric tạm thời |
| **Git workflow** | Làm việc trên `feature/<tên-task>` branch, không push thẳng vào `main` |
| **Docstring đầy đủ** | Mỗi hàm phải có docstring: mô tả Input/Output/Example |
| **Meeting hàng tuần** | Báo cáo tiến độ, blockers, điều chỉnh kế hoạch |
| **Báo cáo cuối kỳ** | Mỗi người viết section của mình; M3 tổng hợp bảng so sánh và kết luận chung |
