# Kế hoạch & Báo cáo kết quả: Nhiệm vụ của Member 2 (Dense & Hybrid Retrieval)

Tài liệu này tổng hợp toàn bộ các nhiệm vụ, kiến trúc kỹ thuật và kết quả mà Member 2 đã thực hiện trong Phase 2 và Phase 3 của dự án hệ thống tìm kiếm thông tin CISI.

## 1. Nhiệm vụ 1: Dense Retrieval (Tìm kiếm Ngữ nghĩa)
- **Mục tiêu:** Vượt qua rào cản "lỗ hổng từ vựng" (Lexical Gap) của hệ thống BM25 truyền thống bằng cách sử dụng công nghệ nhúng từ (Embedding).
- **Mô hình sử dụng:** `sentence-transformers/all-MiniLM-L6-v2` (Bản gọn nhẹ, tối ưu cho tốc độ và bộ nhớ).
- **Công cụ truy xuất:** Sử dụng **FAISS** (Facebook AI Similarity Search) với cấu trúc `IndexFlatIP`. Các vector được chuẩn hóa L2 (L2 Normalization) trước khi tìm kiếm để thuật toán Inner Product tự động tương đương với Cosine Similarity.
- **Kết quả đầu ra:** File `data/dense_top100.json` (Lưu top 100 văn bản có khoảng cách Vector gần nhất với mỗi câu hỏi).

### Metrics đánh giá độc lập của Dense:
Hệ thống thể hiện sự xuất sắc trong việc vớt đúng văn bản ngay ở Top 1 (MRR cao).
* **MRR:** 0.6116
* **P@10:** 0.3987
* **Recall@100:** 0.4706

---

## 2. Nhiệm vụ 2: Hybrid Retrieval (Hệ thống Lai ghép)
- **Mục tiêu:** Kết hợp ưu điểm "Tìm từ khóa chính xác" của BM25 (M1) và "Tìm ngữ nghĩa" của Dense (M2) để bù trừ điểm yếu cho nhau. Đặc biệt tập trung vào việc kéo chỉ số **Recall@100** tăng cao nhất có thể.
- **Phương pháp 1 (Reciprocal Rank Fusion - RRF):**
  - Thuật toán cộng điểm dựa trên thứ hạng (Rank-based), không cần quan tâm đến thang điểm gốc của mô hình.
  - Công thức: $Score = \frac{1}{k + rank_{BM25}} + \frac{1}{k + rank_{Dense}}$
  - Đã thử nghiệm với các hằng số $k \in \{10, 60, 100\}$. Mặc định chọn **k = 60** vì đây là hằng số vàng được chứng minh sự cân bằng tốt nhất trong các báo cáo khoa học.
- **Phương pháp 2 (Weighted Sum Fusion):**
  - Thực hiện Min-Max Normalization (đưa điểm về thang $0 \to 1$) rồi cộng theo tỷ trọng (ví dụ 50% BM25 - 50% Dense).

### Metrics đánh giá hệ thống Hybrid (RRF, k=60):
* **MRR:** 0.6179 *(Bảo toàn và tăng nhẹ so với 0.6116 của Dense)*
* **P@10:** 0.3803 *(Giảm nhẹ do nhiễu của BM25)*
* **Recall@100:** 0.5016 *(Tăng mạnh từ 47.06%, đạt đúng KPI của Team)*

---

## 3. Cấu trúc Source Code bàn giao
Member 2 chịu trách nhiệm bảo trì các tệp tin sau trong dự án:

1. `retrieval/dense_retriever.py`: Cấu trúc mô hình và FAISS Index.
2. `retrieval/rrf.py`: Thuật toán RRF và Weighted Fusion độc lập.
3. `retrieval/hybrid_pipeline.py`: Pipeline CLI (Command Line Interface) để dễ dàng chạy gộp từ Terminal.
4. `evaluate_metrics.py`: Script tự động chấm điểm ra báo cáo JSON theo chuẩn IR.
5. `notebooks/02_dense_hybrid.ipynb`: Notebook kịch bản chạy chính (có thể quăng lên Kaggle để lấy GPU).
6. `notebooks/rrf_tuning.ipynb`: Notebook chạy vòng lặp tuning tìm tham số $k$ tối ưu.

## 4. Kết luận
Phase 2 và Phase 3 đã khép lại thành công rực rỡ. Đầu ra `data/hybrid_top100.json` hiện đang chứa một danh sách kết quả có độ phủ (Recall) cực kỳ cao, là nguyên liệu đầu vào vô cùng hoàn hảo để giao cho **Member 3 (Cross-Encoder / MonoT5)** thực hiện bước sàng lọc cuối cùng (Re-ranking).
