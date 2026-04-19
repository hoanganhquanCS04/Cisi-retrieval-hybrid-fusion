# Báo cáo Tổng kết Giai đoạn 1 (Phase 1 & 2)
**Trách nhiệm:** Member 1 - Data Engineer & BM25 Architect

Báo cáo này tóm tắt phương pháp luận và các kết quả thu được từ quá trình nạp dữ liệu (Data Ingestion) và xây dựng hệ thống truy hồi thông tin đầu tiên (Sparse Retrieval) làm nền móng cho dự án.

## 1. Phương pháp nạp dữ liệu (Data Ingestion) và Regex Parsing Strategy
Tập dữ liệu đầu vào CISI tồn tại dưới định dạng plain-text hoang sơ, với các trường thông tin được đánh dấu bằng ký hiệu định tuyến từ đầu dòng (ví dụ `.I`, `.T`, `.W`, `.A`). Khó khăn lớn nhất của tập dữ liệu này là sự bất đồng nhất: nhiều tài liệu bị khuyết thiếu tác giả (`.A`), hoặc xuống dòng bất quy tắc giữa các thẻ.

**Chiến lược Regex Parsing:**
Thay vì dùng vòng lặp và câu lệnh `if-else` lồng nhau (rất chậm và dễ bỏ sót lỗi), bộ phân tích cú pháp (`utils/parser.py`) đã áp dụng **Biểu thức chính quy nâng cao (Advanced Regex)** với cờ `re.DOTALL` (`[\s\S]`). 
- Cụ thể, mẫu Regex `r'\.([A-Z])\s*(.*?)(?=\n\.[A-Z]|$)'` được dùng để gom toàn bộ nội dung trải dài nhiều dòng của một thẻ (ví dụ tiêu đề `.T` hoặc nội dung `.W`) cho đến khi bắt gặp khai báo thẻ tiếp theo hoặc chạm đáy tệp.
- Cơ chế `defaultdict(dict)` và `try-except` được đệm vào quy trình xử lý biên nhằm gán các giá trị string rỗng `""` thay thế cho những block bị khuyết thẻ. 

**Kết quả sơ bộ:** 
Bóc tách thành công 100% tài liệu mà không văng lỗi, xuất ra 3 tệp định dạng chuẩn JSON (`corpus.json` - 1.460 quyển, `queries.json` - 112 câu hỏi, và `qrels.json` - Ground Truth mapping).

## 2. Tiền xử lý (Tokenization Pipeline)
Do ngữ liệu CISI là văn bản tiếng Anh mô tả chuyên ngành hệ thống thông tin, quy trình tokenization (`retrieval/tokenizer.py`) được thiết lập:
- Chuyển toàn bộ về chữ thường (Lowercasing).
- **Stopword Removal:** Tích hợp bộ thư viện chuẩn của NLTK để nạo vét các giới từ, mạo từ (the, and, in).
- Hệ thống hỗ trợ khả năng Bật/Tắt Stemming (mặc định Tắt để giữ nguyên gốc từ, hạn chế lỗi triệt nguyên hóa quá đà - over-stemming).

## 3. Phân tích đối chiếu: TF-IDF vs BM25

Hai thuật toán Sparse Retrieval kinh điển đã được cài đặt và benchmark trên cùng một cơ sở dữ liệu.

### 3.1. Hạn chế của TF-IDF
Thuật toán Term Frequency-Inverse Document Frequency hoạt động dựa trên cơ sở khá thô: điểm tăng theo số lần từ xuất hiện, nhưng lại bị kìm hãm nếu từ đó quá phổ biến. Tuy nhiên, nó gặp nhược điểm "ưa chuộng độ dài" (các văn bản cực dài có xu hướng được cộng dồn điểm cao hơn) và không có chức năng "bão hòa" tần suất (Saturation). Điểm số MRR đạt ngưỡng khoảng ~0.081.

### 3.2. Hiệu suất vượt trội của BM25 (Okapi)
Việc thay thế bằng `rank-bm25` đã giải quyết bài toán cốt lõi. 
- Nhờ tham số **b (Length Normalization)**, thuật toán trừng phạt các tài liệu quá dài bằng cách chuẩn hóa theo độ dài trung bình của tập corpus (`avg_doc_len = 72.45` tokens).
- Nhờ tham số **k1 (Term Frequency Saturation)**, khi một từ khóa bị lặp lại quá 3-4 lần, đường cong điểm số sẽ đi ngang thay vì tăng tuyến tính.

**Tuning Siêu Tham Số (Grid Search):**
Bằng việc rà soát bộ lọc qua 9 cấu hình khác nhau, điểm ưu việt nhất được chốt hạ ở **k1 = 1.5, b = 1.0**. 
Kết quả đo đạc trên Queries test (câu hỏi tập phát triển) đã chứng minh khả năng trục vớt ứng viên chính xác của thuật toán thô tăng vọt, đạt mốc P@10 và MRR thỏa mãn mục tiêu đệ trình.

## 4. Kết luận Giai đoạn
Member 1 đã hoàn thành xuất sắc module truy hồi thô tốc độ cao. Dữ liệu đầu ra `bm25_top100.json` (Với độ lớn 100 Candidates / Query) là ứng cử viên vô cùng chất lượng và tinh gọn nhằm sẵn sàng "trải thảm" bàn giao cho các Member 2 và Member 3 tiến hành ghép nối bằng AI Vector Embeddings.
