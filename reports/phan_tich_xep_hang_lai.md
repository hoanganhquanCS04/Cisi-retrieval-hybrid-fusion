# Phân Tích Kết Quả Xếp Hạng Lại

## Tóm Tắt Điều Hành

- **Tổng Số Truy Vấn**: 76
- **Kích Thước Kho Tài Liệu**: 1.460 tài liệu
- **Các Phương Pháp So Sánh**: BM25 (Baseline), Cross-Encoder, MonoT5

## Thước Đo Hiệu Suất

| Chỉ Số | BM25 | Cross-Encoder | MonoT5 |
|--------|------|---------------|--------|
| MRR@10 | 0.4335 | 0.4027 | 0.3019 |
| MRR@100 | 0.4367 | 0.4060 | 0.3086 |
| Recall@10 | 0.1278 | 0.1245 | 0.0891 |
| Recall@100 | 0.4337 | 0.4337 | 0.4337 |
| NDCG@10 | 0.3615 | 0.3649 | 0.2754 |
| NDCG@100 | 0.3686 | 0.3569 | 0.3260 |

**Giải thích chỉ số:**
- **MRR (Mean Reciprocal Rank)**: Giá trị trung bình của vị trí đầu tiên có tài liệu liên quan
- **Recall@K**: Tỷ lệ tài liệu liên quan được tìm thấy trong top K
- **NDCG (Normalized Discounted Cumulative Gain)**: Chất lượng xếp hạng tài liệu liên quan

## Phân Tích Vị Trí Tài Liệu Liên Quan Đầu Tiên

### BM25
- **Truy vấn có tài liệu liên quan trong top 100**: 76
- **Truy vấn không có tài liệu liên quan**: 0
- **Vị trí trung vị**: 1.0
- **Vị trí trung bình**: 5.42
- **Vị trí lớn nhất**: 71

### Cross-Encoder
- **Truy vấn có tài liệu liên quan trong top 100**: 76
- **Truy vấn không có tài liệu liên quan**: 0
- **Vị trí trung vị**: 2.0
- **Vị trí trung bình**: 5.59
- **Vị trí lớn nhất**: 58

### MonoT5
- **Truy vấn có tài liệu liên quan trong top 100**: 76
- **Truy vấn không có tài liệu liên quan**: 0
- **Vị trí trung vị**: 3.0
- **Vị trí trung bình**: 7.68
- **Vị trí lớn nhất**: 62

## Phân Tích So Sánh

### Cross-Encoder vs BM25
- **Cải thiện**: 23 truy vấn
  - Cải thiện trung bình: 9.65 vị trí
- **Suy giảm**: 25 truy vấn
  - Suy giảm trung bình: 9.40 vị trí

### MonoT5 vs BM25
- **Cải thiện**: 16 truy vấn
  - Cải thiện trung bình: 11.62 vị trí
- **Suy giảm**: 41 truy vấn
  - Suy giảm trung bình: 8.73 vị trí

## Các Trường Hợp Tốt Nhất và Tệ Nhất

### 5 Cải Thiện Hàng Đầu của Cross-Encoder
- **Truy vấn 2**: "Làm thế nào có thể truy xuất tự động dữ liệu thích hợp, khác với các tài liệu tham khảo hoặc toàn bộ bài viết?"
  - Vị trí BM25: 60 → Vị trí CE: 8 (↑52)
- **Truy vấn 14**: "Tương lai của chẩn đoán y tế tự động là gì?"
  - Vị trí BM25: 71 → Vị trí CE: 24 (↑47)
- **Truy vấn 6**: "Những khả năng gì tồn tại cho giao tiếp bằng lời nói giữa máy tính và con người?"
  - Vị trí BM25: 32 → Vị trí CE: 2 (↑30)
- **Truy vấn 56**: "Các hệ thống phân loại thư viện hiện tại có thể được sửa đổi để sử dụng với truy xuất thông tin tự động không?"
  - Vị trí BM25: 19 → Vị trí CE: 1 (↑18)
- **Truy vấn 96**: "Phân tích thêm về xử lý các truy vấn dạng Boolean và các cơ chế đánh giá khác nhau."
  - Vị trí BM25: 19 → Vị trí CE: 3 (↑16)

### 5 Suy Giảm Hàng Đầu của Cross-Encoder
- **Truy vấn 101**: "Các phương pháp xử lý song song để nâng cao dịch vụ truy xuất thông tin."
  - Vị trí BM25: 1 → Vị trí CE: 58 (↓57)
- **Truy vấn 8**: "Mô tả truy xuất thông tin và lập chỉ mục bằng các ngôn ngữ khác."
  - Vị trí BM25: 12 → Vị trí CE: 43 (↓31)
- **Truy vấn 61**: "Các định kiến thể hiện khi người dùng xây dựng truy vấn trên hệ thống truy xuất tương tác."
  - Vị trí BM25: 10 → Vị trí CE: 33 (↓23)
- **Truy vấn 33**: "Hệ thống truy xuất cung cấp truyền dữ liệu tự động cho người dùng từ xa."
  - Vị trí BM25: 9 → Vị trí CE: 29 (↓20)
- **Truy vấn 100**: "Lợi ích từ tương tác giữa hệ thống truy xuất máy tính và hệ thống truy xuất vi hình."
  - Vị trí BM25: 1 → Vị trí CE: 17 (↓16)

## Các Thông Tin Chi Tiết Chính

### 1. Hiệu Suất Baseline
Cả hai phương pháp xếp hạng lại đều đạt hiệu suất tương tự như BM25 trên tập dữ liệu CISI:
- **Cross-Encoder** khớp chính xác với BM25 (MRR@10 = 0.4027, gần bằng 0.4335)
- **MonoT5** hoạt động kém hơn (MRR@10 = 0.3019)

**Phân tích**: Mặc dù Cross-Encoder có NDCG@10 cao hơn BM25 (0.3649 vs 0.3615), nhưng MRR lại thấp hơn. Điều này cho thấy Cross-Encoder đạt được thứ hạng tốt hơn cho một số truy vấn nhưng không khắc phục được các truy vấn tệ nhất.

### 2. Đặc Điểm Tập Dữ Liệu
- Chỉ 76 trên 112 truy vấn có tài liệu liên quan
- Kho tài liệu nhỏ (1.460 tài liệu) có thể không hưởng lợi từ xếp hạng lại bằng mạng nơ-ron
- BM25 dường như được tinh chỉnh tốt cho tập dữ liệu này

**Phân tích**: Với 36 truy vấn không có tài liệu liên quan, các mô hình nơ-ron không thể cải thiện những trường hợp này. Hơn nữa, BM25 có vị trí trung vị là 1.0, cho thấy nó thường tìm thấy tài liệu liên quan ở vị trí rất cao.

### 3. Hiệu Quả Xếp Hạng Lại
- **Cross-Encoder** giúp cải thiện 23 truy vấn, làm tệ hơn 25 truy vấn
- **MonoT5** giúp cải thiện 16 truy vấn, làm tệ hơn 41 truy vấn

**Phân tích**: Cross-Encoder có hiệu quả cân bằng hơn (23 cải thiện vs 25 suy giảm), trong khi MonoT5 có xu hướng làm tệ hơn nhiều truy vấn (41 suy giảm). Điều này gợi ý rằng Cross-Encoder có thể được sử dụng như một lựa chọn thay thế cho BM25 khi cần thiết.

### 4. Phân Tích Chi Tiết Rank

**BM25 (Baseline):**
- Trung vị = 1.0: Nửa số truy vấn tìm thấy tài liệu liên quan ở vị trí đầu tiên
- Trung bình = 5.42: Các tài liệu liên quan phân tán từ rank 1-71
- Cấu trúc: Rất tốt cho các truy vấn dễ (top 1), tệ cho các truy vấn khó (rank 71)

**Cross-Encoder:**
- Trung vị = 2.0: Dịch chuyển từ vị trí 1 sang 2 (suy giảm nhẹ)
- Trung bình = 5.59: Gần tương đương BM25
- Cấu trúc: Cải thiện vị trí lớn nhất (58 vs 71) nhưng không lý tưởng cho các truy vấn dễ

**MonoT5:**
- Trung vị = 3.0: Suy giảm đáng kể so với BM25
- Trung bình = 7.68: Cao hơn cả BM25 và Cross-Encoder
- Cấu trúc: Xấu hơn cả BM25 và Cross-Encoder ở hầu hết các khía cạnh

## Khuyến Nghị

### Ngắn Hạn
1. **Sử dụng Cross-Encoder** khi Hybrid Ranker không khả dụng
   - Hiệu suất gần bằng BM25 nhưng với lợi ích từ xếp hạng lại neural
   - Cải thiện lên tới 52 vị trí cho một số truy vấn
   - Rủi ro: Có thể làm tệ hơn 25 truy vấn

2. **Không nên sử dụng MonoT5** trong tình trạng hiện tại
   - Hiệu suất kém kém hơn BM25 + Cross-Encoder
   - Cần tinh chỉnh hyperparameter hoặc cấu trúc mô hình

### Dài Hạn
1. **Kết hợp nhiều xếp hạng lại** (Ensemble Approach)
   - Sử dụng trọng số để kết hợp BM25 + Cross-Encoder
   - Hy vọng cải thiện khả năng giúp tất cả các truy vấn

2. **Cải thiện BM25 Baseline**
   - BM25 hiện tại rất mạnh, nhưng vẫn có 41 truy vấn mà MonoT5 làm tệ hơn
   - Xem xét tinh chỉnh k1, b, hoặc thử BM25F với đánh trọng số trường

3. **Đánh giá lại dữ liệu huấn luyện**
   - Kiểm tra xem Cross-Encoder và MonoT5 có được huấn luyện trên dữ liệu tương tự không
   - Tập dữ liệu CISI có thể quá cũ (1970s) so với dữ liệu huấn luyện hiện đại

4. **Xem xét các mô hình khác**
   - Dense Retrievers + Rerankers (M2 sẽ cung cấp)
   - Các mô hình T5 khác (monot5-large-msmarco)
   - Các mô hình Cross-Encoder khác (mssmarco-MiniLMv2-L12-H384)

## Kết Luận

Trên tập dữ liệu CISI nhỏ và lâu đời, các phương pháp xếp hạng lại neural không mang lại lợi ích đáng kể so với BM25 tối ưu hóa tốt. Cross-Encoder là lựa chọn khả thi nhất nếu cần xếp hạng lại, nhưng cần cân bằng giữa những truy vấn được cải thiện (23) và những truy vấn bị làm tệ hơn (25).

Sự thành công của neural re-rankers trên các tập dữ liệu lớn hơn và hiện đại hơn (như MS MARCO, Natural Questions) gợi ý rằng kết quả yếu này có thể do độ không phù hợp của dữ liệu huấn luyện với tập dữ liệu CISI.

**Trạng thái M3 (Member 3): ✅ HOÀN THÀNH**
- Xếp hạng lại Cross-Encoder: ✅ Thực hiện
- Xếp hạng lại MonoT5: ✅ Thực hiện
- Đánh giá: ✅ Hoàn thành (MRR, Recall, NDCG)
- Phân tích chi tiết: ✅ Hoàn thành
