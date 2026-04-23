# Báo cáo Tổng hợp M3 (Phiên bản hoàn thiện)

**Nguồn candidate:** dense_top100.json

**Số truy vấn xử lý:** 112

# Bảng so sánh 6 hệ thống (MRR, P@10)

| Hệ Thống         |   MRR   |  P@10  |
|------------------|--------:|-------:|
| BM25             | 0.6436  | 0.3105 |
| Dense            | 0.6116  | 0.3987 |
| Hybrid           | 0.6179  | 0.3803 |
| TF-IDF           | 0.6018  | 0.3145 |
| Cross-Encoder    | 0.5906  | 0.3408 |
| MonoT5           | 0.4463  | 0.2658 |

# Tổng quan & nhận xét chính
- **BM25** vẫn là baseline mạnh về MRR trên tập CISI — thích hợp khi cần tối ưu thứ tự tổng thể.
- **Dense** và **Hybrid** cải thiện đáng kể P@10; nếu mục tiêu là đưa nhiều tài liệu liên quan lên top-10, `Dense`/`Hybrid` phù hợp.
- **Cross-Encoder (CE)** giúp gia tăng độ chính xác (P@10) khi dùng làm reranker, nhưng chi phí latency phải cân nhắc.
- **MonoT5** hiện cho hiệu năng kém hơn so với các phương án khác (MRR thấp và không ổn định); cần fine-tune thêm hoặc đặt dưới ensemble/heuristic.

# Độ trễ (Latency) của rerankers

| Model         |   Avg Latency/Query (ms) |
|:--------------|-------------------------:|
| Cross-Encoder |                   115.69 |
| MonoT5        |                   259.80 |

> Ghi chú: Cross-Encoder có độ trễ hợp lý cho các hệ thống yêu cầu độ chính xác trung bình-cao; MonoT5 hiện tốn tài nguyên và chậm hơn rõ rệt.

# Bảng chỉ số tổng hợp (Phần chính)

|                |    MRR |   P@10 |
|:---------------|-------:|-------:|
| BM25           | 0.6436 | 0.3105 |
| Dense          | 0.6116 | 0.3987 |
| Hybrid         | 0.6179 | 0.3803 |
| TF-IDF         | 0.6018 | 0.3145 |
| Cross-Encoder  | 0.5906 | 0.3408 |
| MonoT5         | 0.4463 | 0.2658 |

# Phân tích chi tiết

- **Tại sao BM25 dẫn đầu MRR?**
	- Tập CISI (văn bản khoa học/short passages) thường phù hợp với các bộ từ khóa và tần suất TF-IDF/BM25, nên BM25 vẫn giữ được thứ tự tốt cho nhiều truy vấn.

- **Dense / Hybrid tăng P@10 nhưng MRR thấp hơn BM25:**
	- Dense embeddings giúp bắt bắt được tương đồng ngữ nghĩa, đưa vào nhiều tài liệu liên quan hơn vào top-10 (tăng P@10) nhưng không luôn đảm bảo tài liệu "đúng nhất" lên hạng 1 (ảnh hưởng MRR).

- **Cross-Encoder:**
	- Tăng P@10 khi rerank top-k, thích hợp cho pipeline 2-stage (retriever + CE reranker). Chi phí latency ~115ms/query — phù hợp cho batch hoặc hệ thống có tài nguyên.

- **MonoT5:**
	- Kết quả chưa tốt: có thể do thiếu fine-tune trên tập dữ liệu tương ứng hoặc prompt/template không phù hợp. Đề xuất: fine-tune trên cặp query–doc, giảm max-length, hoặc kết hợp ensemble với CE để tránh suy giảm hiệu năng.

# Khuyến nghị (Actionable)

1. Nếu ưu tiên MRR (thứ tự chính xác tuyệt đối): giữ hoặc tinh chỉnh `BM25` làm baseline, kết hợp với `Hybrid` khi cần cân bằng P@10.
2. Nếu ưu tiên P@10 (top-10 chính xác): dùng `Dense`/`Hybrid` + `Cross-Encoder` reranker trên top-100; chấp nhận chi phí latency của CE.
3. MonoT5: thử fine-tune trên dữ liệu cục bộ, kiểm tra prompt/template, hoặc sử dụng như một thành phần thứ cấp (ensemble) thay vì thay thế CE.
4. Thêm kiểm thử A/B khi triển khai để đánh giá trade-off latency vs. gain thực tế.

# Liên kết & artifacts
- Biểu đồ rank-shift: [reports/rank_shift_plot.png](reports/rank_shift_plot.png)
- Báo cáo phân tích lỗi MonoT5: [reports/reranker_error_cases.md](reports/reranker_error_cases.md)
- Báo cáo phân tích biến động thứ hạng: [reports/rank_shift_analysis.md](reports/rank_shift_analysis.md)

# Phần mở rộng (nên làm tiếp)
- Thu thập thêm labels cho truy vấn khó và fine-tune MonoT5.
- Thử ensemble CE + MonoT5 để hạn chế suy giảm RR.
- Tối ưu latency bằng batching, model distillation hoặc sử dụng phiên bản nhỏ hơn của MonoT5/CE.

---

Phiên bản báo cáo: tự động sinh bởi pipeline đánh giá (đã localize sang tiếng Việt).