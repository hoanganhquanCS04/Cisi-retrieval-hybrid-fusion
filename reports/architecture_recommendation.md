# Báo cáo Hiệu năng Toàn bộ Pipeline

## So sánh chỉ số các hệ thống

| Hệ thống | MRR | P@10 |
|--------|-----|------|
| BM25 | 0.6436 | 0.3105 |
| Dense | 0.6116 | 0.3987 |
| Hybrid | 0.6179 | 0.3803 |
| TF-IDF | 0.6018 | 0.3145 |
| Cross-Encoder | 0.5906 | 0.3408 |
| MonoT5 | 0.4463 | 0.2658 |

## Phân tích hiệu năng

**Hệ thống hoạt động tốt nhất:** BM25 (MRR: 0.6436)

**So sánh (thay đổi MRR so với BM25):**
- Cross-Encoder: -0.0530
- MonoT5: -0.1974
- Dense: -0.0321
- Hybrid: -0.0257

## Ghi chú kỹ thuật
- Đánh giá dựa trên ground truth của tập dữ liệu CISI
- Chỉ số: MRR (Mean Reciprocal Rank), P@10 (Precision@10)
- Tất cả hệ thống được đánh giá trên cùng 112 truy vấn
- Cross-Encoder: điểm liên quan dựa trên mô hình BERT
- MonoT5: tiếp cận dạng text-to-text
