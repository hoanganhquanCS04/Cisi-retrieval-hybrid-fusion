# Phân tích dịch chuyển thứ hạng: Ảnh hưởng của Neural Re-ranking

Phân tích cách các tài liệu liên quan (relevant) thay đổi vị trí trong bảng xếp hạng sau khi áp dụng re-ranking bằng Cross-Encoder và MonoT5.

## Phương pháp
- So sánh vị trí của các tài liệu relevant giữa baseline Hybrid và kết quả re-ranking (Cross-Encoder, MonoT5).
- **Δ (Delta):** Δ = Hạng_Hybrid − Hạng_Re-ranker. Giá trị Δ dương nghĩa là tài liệu được đẩy lên (cải thiện), Δ âm nghĩa là tài liệu bị tụt.
- Khi cần đánh giá theo RR (Reciprocal Rank): RR = 1 / rank nếu rank ≤ 10, ngược lại RR = 0.
- Phân tích trên 10 truy vấn đại diện từ bộ dữ liệu (bảng chi tiết bên dưới).

## Bảng kết quả (chi tiết)

| Truy vấn | Doc | Hạng Hybrid | Hạng CE | Hạng MonoT5 | Δ CE | Δ T5 |
|---------:|----:|------------:|--------:|------------:|-----:|-----:|
| 1 | 1281 | 3 | 12 | 78 | -9 | -75 |
| 1 | 650 | 8 | 9 | 2 | -1 | +6 |
| 1 | 1162 | 39 | 54 | 54 | -15 | -15 |
| 1 | 524 | 24 | 35 | 23 | -11 | +1 |
| 1 | 269 | 27 | 47 | 81 | -20 | -54 |
| 1 | 1164 | 41 | 55 | 49 | -14 | -8 |
| 1 | 783 | 44 | 62 | 39 | -18 | +5 |
| 1 | 28 | 38 | 43 | 53 | -5 | -15 |
| 1 | 541 | 52 | 14 | 17 | +38 | +35 |
| 1 | 38 | 9 | 16 | 20 | -7 | -11 |
| 1 | 43 | 77 | 39 | 46 | +38 | +31 |
| 1 | 1195 | 4 | 25 | 79 | -21 | -75 |
| 1 | 429 | 2 | 2 | 7 | +0 | -5 |
| 1 | 813 | 10 | 34 | 22 | -24 | -12 |
| 1 | 1196 | 48 | 44 | 31 | +4 | +17 |
| 1 | 52 | 28 | 60 | 45 | -32 | -17 |
| 1 | 820 | 29 | 13 | 12 | +16 | +17 |
| 1 | 195 | 20 | 48 | 96 | -28 | -76 |
| 1 | 582 | 22 | 8 | 9 | +14 | +13 |
| 1 | 711 | 6 | 58 | 38 | -52 | -32 |
| 1 | 76 | 16 | 52 | 58 | -36 | -42 |
| 1 | 589 | 5 | 19 | 15 | -14 | -10 |
| 1 | 466 | 23 | 53 | 37 | -30 | -14 |
| 1 | 722 | 1 | 4 | 1 | -3 | +0 |
| 1 | 86 | 63 | 72 | 40 | -9 | +23 |
| 1 | 215 | 36 | 59 | 87 | -23 | -51 |
| 1 | 726 | 12 | 63 | 36 | -51 | -24 |
| 1 | 603 | 15 | 11 | 21 | +4 | -6 |
| 1 | 483 | 35 | 30 | 27 | +5 | +8 |
| 1 | 868 | 45 | 45 | 43 | +0 | +2 |
| 1 | 869 | 14 | 31 | 44 | -17 | -30 |
| 1 | 510 | 7 | 6 | 8 | +1 | -1 |
| 2 | 319 | 89 | 16 | 34 | +73 | +55 |
| 2 | 68 | 43 | 25 | 35 | +18 | +8 |
| 3 | 640 | 11 | 5 | 3 | +6 | +8 |
| 3 | 131 | 26 | 8 | 8 | +18 | +18 |
| 3 | 133 | 34 | 29 | 21 | +5 | +13 |
| 3 | 1169 | 3 | 6 | 2 | -3 | +1 |
| 3 | 537 | 27 | 12 | 9 | +15 | +18 |
| 3 | 1179 | 13 | 14 | 30 | -1 | -17 |
| 3 | 540 | 28 | 21 | 28 | +7 | +0 |
| 3 | 1181 | 4 | 10 | 15 | -6 | -11 |
| 3 | 803 | 5 | 27 | 4 | -22 | +1 |
| 3 | 1190 | 31 | 24 | 22 | +7 | +9 |
| 3 | 554 | 33 | 7 | 12 | +26 | +21 |
| 3 | 1326 | 9 | 47 | 55 | -38 | -46 |
| 3 | 60 | 7 | 4 | 6 | +3 | +1 |
| 3 | 585 | 39 | 35 | 13 | +4 | +26 |
| 3 | 85 | 8 | 3 | 11 | +5 | -3 |
| 3 | 469 | 1 | 1 | 7 | +0 | -6 |
| 3 | 599 | 2 | 2 | 1 | +0 | +1 |
| 3 | 346 | 14 | 36 | 45 | -22 | -31 |
| 3 | 363 | 29 | 9 | 14 | +20 | +15 |
| 3 | 114 | 85 | 62 | 69 | +23 | +16 |
| 3 | 372 | 12 | 22 | 39 | -10 | -27 |
| 4 | 321 | 25 | 28 | 91 | -3 | -66 |
| 4 | 420 | 11 | 7 | 3 | +4 | +8 |
| 4 | 332 | 8 | 16 | 32 | -8 | -24 |
| 4 | 980 | 6 | 6 | 14 | +0 | -8 |
| 4 | 601 | 1 | 1 | 1 | +0 | +0 |
| 5 | 648 | 1 | 1 | 26 | +0 | -25 |
| 5 | 137 | 94 | 5 | 41 | +89 | +53 |
| 5 | 525 | 16 | 10 | 11 | +6 | +5 |
| 5 | 451 | 7 | 43 | 20 | -36 | -13 |
| 5 | 453 | 73 | 63 | 63 | +10 | +10 |
| 5 | 1356 | 56 | 72 | 46 | -16 | +10 |
| 5 | 471 | 5 | 16 | 18 | -11 | -13 |
| 5 | 241 | 12 | 34 | 45 | -22 | -33 |
| 5 | 114 | 10 | 8 | 10 | +2 | +0 |
| 6 | 400 | 32 | 2 | 1 | +30 | +31 |
| 7 | 725 | 13 | 3 | 3 | +10 | +10 |
| 7 | 376 | 7 | 2 | 21 | +5 | -14 |
| 8 | 1024 | 19 | 80 | 62 | -61 | -43 |
| 8 | 1161 | 43 | 77 | 52 | -34 | -9 |
| 9 | 1164 | 96 | 54 | 82 | +42 | +14 |
| 9 | 1294 | 5 | 48 | 20 | -43 | -15 |
| 9 | 1323 | 2 | 2 | 58 | +0 | -56 |
| 9 | 175 | 3 | 3 | 9 | +0 | -6 |
| 9 | 1327 | 13 | 15 | 7 | -2 | +6 |
| 9 | 179 | 7 | 5 | 2 | +2 | +5 |
| 9 | 572 | 20 | 57 | 37 | -37 | -17 |
| 9 | 577 | 55 | 68 | 38 | -13 | +17 |
| 9 | 1224 | 24 | 28 | 30 | -4 | -6 |
| 9 | 1120 | 4 | 18 | 48 | -14 | -44 |
| 9 | 1130 | 88 | 69 | 66 | +19 | +22 |
| 10 | 25 | 2 | 22 | 34 | -20 | -32 |
| 10 | 175 | 12 | 15 | 13 | -3 | -1 |
| 10 | 829 | 10 | 2 | 11 | +8 | -1 |
| 10 | 462 | 1 | 1 | 19 | +0 | -18 |
| 10 | 1117 | 32 | 39 | 2 | -7 | +30 |

## Bảng tóm tắt: tài liệu 'đúng' theo Hybrid (first relevant)

Bảng dưới đây liệt kê cho mỗi truy vấn (1–10) tài liệu relevant mà Hybrid xếp hạng cao nhất (rank nhỏ nhất). Bảng in rõ hạng trước/sau re-ranking và RR tương ứng để dễ xác định xem tài liệu "đúng" có bị dịch ra khỏi top hay không.

| Truy vấn | Doc | Hạng Hybrid | Hạng CE | Hạng MonoT5 | Δ CE | Δ T5 | RR Hybrid | RR CE | RR MonoT5 |
|--------:|----:|------------:|--------:|------------:|-----:|-----:|---------:|-----:|---------:|
| 1 | 722 | 1 | 4 | 1 | -3 | 0 | 1.000 | 0.250 | 1.000 |
| 2 | 68 | 43 | 25 | 35 | 18 | 8 | 0.000 | 0.000 | 0.000 |
| 3 | 469 | 1 | 1 | 7 | 0 | -6 | 1.000 | 1.000 | 0.143 |
| 4 | 601 | 1 | 1 | 1 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| 5 | 648 | 1 | 1 | 26 | 0 | -25 | 1.000 | 1.000 | 0.000 |
| 6 | 400 | 32 | 2 | 1 | 30 | 31 | 0.000 | 0.500 | 1.000 |
| 7 | 376 | 7 | 2 | 21 | 5 | -14 | 0.143 | 0.500 | 0.000 |
| 8 | 1024 | 19 | 80 | 62 | -61 | -43 | 0.000 | 0.000 | 0.000 |
| 9 | 1323 | 2 | 2 | 58 | 0 | -56 | 0.500 | 0.500 | 0.000 |
| 10 | 462 | 1 | 1 | 19 | 0 | -18 | 1.000 | 1.000 | 0.000 |

## Key Findings (tính toán từ bảng trên)

- **Tổng số so sánh:** 90 (tất cả các cặp (query, doc relevant) trong 10 truy vấn mẫu).
- **Cross-Encoder (CE):** cải thiện 34/90 (38%), làm tệ đi 46/90 (51%), không đổi 10/90 (11%).
	- Mean (CE delta) = -3.3222 vị trí (nghĩa là trung bình CE làm giảm vị trí 3.3 bậc).
	- Median = -1 vị trí (nửa số trường hợp CE làm tệ hơn ít nhất 1 bậc).
- **MonoT5 (T5):** cải thiện 39/90 (43%), làm tệ đi 47/90 (52%), không đổi 4/90 (4%).
	- Mean (T5 delta) = -6.3111 vị trí (trung bình T5 làm giảm vị trí 6.3 bậc).
	- Median = -1 vị trí.

### Các ví dụ điển hình (một số trường hợp cải thiện/suy giảm mạnh)

- Top cải thiện của CE (CE delta lớn nhất):
	1. Query 5, Doc 137: Hybrid 94 → CE 5 (Δ = +89)
	2. Query 2, Doc 319: Hybrid 89 → CE 16 (Δ = +73)
	3. Query 9, Doc 1164: Hybrid 96 → CE 54 (Δ = +42)
	4. Query 1, Doc 541: Hybrid 52 → CE 14 (Δ = +38)
	5. Query 1, Doc 43: Hybrid 77 → CE 39 (Δ = +38)

- Trường hợp CE làm tệ nhất (CE delta âm lớn):
	1. Query 3, Doc 1326: Hybrid 9 → CE 47 (Δ = -38)
	2. Query 9, Doc 1294: Hybrid 5 → CE 48 (Δ = -43)
	3. Query 1, Doc 726: Hybrid 12 → CE 63 (Δ = -51)
	4. Query 1, Doc 711: Hybrid 6 → CE 58 (Δ = -52)
	5. Query 8, Doc 1024: Hybrid 19 → CE 80 (Δ = -61)

- Top cải thiện của MonoT5 (T5 delta lớn nhất):
	1. Query 2, Doc 319: Hybrid 89 → T5 34 (Δ = +55)
	2. Query 5, Doc 137: Hybrid 94 → T5 41 (Δ = +53)
	3. Query 1, Doc 541: Hybrid 52 → T5 17 (Δ = +35)
	4. Query 1, Doc 43: Hybrid 77 → T5 46 (Δ = +31)
	5. Query 6, Doc 400: Hybrid 32 → T5 1 (Δ = +31)

- Trường hợp MonoT5 làm tệ nhất:
	1. Query 9, Doc 1323: Hybrid 2 → T5 58 (Δ = -56)
	2. Query 4, Doc 321: Hybrid 25 → T5 91 (Δ = -66)
	3. Query 1, Doc 1281: Hybrid 3 → T5 78 (Δ = -75)
	4. Query 1, Doc 1195: Hybrid 4 → T5 79 (Δ = -75)
	5. Query 1, Doc 195: Hybrid 20 → T5 96 (Δ = -76)

## Interpretation (diễn giải)

- Mặc dù cả hai mô hình re-ranking có thể đem lại lợi ích lớn cho một số ít cặp (cải thiện rất nhiều vị trí), nhưng trên tổng thể cả CE và MonoT5 đều có xu hướng làm giảm vị trí trung bình của các tài liệu relevant trong tập mẫu này (mean âm). Điều này nghĩa là: một phần nhỏ trường hợp hưởng lợi mạnh, nhưng số lượng trường hợp bị tổn hại nhiều hơn dẫn tới hiệu ứng ròng tiêu cực.
- **So sánh chung:** CE có mean gần -3.3 (ít gây suy giảm hơn), MonoT5 có mean mạnh hơn về chiều suy giảm (-6.3) — tức MonoT5 hành xử “mạnh tay” hơn: vừa có nhiều cú boost lớn, vừa có nhiều cú degrade lớn.
- **Hậu quả thực tiễn:** nếu đưa MonoT5/CE trực tiếp vào production mà không có biện pháp phòng ngừa, ta có thể cải thiện trải nghiệm cho vài truy vấn nhưng làm xấu đi cho nhiều truy vấn khác.

## Khuyến nghị thực tế

1. **Ngưỡng an toàn (score/threshold):** chỉ áp dụng re-ranking khi score/confidence của re-ranker vượt ngưỡng (tránh trường hợp rủi ro khi re-ranker không tự tin).
2. **Ensemble / fallback:** dùng ensemble (ví dụ trung bình xếp hạng) hoặc nếu re-ranked top-1 nằm ngoài top-K quá nhiều thì fallback về Hybrid.
3. **Phân loại truy vấn trước khi re-rank:** chỉ re-rank với MonoT5/CE cho những truy vấn dạng ngắn, rõ ràng hoặc thuộc các chủ đề mô hình đã được tune.
4. **Fine-tune và calibration:** fine-tune trên tập dữ liệu có tính chuyên ngành và điều chỉnh loss/threshold nhằm giảm lượng degrade.
5. **Theo dõi định kỳ:** lặp lại phân tích rank-shift sau mỗi thay đổi (báo cáo tương tự) để đảm bảo cải tiến thực sự.

## Trực quan hóa (script tham khảo)

Đoạn script Python sau parse bảng trong file này và vẽ "slope-graph" (mỗi đường nối vị trí Hybrid → CE → T5 cho một cặp). Lưu ảnh vào `reports/rank_shift_plot.png`.

```python
import matplotlib.pyplot as plt
from pathlib import Path

def parse_table(md_path):
		lines=Path(md_path).read_text().splitlines()
		for i,l in enumerate(lines):
				h=l.lower()
				# detect header line robustly (supports Vietnamese/English headers)
				if 'hybrid' in h and 'ce' in h and ('t5' in h or 'mono' in h):
						start=i+2
						break
		rows=[]
		for line in lines[start:]:
				if line.startswith('## '):
						break
				if not line.strip().startswith('|'):
						continue
				parts=[p.strip() for p in line.split('|')[1:-1]]
				if len(parts)<7:
						continue
				q,doc,hy,ce,t5,ced,t5d=parts[:7]
				try:
						rows.append((int(q),int(doc),int(hy),int(ce),int(t5)))
				except:
						continue
		return rows

rows=parse_table('reports/rank_shift_analysis.md')

fig,ax=plt.subplots(figsize=(8,10))
for q,doc,hy,ce,t5 in rows:
		xs=[0,1,2]
		ys=[hy,ce,t5]
		ax.plot(xs,ys,'-o',color='gray',alpha=0.6)

ax.set_xticks([0,1,2])
ax.set_xticklabels(['Hybrid','CE','MonoT5'])
ax.invert_yaxis()
ax.set_ylabel('Rank (1 = best)')
ax.set_title('Rank shift: Hybrid → CE → MonoT5 (10 queries sample)')
plt.tight_layout()
plt.savefig('reports/rank_shift_plot.png',dpi=200)
```

Chạy script trên (từ repository root):

```bash
python -c "$(sed -n '1,200p' reports/rank_shift_analysis.md | sed -n '/```python/,/```/p' | sed '1d;$d')"
```

Hoặc lưu đoạn script thành `scripts/plot_rank_shift.py` và chạy `python scripts/plot_rank_shift.py`.

