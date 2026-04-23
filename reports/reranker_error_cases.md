# Phân Tích Lỗi MonoT5 Re-ranking


Báo cáo này phân tích sâu 5 truy vấn mà MonoT5 re-ranking thất bại nặng nề so với baseline Hybrid. Mục tiêu là chỉ ra nguyên nhân cụ thể, insight về hành vi mô hình, và đề xuất hướng cải thiện thực tế cho pipeline.

## Phương pháp & Ý nghĩa chỉ số
- **Reciprocal Rank (RR):** Đo lường vị trí tài liệu liên quan đầu tiên trong danh sách xếp hạng. RR = 1 nếu tài liệu đúng ở vị trí 1, RR = 0.5 nếu ở vị trí 2, v.v. RR = 0 nếu không có tài liệu đúng trong top-10.
- **Suy giảm (MonoT5 − Hybrid):** Suy giảm = RR_MonoT5 − RR_Hybrid. Giá trị âm nghĩa là MonoT5 kém hơn Hybrid (ví dụ RR_MonoT5=0 và RR_Hybrid=1 → Suy giảm = -1.0).
- **Lý do chọn 5 truy vấn:** Đây là các trường hợp có suy giảm RR lớn nhất, thể hiện rõ nhất điểm yếu của MonoT5.


## 1. Bảng tổng hợp 5 truy vấn lỗi nặng

| Query ID | Hybrid RR | MonoT5 RR | Suy Giảm | Số Doc Liên Quan |
|----------|-----------|-----------|----------|------------------|
| 79 | 1.000 | 0.000 | -1.000 | 11 |
| 97 | 1.000 | 0.000 | -1.000 | 6 |
| 5 | 1.000 | 0.100 | -0.900 | 24 |
| 62 | 1.000 | 0.100 | -0.900 | 12 |
| 66 | 1.000 | 0.100 | -0.900 | 35 |


## 2. Phân tích chi tiết từng truy vấn

### Trường hợp 1: Query 79
**Nội dung truy vấn:**
> Algorithms are given to process partially specified queries in a compressed database system. The proposed methods handle effectively queries that use either whole words or word fragments as language elements. The methods are compared and critically evaluated in terms of the design and retrieval costs. The analyses show that the method which exploits the interdependence of fragments as well as the relevance of fragments to records in the file has maximum design cost and least retrieval cost.

**Nhận xét:**
- Truy vấn mang tính kỹ thuật, đề cập đến "partially specified queries", "word fragments", "compressed database".
- Hybrid dễ dàng tìm đúng doc nhờ matching từ khóa kỹ thuật, còn MonoT5 gặp khó khăn với ngữ cảnh chuyên sâu, có thể không nhận diện đúng ý nghĩa "fragment" hoặc "interdependence".
- Top-5 của MonoT5 hoàn toàn không chứa tài liệu liên quan, cho thấy mô hình không "hiểu" được trọng tâm kỹ thuật của truy vấn này.

**Nguyên nhân tiềm năng:**
- MonoT5 chưa từng được fine-tune trên các truy vấn kỹ thuật dạng này, vốn hiếm trong tập huấn luyện phổ thông.
- Prompt dạng "Relevant:" có thể không đủ định hướng cho các truy vấn nhiều khái niệm lồng ghép.

**Insight:**
- MonoT5 có xu hướng "hallucinate" hoặc chọn doc có ngôn ngữ tự nhiên gần truy vấn, nhưng không đủ khả năng matching chính xác các khái niệm kỹ thuật đặc thù.

### Trường hợp 1: Query 79
- **Hybrid RR:** 1.000
- **MonoT5 RR:** 0.000
- **Suy Giảm:** -1.000
- **Số doc liên quan:** 11

**Top 5 Hybrid:**
1. Doc 450 ✓
2. Doc 487 ✗
3. Doc 523 ✗
4. Doc 318 ✗
5. Doc 1124 ✗

**Top 5 MonoT5:**
1. Doc 517 ✗
2. Doc 442 ✗
3. Doc 875 ✗
4. Doc 739 ✗
5. Doc 492 ✗

---

---

### Trường hợp 2: Query 97
**Nội dung truy vấn:**
> There has been a good deal of work on information retrieval systems that have continuous weights assigned to the index terms that describe the records in the database, and/or to the query terms that describe the user queries. Recent articles have analyzed retrieval systems with continuous weights of either type and/or with a Boolean structure for the queries. They have also suggested criteria which such systems ought to satisfy and record evaluation mechanisms which partially satisfy these criteria. We offer a more careful analysis, based on a generalization of the discrete weights. We also look at the weights from an entirely different approach involving thresholds, and we generate an improved evaluation mechanism which seems to fulfill a larger subset of the desired criteria than previous mechanisms. This new mechanism allows the user to attach a "threshold" to the query term.

**Nhận xét:**
- Truy vấn rất dài, nhiều mệnh đề, đề cập đến "continuous weights", "Boolean structure", "thresholds".
- Hybrid tận dụng tốt các từ khóa đặc biệt ("continuous weights", "thresholds"), còn MonoT5 có thể bị "loãng" thông tin, không xác định được trọng tâm.
- Top-5 MonoT5 toàn tài liệu không liên quan, cho thấy mô hình bị "overwhelmed" bởi truy vấn phức tạp.

**Nguyên nhân tiềm năng:**
- MonoT5 gặp khó với truy vấn dài, nhiều khái niệm trừu tượng, không có "anchor" rõ ràng để matching.
- Có thể mô hình ưu tiên các doc có ngôn ngữ tự nhiên tương tự, bỏ qua các doc kỹ thuật đúng.

**Insight:**
- MonoT5 không mạnh với các truy vấn cần matching logic/phép toán hoặc nhiều thuật ngữ chuyên ngành.
- **Hybrid RR:** 1.000
- **MonoT5 RR:** 0.000
- **Suy Giảm:** -1.000
- **Số doc liên quan:** 6

**Top 5 Hybrid:**
1. Doc 824 ✓
2. Doc 531 ✓
3. Doc 448 ✗
4. Doc 894 ✗
5. Doc 660 ✗

**Top 5 MonoT5:**
1. Doc 660 ✗
2. Doc 1091 ✗
3. Doc 894 ✗
4. Doc 1054 ✗
5. Doc 1124 ✗

---

---

### Trường hợp 3: Query 5
**Nội dung truy vấn:**
> What special training will ordinary researchers and businessmen need for proper information management and unobstructed use of information retrieval systems? What problems are they likely to encounter?

**Nhận xét:**
- Truy vấn về "training", "information management", "problems" cho người dùng phổ thông.
- Hybrid có thể tận dụng các từ khóa "training", "information retrieval systems" để tìm doc đúng.
- MonoT5 lại chọn các doc không liên quan, có thể do mô hình thiên về các doc có ngôn ngữ "general" hoặc "business", bỏ qua ngữ cảnh "training" cụ thể.

**Nguyên nhân tiềm năng:**
- MonoT5 chưa được huấn luyện nhiều về các truy vấn liên quan đến nhu cầu đào tạo, hoặc không nhận diện được "problems" trong ngữ cảnh này.

**Insight:**
- MonoT5 dễ bị "distract" bởi các doc có từ khóa phổ biến, không đủ sắc bén để nhận diện đúng doc liên quan đến "training" và "problem" trong IR.
- **Hybrid RR:** 1.000
- **MonoT5 RR:** 0.100
- **Suy Giảm:** -0.900
- **Số doc liên quan:** 24

**Top 5 Hybrid:**
1. Doc 648 ✓
2. Doc 839 ✗
3. Doc 1166 ✗
4. Doc 1405 ✗
5. Doc 471 ✓

**Top 5 MonoT5:**
1. Doc 175 ✗
2. Doc 1405 ✗
3. Doc 454 ✗
4. Doc 1012 ✗
5. Doc 459 ✗

---

---

### Trường hợp 4: Query 62
**Nội dung truy vấn:**
> This article concerns the problem of how to permit a patron to represent the relative importance of various index terms in a Boolean request while retaining the desirable properties of a Boolean system. The character of classical Boolean systems is reviewed and related to the notion of fuzzy sets. The fuzzy set concept then forms the basis of the concept of a fuzzy request in which weights are assigned to index terms. Ther properties of such a system are discussed, and it is shown that such systems retain the manipulability of traditional Boolean requests.

**Nhận xét:**
- Truy vấn về "fuzzy set", "Boolean request", "weights assigned to index terms".
- Hybrid dễ matching các doc có từ khóa "fuzzy", "Boolean", "weights".
- MonoT5 lại chọn doc không liên quan, có thể do không "hiểu" được khái niệm "fuzzy set" trong IR.

**Nguyên nhân tiềm năng:**
- MonoT5 không được fine-tune trên các truy vấn về lý thuyết tập mờ hoặc Boolean logic.
- Prompt "Relevant:" không đủ định hướng cho các truy vấn lý thuyết.

**Insight:**
- MonoT5 yếu ở các truy vấn cần matching khái niệm toán học/lý thuyết, đặc biệt là các chủ đề ít xuất hiện trong tập huấn luyện.
- **Hybrid RR:** 1.000
- **MonoT5 RR:** 0.100
- **Suy Giảm:** -0.900
- **Số doc liên quan:** 12

**Top 5 Hybrid:**
1. Doc 54 ✓
2. Doc 464 ✓
3. Doc 455 ✓
4. Doc 443 ✓
5. Doc 319 ✓

**Top 5 MonoT5:**
1. Doc 615 ✗
2. Doc 1137 ✗
3. Doc 454 ✗
4. Doc 1117 ✗
5. Doc 1119 ✗

---

---

### Trường hợp 5: Query 66
**Nội dung truy vấn:**
> Current online library network technology is described, including the physical and functional aspects of networks. Three types of networks are distinguished: search service (e.g., SDC, Lockheed), customized service that provide bibliographic files (e.g., OCLC, Inc., RLIN), and service center (e.g., NELINET, INCOLSA). It is predicted that as technology evolves more services will be provided outside the library directly to the user through his home or office.

**Nhận xét:**
- Truy vấn về "online library network technology", "types of networks", "service center".
- Hybrid tận dụng tốt các từ khóa "network", "service", "library" để tìm doc đúng.
- MonoT5 lại chọn doc không liên quan, có thể do mô hình không nhận diện được các khái niệm về "network technology" trong ngữ cảnh thư viện.

**Nguyên nhân tiềm năng:**
- MonoT5 thiếu trải nghiệm với các truy vấn về công nghệ mạng thư viện, vốn là chủ đề hẹp.

**Insight:**
- MonoT5 dễ bị "lạc đề" với các truy vấn về công nghệ chuyên ngành, đặc biệt khi các từ khóa có thể xuất hiện trong nhiều ngữ cảnh khác nhau.
- **Hybrid RR:** 1.000
- **MonoT5 RR:** 0.100
- **Suy Giảm:** -0.900
- **Số doc liên quan:** 35

**Top 5 Hybrid:**
1. Doc 947 ✓
2. Doc 885 ✓
3. Doc 1367 ✓
4. Doc 654 ✓
5. Doc 119 ✓

**Top 5 MonoT5:**
1. Doc 743 ✗
2. Doc 728 ✗
3. Doc 879 ✗
4. Doc 297 ✗
5. Doc 300 ✗

---


---

## 3. Tổng kết insight & khuyến nghị

### Mẫu hình thất bại của MonoT5
1. **Không "hiểu" truy vấn kỹ thuật/phức tạp:** MonoT5 gặp khó với các truy vấn nhiều khái niệm chuyên ngành, dài, hoặc có logic phức tạp.
2. **Thiên vị ngôn ngữ tự nhiên:** Mô hình ưu tiên doc có ngôn ngữ gần truy vấn, bỏ qua doc đúng về mặt kỹ thuật.
3. **Yếu ở matching toán học/lý thuyết:** Các chủ đề như "fuzzy set", "Boolean logic" thường bị bỏ qua.
4. **Chưa tối ưu prompt:** Prompt "Relevant:" không đủ định hướng cho các truy vấn đặc thù.
5. **Domain mismatch:** MonoT5 huấn luyện trên tập dữ liệu khác, không sát với IR khoa học/kỹ thuật.

### Đề xuất cải thiện
- **Fine-tune MonoT5** trên tập truy vấn/doc chuyên ngành IR, đặc biệt là các chủ đề kỹ thuật, toán học, thư viện.
- **Thử nghiệm prompt mới:** Ví dụ, "Is this document relevant to the following technical query?" hoặc prompt có ví dụ minh họa.
- **Kết hợp Cross-Encoder** cho các truy vấn khó, hoặc dùng ensemble để giảm rủi ro "hallucination" của MonoT5.
- **Phân tích lỗi định kỳ:** Lặp lại phân tích này sau mỗi lần fine-tune để phát hiện sớm các chủ đề mô hình còn yếu.

## Khuyến nghị
- Fine-tune MonoT5 trên dữ liệu chuyên ngành
- Thử nghiệm các template prompt khác nhau
- Ưu tiên Cross-Encoder nếu cần re-ranking ổn định hơn
