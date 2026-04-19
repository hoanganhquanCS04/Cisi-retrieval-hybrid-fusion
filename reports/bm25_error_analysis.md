# Phân Tích Lỗi BM25 (Error Analysis)

**Người thực hiện:** Member 1 (Data Engineer & BM25 Architect)

Mặc dù mô hình BM25 đóng vai trò là một Baseline rất mạnhtrọng Information Retrieval, kết quả chạy thực tế trên tệp 112 câu hỏi của tập CISI cho thấy có những câu hỏi bị đánh điểm Reciprocal Rank (RR) rất tệ (gần bằng 0). Điều này có nghĩa là tài liệu chứa đáp án chuẩn bị đẩy xuống tận vị trí thứ 50, 60 hoặc thậm chí rớt khỏi Top 100.

Dưới đây là thống kê 10 Queries có điểm MRR thấp nhất và phân tích nguyên nhân học thuật dẫn đến sự thất bại của cơ chế dựa trên từ khóa thô (Sparse Retrieval).

## 1. Danh sách 10 Queries "đội sổ" của BM25

| Hạng Xấu | Query ID | Điểm RR | Nội dung truy vấn (Query) | Document chuẩn (Ground Truth) nhưng bị rớt hạng |
|:---:|:---:|:---:|---|---|
| 1 | 14 | 0.0141 | What future is there for automatic medical diagnosis? | Doc 890: ...application of pattern recognition and substructural analysis to the problem of predicting the antineoplastic activity... |
| 2 | 2 | 0.0167 | How can actually pertinent data... be retrieved automatically in response to information requests? | Doc 669: The Wiswesser chemical line-notation is an unique and unambiguous method of representing chemical structures... |
| 3 | 6 | 0.0312 | What possibilities are there for verbal communication between computers and humans... via the spoken word? | Doc 400: Communication technology has entered a period of revolutionary change. The last decade has brought new inventions... |
| 4 | 56 | 0.0526 | ...standard method of finding information... alphabetically arranged card catalog... modified for automated information retrieval? | Doc 1280: Machine indexing and text searching offer an approach to the basic problems of library automation. |
| 5 | 96 | 0.0526 | ...analyzed recent developments in... queries expressed as Boolean expressions... fuzzy-set-theoretic considerations. | Doc 449: The concept of situational relevance is introduced, based on W.S.Cooper's definitions of logical relevance... |
| 6 | 43 | 0.0556 | ...failure to plan adequately for document analysis, indexing, and machine coding... effectiveness of programming? | Doc 128: The co-joining of "design" with "evaluation" that is called for by this chapter posed organizational... |
| 7 | 84 | 0.0588 | A technique is described for automatic reformulation of boolean queries... Results compare favourably with feedback... | Doc 449: The concept of situational relevance is introduced... on the notions of a person... |
| 8 | 8 | 0.0833 | Describe information retrieval and indexing in other languages. What bearing does it have on the science in general? | Doc 864: A new processing format, based on MARC II and some of BNB's elaborations of MARC II... encompass French cataloging... |
| 9 | 7 | 0.1000 | Describe presently working and planned systems for publishing and printing original papers by computer... | Doc 320: The Teachable Language Comprehender (TLC) is a program designed to be capable of being taught to "comprehend" English text... |
| 10 | 61 | 0.1000 | The way that individuals construct and modify search queries... is subject to systematic biases... | Doc 642: On-line interactive searching of several information bases through several service operators... |

## 2. Phân tích Nguyên Nhân Thất Bại

Dựa vào việc đối chiếu trực tiếp giữa câu hỏi (Query) và đoạn văn bản tóm tắt của sách chuẩn (Doc Truth), chúng ta có thể kết luận điểm yếu chí mạng của BM25 nằm ở **Lexical Gap** (Khoảng trống từ vựng) và **Vocabulary Mismatch** (Lệch pha từ vựng).

### 2.1. Lệch pha từ vựng (Vocabulary Mismatch)
Thuật toán BM25 kiếm điểm nhờ vào việc **đếm số lần trùng lặp chính xác (exact match)** của một từ khóa giữa câu hỏi và văn bản. 
- **Ví dụ Query 14**: Người dùng hỏi về `"automatic medical diagnosis"` (chuẩn đoán y khoa tự động). Tuy nhiên, tài liệu y khoa số 890 lại dùng các thuật ngữ hàn lâm như `"pattern recognition"` (nhận diện mẫu), `"predicting antineoplastic activity"` (dự đoán hoạt động kháng khối u), `"mouse brain tumor"` (khối u não chuột). 
- Sự vắng mặt hoàn toàn của các chữ "medical" hay "diagnosis" trong đoạn đầu của tệp Truth Doc khiến BM25 mù tịt và cho 0 điểm tệp tài liệu này, dù về mặt ngữ nghĩa (Semantic), chúng hoàn toàn khớp với ngữ cảnh y khoa.

### 2.2. Xung đột từ đồng nghĩa (Synonymy)
- **Ví dụ Query 56**: Câu hỏi dùng từ `"automated information retrieval"`, nhưng tác giả tài liệu số 1280 lại viết theo kiểu `"machine indexing and text searching"` và `"library automation"`. 
- Bằng mắt người, ta hiểu "machine" tương đồng với "automated", và "text searching" tương đồng với "information retrieval". Nhưng đối với BM25, đây là 4 từ hoàn toàn khác biệt. Hệ quả là nó sẽ nhường Top 1 cho một bài báo rác ngẫu nhiên lặp đi lặp lại từ "information".

### 2.3. Cấu trúc câu quá dài chứa từ nhiễu
Nhiều queries trong CISI là một đoạn văn miêu tả (như query 56, 84, 96) chứa đựng một lượng lớn các từ vựng kể chuyện (`purpose of this paper`, `several papers have appeared`, `results compare`). 
Mặc dù đã có Stopword Removal, sự dư thừa của các danh từ chung này làm BM25 đánh điểm sai đối tượng, kéo theo các tài liệu có bối cảnh văn phong giống vậy lên trên, thay vì tập trung vào Keyword chuyên môn cốt lõi.

## 3. Kiến nghị hướng giải quyết (Cho Member 2 & 3)
Chính sự bất lực của Sparse Retrieval (BM25) trước **từ đồng nghĩa** và **sự hiểu ngữ cảnh** là động lực bắt buộc hệ thống phải tích hợp thêm thuật toán **Dense Retrieval**. 
Bằng cách sử dụng các mô hình Neural Networks như **MiniLM** (Phase 2 - M2) dựa trên Vector Embeddings, hệ thống máy tính sẽ biết cách dịch "machine searching" và "automated retrieval" về cùng một cụm tọa độ không gian điểm, qua đó trục vớt thành công 10 tài liệu bị chìm nghỉm bên trên lên lại Top đầu!
