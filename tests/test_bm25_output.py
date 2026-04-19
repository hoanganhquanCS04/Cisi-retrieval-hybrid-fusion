import json
import unittest
from pathlib import Path

class TestBM25Output(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Thiết lập đường dẫn tương đối tới các file JSON
        project_root = Path(__file__).resolve().parents[1]
        cls.corpus_path = project_root / "data" / "corpus.json"
        cls.bm25_top100_path = project_root / "data" / "bm25_top100.json"
        
        # Load corpus và trích xuất danh sách doc_ids hợp lệ
        with open(cls.corpus_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        cls.valid_doc_ids = {int(doc["doc_id"]) for doc in corpus}
        
        # Load báo cáo kết quả BM25
        with open(cls.bm25_top100_path, "r", encoding="utf-8") as f:
            cls.bm25_results = json.load(f)

    def test_top100_format_and_types(self):
        """
        Chức năng của hàm:
            Kiểm tra định dạng kiểu dữ liệu đầu ra và số lượng kết quả xem có chuẩn JSON schema quy định và số lượng <= 100 không.
        
        Input:
            Không có tham số truyền vào trực tiếp. Hàm tự lấy dữ liệu từ `self.bm25_results` trong bộ nhớ class.
            
        Yêu cầu đầu vào:
            Giá trị của `self.bm25_results` phải được load thành công thành dạng Dictionary theo chuẩn sinh ra từ Task 2.9.
            
        Output:
            Không trả về gì. Nếu định dạng sai bất kỳ đoạn nào, thư viện unittest tự động kích hoạt ném Exception làm báo đỏ (Fail) test case.
        """
        for qid, results in self.bm25_results.items():
            self.assertTrue(isinstance(results, list), f"Value của query {qid} phải là list.")
            self.assertLessEqual(len(results), 100, f"Query {qid} có lớn hơn 100 kết quả ({len(results)}).")
            
            for item in results:
                self.assertIn("doc_id", item, f"Thiếu key 'doc_id' ở query {qid}")
                self.assertIn("score", item, f"Thiếu key 'score' ở query {qid}")
                self.assertTrue(isinstance(item["doc_id"], int), f"doc_id không phải int: {item['doc_id']}")
                self.assertTrue(isinstance(item["score"], float), f"score không phải float: {item['score']}")

    def test_doc_ids_in_corpus_range(self):
        """
        Chức năng của hàm:
            Đảm bảo không có dòng dữ liệu rác hay doc_id nào nằm ngoài phạm vi 1.460 tài liệu corpus gốc.
            
        Input:
            Sử dụng `self.bm25_results` (kết quả model) và `self.valid_doc_ids` (ID chuẩn lấy trực tiếp từ file sách gốc).
            
        Yêu cầu đầu vào:
            Danh sách ID hợp lệ (`valid_doc_ids`) phải là tập hợp (set) chứa kiểu số nguyên int để lệnh `not in` chạy được tối đa tốc độ.
            
        Output:
            Không trả về gì. Sẽ chủ động xuất ra lỗi Assertion nếu phát hiện tài liệu "ma" không thuộc thư viện sách nhưng lại chui vào báo cáo.
        """
        invalid_docs = []
        for qid, results in self.bm25_results.items():
            for item in results:
                doc_id = item["doc_id"]
                if doc_id not in self.valid_doc_ids:
                    invalid_docs.append((qid, doc_id))
        
        self.assertEqual(
            len(invalid_docs), 0, 
            f"Phát hiện tài liệu lạ lấy từ đâu ra không có trong corpus: {invalid_docs[:5]}..."
        )

    def test_scores_are_sorted_descending(self):
        """
        Chức năng của hàm:
            Đảm bảo BM25 list trả về tuân thủ hoàn toàn luật lệ xếp hạng điểm từ cao xuống thấp (Descending).
            
        Input:
            Rút trích các điểm số mảng (`scores`) từ file kết quả JSON.
            
        Yêu cầu đầu vào:
            Mảng điểm số rút ra buộc toàn bộ phải mang kiểu số thực thập phân `float`.
            
        Output:
            Không trả về. Ném vỡ luồng với Exception nếu mảng điểm tồn tại số bé lại bị đặt đứng trước số lớn.
        """
        for qid, results in self.bm25_results.items():
            scores = [item["score"] for item in results]
            expected_scores = sorted(scores, reverse=True)
            self.assertEqual(
                scores, expected_scores, 
                f"Kết quả của query {qid} không được xếp hạng từ cao xuống thấp đúng cách."
            )

if __name__ == '__main__':
    unittest.main()
