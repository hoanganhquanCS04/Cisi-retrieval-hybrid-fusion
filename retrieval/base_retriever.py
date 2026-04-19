from abc import ABC, abstractmethod

class BaseRetriever(ABC):
    """
    Abstract Base Class (Khuôn mẫu) cho tất cả các mô hình truy hồi trong dự án.
    Tất cả các mô hình như BM25, Dense, Hybrid đều phải kế thừa class này 
    và bắt buộc phải định nghĩa hàm `retrieve`.
    """

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 100) -> list[tuple[int, float]]:
        """
        Chức năng của hàm:
            Thực hiện tìm kiếm top-k tài liệu tốt nhất dựa trên văn bản truy vấn.
            Đây là hàm trừu tượng (khoán trắng), nên mô hình con khi nhúng vào sẽ phải tự viết logic quét tệp.

        Input:
            - query (str): Nội dung câu hỏi/truy vấn của người dùng.
            - top_k (int, optional): Số lượng tài liệu tối đa muốn tải về. Mặc định là 100.

        Yêu cầu đầu vào:
            - `query` phải là chuỗi hợp lệ không bị rỗng, thường đã trải qua bước chuẩn hóa cơ bản ở lớp API.
            - `top_k` phải là số nguyên (int) > 0. 

        Output:
            - Dạng trả về là danh sách kết quả được nối theo từng cặp: list[tuple[int, float]].
            - Trả về danh sách được xếp hạng từ điểm cao xuống điểm thấp.
            - Ví dụ output thực tế: [(25, 14.5), (102, 12.3), (1, 10.0), ...]
        """
        pass
