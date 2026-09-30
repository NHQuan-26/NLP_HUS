# TỔNG KẾT VÀ TỰ ĐÁNH GIÁ (REFLECTION & LEARNING CHECK)

Môn học: Xử lý ngôn ngữ tự nhiên và ứng dụng  
Sinh viên: Nguyễn Hồng Quân - 23001921

---

## 1. Phần tự đánh giá (Reflection)

**1. Prediction nào của em sai?**  
Dự đoán của em về độ thưa ma trận (~99.8%) và từ vựng (~150K - 200K) khá gần với kết quả chạy thật (V = 193.540 từ, độ thưa = 99.91%). Tuy nhiên, ở Pipeline A (không tiền xử lý gì ngoài tách từ theo khoảng trắng), từ vựng thực tế vọt lên tận **473.388 từ** do dính quá nhiều dấu câu và URL lộn xộn, cao hơn nhiều so với dự đoán ban đầu của em.

**2. Kết quả nào bất ngờ nhất?**  
Kết quả bất ngờ nhất là khi em tìm kiếm cụm từ `"transformer language model"`. Em nghĩ nó sẽ ra các bài viết về AI hoặc mô hình Transformer, nhưng tài liệu đứng đầu lại là một bài viết hướng dẫn sửa chữa mạch điện của xe mô hình (Doc 27936), vì trong bài đó có cả từ "transformer" (máy biến áp) và "model" (mô hình xe).

**3. Experiment nào cung cấp evidence mạnh nhất?**  
Thực nghiệm so sánh tiền xử lý (Part F - Preprocessing Ablation) cho thấy bằng chứng rõ ràng nhất. Chỉ cần chuyển chữ thường, bỏ dấu câu và bỏ stopwords (Pipeline B), từ vựng đã giảm mạnh từ hơn 473K xuống còn 167K từ (giảm hơn 64%), giúp tiết kiệm rất nhiều bộ nhớ và loại bỏ các từ vô nghĩa.

**4. Failure case quan trọng nhất là gì?**  
Đó là sự khác biệt giữa hai cụm từ y tế `"heart attack prevention"` và `"myocardial infarction therapy"`. Cả hai đều nói về bệnh nhồi máu cơ tim, nhưng vì viết chữ khác nhau nên TF-IDF tính ra độ tương đồng bằng 0 và không tìm được tài liệu.

**5. Nếu được xây lại search engine, em sẽ thay đổi điều gì?**  
Em sẽ không chỉ dùng mỗi TF-IDF. Em sẽ kết hợp TF-IDF với mô hình Dense Embedding (như Sentence-BERT) để vừa tìm đúng từ khóa chính xác, vừa hiểu được ngữ nghĩa của câu khi người dùng dùng từ đồng nghĩa.

**6. Khai báo sử dụng AI:**

- AI hỗ trợ em viết khung code các phép tính ma trận trong `implementation.py` và giải thích công thức làm mịn IDF của thư viện scikit-learn.
- Toàn bộ các bài tính tay trong `calculations.md`, phần đưa ra dự đoán ban đầu, chọn ví dụ phân tích lỗi và trả lời các câu hỏi tự kiểm tra đều do em tự làm và kiểm chứng.

---

## 2. Trả lời câu hỏi ôn tập (Learning Check)

**Câu 1: Tại sao TF-IDF tạo ra sparse representation?**  
Vì toàn bộ kho ngữ liệu có tới hàng trăm nghìn từ khác nhau, nhưng mỗi văn bản thực tế chỉ chứa khoảng 100 đến 200 từ. Khi biểu diễn văn bản thành vector có chiều dài bằng toàn bộ từ vựng, những từ không có mặt đều nhận giá trị 0, khiến cho hơn 99% các giá trị trong vector là số 0.

**Câu 2: Tại sao một term xuất hiện trong hầu hết documents có IDF thấp?**  
Theo công thức IDF = ln(N / df), nếu từ xuất hiện ở hầu hết văn bản thì df xấp xỉ N, khi đó tỷ số N / df xấp xỉ 1 và ln(1) = 0. Về mặt ý nghĩa, từ nào câu nào cũng có (như _the, is, and_) thì không mang giá trị để phân biệt nội dung các câu với nhau.

**Câu 3: Tại sao một term có IDF cao chưa chắc có TF-IDF cao trong một document?**  
Vì công thức TF-IDF = TF \* IDF. Một từ dù có IDF rất cao (từ rất hiếm), nhưng nếu nó không xuất hiện trong văn bản đang xét thì TF của nó bằng 0, dẫn đến TF-IDF cũng bằng 0.

**Câu 4: Tại sao cosine similarity phù hợp với document vectors?**  
Vì cosine similarity đo góc giữa 2 vector chứ không bị ảnh hưởng bởi độ dài của văn bản. Nếu một bài viết dài nói cùng chủ đề với một bài viết ngắn, phép đo này vẫn nhận diện được chúng giống nhau, không bị lệch điểm do bài viết dài có nhiều từ lặp lại hơn.

**Câu 5: Tại sao preprocessing có thể thay đổi search result?**  
Vì tiền xử lý quyết định xem các từ có khớp nhau hay không. Ví dụ nếu không chuyển về chữ thường thì từ "Medical" viết hoa đầu câu sẽ không trùng với từ "medical" viết thường trong ô tìm kiếm.

**Câu 6: Một failure case của TF-IDF search mà em quan sát được là gì?**  
Khi tìm kiếm `"transformer language model"`, hệ thống trả về bài viết về máy biến áp và mô hình xe hơi vì bị nhầm từ đa nghĩa (polysemy).

**Câu 7: Failure case đó gợi ý nhu cầu về representation nào tiếp theo?**  
Nó gợi ý rằng ta cần cách biểu diễn từ theo ngữ cảnh (Contextual Embeddings, ví dụ như mô hình BERT), để từ "transformer" khi đi cùng "language" sẽ có vector mang nghĩa trí tuệ nhân tạo, khác với khi nó đi cùng từ "circuit board".
