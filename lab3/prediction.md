# DỰ ĐOÁN TRƯỚC THỰC NGHIỆM (PREDICTION BEFORE EXPERIMENT) — LAB 03

Môn học: Xử lý ngôn ngữ tự nhiên và ứng dụng  
Học kỳ: Học kỳ I - 2026  
Sinh viên: Nguyễn Hồng Quân - 23001921  
Giảng viên / TA: Phạm Ngọc Hải  
Chủ đề: Word Representations and Embeddings

---

Theo yêu cầu tại Mục 9 của tài liệu `W3.pdf`, trước khi chạy code thực nghiệm, em đưa ra các dự đoán cá nhân như sau:

## Prediction 1 — Những từ nào gần nhau nhất?

- **Câu hỏi:** Trong các từ `doctor`, `physician`, `hospital`, `banana`, `car`, những từ nào sẽ gần nhau nhất?
- **Dự đoán:** `doctor` và `physician` sẽ gần nhau nhất. Từ `hospital` cũng sẽ tương đối gần với hai từ này, còn `car` và `banana` sẽ ở xa hẳn nhóm từ y tế.
- **Lý do:** Vì `doctor` và `physician` đều là từ chỉ bác sĩ, gần như đồng nghĩa nên thường đi chung với các từ ngữ cảnh tương tự trong bài viết y tế. `hospital` là bệnh viện nên có liên quan mật thiết đến bác sĩ. Còn `banana` (hoa quả) và `car` (phương tiện) thuộc các chủ đề hoàn toàn khác.

---

## Prediction 2 — Tăng context window từ 2 lên 5 có làm thay đổi similarity không?

- **Câu hỏi:** Nếu context window tăng từ 2 lên 5 thì similarity có thay đổi không?
- **Dự đoán:** Có thay đổi, nhưng không phải cặp từ nào cũng tăng hoặc giảm giống nhau.
- **Lý do:** Khi cửa sổ mở rộng từ 2 lên 5, mô hình sẽ gom thêm các từ ở xa hơn vào ngữ cảnh. Việc này giúp bắt được các từ liên quan về mặt chủ đề rộng (như `doctor` và `hospital`), nhưng đồng thời cũng làm lọt thêm nhiều từ không liên quan vào ngữ cảnh, có thể làm giảm nhẹ độ tương đồng của những cặp từ đồng nghĩa thay thế trực tiếp (như `doctor` và `physician`).

---

## Prediction 3 — Tăng embedding dimension từ 50 lên 100 rồi 300 có chắc chắn cải thiện chất lượng không?

- **Câu hỏi:** Nếu tăng số chiều vector từ 50 lên 100 rồi 300 thì chất lượng có chắc chắn tăng không?
- **Dự đoán:** Không chắc chắn tăng.
- **Lý do:** Nhiều chiều hơn thì vector chứa được nhiều thông tin hơn, nhưng cũng đòi hỏi lượng dữ liệu huấn luyện phải đủ lớn và tốn nhiều thời gian tính toán hơn. Nếu tập dữ liệu mẫu của mình nhỏ (10.000 câu), tăng lên 300 chiều có thể làm mô hình học phải nhiễu và chất lượng tính similarity có khi còn bị giảm đi so với 50 hay 100 chiều.

---

## Prediction 4 — `doctor` và `physician` có chắc chắn gần nhau nếu corpus chỉ có 100 câu không?

- **Câu hỏi:** Hai từ `doctor` và `physician` có chắc chắn gần nhau nếu corpus chỉ có 100 câu không?
- **Dự đoán:** Không chắc chắn gần nhau.
- **Lý do:** Với chỉ 100 câu thì dữ liệu quá ít. Có thể từ `physician` chỉ xuất hiện 1-2 lần hoặc thậm chí không xuất hiện lần nào trong cùng ngữ cảnh với `doctor`. Khi không có đủ số lần đồng xuất hiện trong dữ liệu thì mô hình không thể học được hai từ này có nghĩa tương đương.
