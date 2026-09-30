# DỰ ĐOÁN TRƯỚC THỰC NGHIỆM (PART C)

Môn học: Xử lý ngôn ngữ tự nhiên và ứng dụng  
Sinh viên: Nguyễn Hồng Quân - 23001921

---

Trước khi chạy code trên tập dữ liệu 30.000 documents của C4, em đưa ra 3 dự đoán cá nhân như sau:

## Prediction 1 — Kích thước từ vựng (Vocabulary)

- **Câu hỏi:** Với corpus 30K documents, vocabulary sẽ có khoảng bao nhiêu từ độc nhất (unique terms)?
- **Dự đoán của em:** Khoảng **150.000 đến 200.000 từ**.
- **Giải thích:** Tập C4 là dữ liệu cào từ trang web nên rất lộn xộn, có nhiều từ lóng, tên riêng, số hiệu và cả từ viết sai chính tả. Theo quy luật thông thường trong ngôn ngữ (Heaps' law), khi số văn bản tăng lên thì số từ mới xuất hiện cũng tăng rất nhanh, nên từ vựng chắc chắn sẽ rất lớn, vượt xa mức 100.000 từ.

---

## Prediction 2 — Độ thưa của ma trận (Sparsity)

- **Câu hỏi:** Ma trận TF-IDF sẽ dày (dense) hay thưa (sparse)? Tỷ lệ số 0 lớn đến mức nào?
- **Dự đoán của em:** Ma trận sẽ **cực kỳ thưa (sparse)**, tỷ lệ phần tử bằng 0 sẽ lên tới **hơn 99.8%**.
- **Giải thích:** Một văn bản bình thường chỉ dài tầm 100 đến 200 từ khác nhau. Trong khi đó, vector đại diện cho văn bản lại phải có số chiều bằng toàn bộ từ vựng (khoảng gần 200.000 cột). Do đó trong một hàng, gần như tất cả các vị trí đều bằng 0, chỉ có một số ít vị trí có giá trị khác 0.

---

## Prediction 3 — Chất lượng tìm kiếm (Search Quality)

- **Câu hỏi:** Các documents đứng đầu kết quả tìm kiếm có nhất thiết là documents gần nghĩa nhất không?
- **Dự đoán của em:** **Không nhất thiết.**
- **Giải thích:** Vì TF-IDF chỉ so khớp xem các từ có viết giống hệt nhau không (lexical matching) chứ không hiểu nghĩa của từ. Ví dụ nếu em tìm "heart attack" mà văn bản lại viết là "myocardial infarction" (đều là nhồi máu cơ tim) thì TF-IDF sẽ cho điểm bằng 0 vì không có từ nào trùng chữ. Hoặc nếu từ bị đa nghĩa (như "apple" vừa là quả táo vừa là hãng công nghệ), nó có thể tìm ra bài viết không liên quan đến ý định người tìm.
