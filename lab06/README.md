# LAB 06 — Cơ chế attention

Mở `self_attention.ipynb` bằng kernel `HUS_NLP` và chạy các cell theo thứ tự. Các thí nghiệm, huấn luyện và hình đều nằm trong notebook. Dữ liệu được đọc từ `../lab05/IMDB_Dataset.csv`.

Notebook dùng 3000 mẫu train, 500 validation, 500 test; tối đa 80 token; 3 epoch. Đây là một lần thử nhỏ, chưa đủ để kết luận mô hình nào luôn tốt hơn. Các bài tính tay và `calculations.md` được bỏ theo yêu cầu.

## Ý tưởng chính

Query (Q) biểu diễn thông tin cần tìm; Key (K) dùng để so khớp với query; Value (V) chứa thông tin để tạo context. `QKᵀ` tạo score, softmax theo từng hàng tạo trọng số, rồi `AV` tạo context. Chia score cho `√dₖ` giúp softmax bớt tập trung quá mức khi số chiều lớn. Trừ giá trị lớn nhất trước khi lấy hàm mũ giúp tránh tràn số và không đổi kết quả softmax.

Trong self-attention, Q, K và V đều được tạo từ cùng một chuỗi. Ma trận score có kích thước tăng theo bình phương độ dài chuỗi. Mô hình trong lab thêm embedding vị trí để có thông tin về thứ tự token.

## Đọc heatmap

Trong `attention_heatmap.png`, hàng là token đang chú ý, cột là token được chú ý. Mỗi ô trong **cùng một ma trận** có kích thước bằng nhau; màu xanh đậm hơn nghĩa là trọng số attention lớn hơn. Hai câu đặt cạnh nhau có thể làm kích thước hiển thị khác nhau nếu số token khác nhau hoặc ảnh bị co giãn, nhưng kích thước ô không biểu diễn trọng số. Mỗi hàng được chuẩn hóa sao cho tổng trọng số bằng 1.

Ví dụ, ở câu đầu, hàng `adopted` có ô tại cột `cat` tương đối đậm (trọng số khoảng 0,286). Các vector ở hình này được **gán thủ công** để minh họa, chưa qua huấn luyện; hình không chứng minh mô hình hiểu ngữ pháp.

## Tệp chính

- `attention_numpy.py`: cài đặt attention bằng NumPy.
- `self_attention.ipynb`: toàn bộ thí nghiệm, so sánh mô hình và vẽ hình.
- `prediction.md`, `error_analysis.md`, `reflection.md`: phần viết ngắn.
- `results.csv`: kết quả so sánh mô hình; notebook lưu tệp này.
- `attention_heatmap.png`, `confusion_matrix.png`: hai hình được lưu theo deliverables. Các bảng và hình còn lại chỉ hiển thị trong notebook.

## Khai báo hỗ trợ AI

- **Công cụ:** OpenAI Codex.
- **Mục đích:** Hỗ trợ viết code, tài liệu, trực quan hóa và thí nghiệm.
- **Nội dung tạo:** Code NumPy/PyTorch, notebook, hình và bản nháp phân tích.
- **Nội dung chỉnh sửa:** Sửa LSTM để lấy hidden state tại token thật cuối cùng thay vì phần padding.
- **Kiểm chứng:** Đã kiểm tra cú pháp các cell của notebook sau khi chuyển mã. Các kết quả CSV hiện có thuộc lần chạy trước; sinh viên sẽ chạy notebook để tạo lại và tự xem lại các nhận xét trước khi nộp.
