# Nhận xét

- Với query về cảm xúc, “good” nhận trọng số cao nhất; với query về chủ đề, “movie” nổi bật hơn. Attention thay đổi theo query.
- Khi temperature là 0,5; 1; 2, trọng số lớn nhất lần lượt là 0,982; 0,867; 0,665. Temperature cao làm phân phối phẳng hơn. Scaling cũng làm phân phối phẳng hơn trong ví dụ này.
- Heatmap minh họa cho “cat”, “adopted”, “sleeping” có các vùng màu đậm ở những token được gán đặc trưng giống nhau. Mỗi ô có kích thước như nhau; **độ đậm của màu**, không phải diện tích ô, thể hiện trọng số. Vector được gán thủ công nên không thể xem đây là bằng chứng mô hình hiểu ngữ pháp.
- Trên tập IMDB nhỏ, Attention đạt accuracy 0,676 và F1 0,727; LSTM đạt 0,530 và 0,591. LSTM dùng hidden state cuối, còn Attention kết hợp thông tin trực tiếp từ nhiều token. Kết quả có thể đổi khi thay dữ liệu hoặc cấu hình huấn luyện.
- Attention kết nối trực tiếp các vị trí xa và không cần tính tuần tự như RNN. Đổi lại, full self-attention cần ma trận N × N, nên chi phí bộ nhớ và tính toán tăng theo `O(N²)`. Trọng số attention cũng không chứng minh nguyên nhân của dự đoán.
- Lab này chỉ dùng một tầng attention. Transformer còn có nhiều head và các thành phần xử lý vị trí; đó là bước tiếp theo.
