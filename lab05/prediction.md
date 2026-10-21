# Prediction

1. **Word order:** RNN xử lý trực tiếp hơn TF-IDF unigram vì hidden state được cập nhật theo thứ tự token.

2. **Dependency dài:** LSTM có lợi thế hơn RNN nhờ cell state và gates kiểm soát việc giữ/bỏ thông tin.

3. **Length 50 → 200:** training có thể chậm hơn vì cần xử lý nhiều timestep; giữ thêm context nhưng chưa chắc tăng F1.

4. **Truncate 30 tokens:** có thể mất lời kết luận, đoạn chuyển ý hoặc từ quyết định sentiment ở cuối review.

5. **LSTM vs RNN:** LSTM không chắc luôn có accuracy cao hơn; kết quả còn phụ thuộc dữ liệu, cấu hình và thời gian train.
