# Reflection

- TF-IDF + LR đạt F1=0.8505, cao hơn các sequence model trong lần thử này và train nhanh hơn. Tuy nhiên, baseline đọc toàn review còn sequence model bị cắt context; chỉ chạy 6.000 mẫu, 3 epochs nên chưa thể kết luận kiến trúc nào luôn tốt hơn.

- Ở length=100, GRU có accuracy cao nhất trong ba recurrent model (0.5922), nhưng RNN có F1 nhỉnh hơn (0.5575 so với 0.5552). LSTM có recall thấp (0.3969), nên chỉ nhìn accuracy là chưa đủ.

- Tăng RNN từ 50 lên 200 token giữ được 21.22% → 68.60% token, nhưng thời gian tăng 2.30 → 11.28 giây và F1 giảm nhẹ 0.5665 → 0.5552. Thêm context không tự động cải thiện kết quả.

- Khi shuffle, xác suất TF-IDF unigram không đổi; các recurrent model có thay đổi. Tuy nhiên, F1 của LSTM và GRU còn tăng sau shuffle, nên độ nhạy với thứ tự chưa đồng nghĩa với sử dụng thứ tự hiệu quả.

- Với RNN length=100, train loss giảm qua ba epochs nhưng validation loss tăng sau epoch đầu, cho thấy dấu hiệu overfitting. LSTM vẫn giảm validation loss trong cả ba epochs.

- Probe keyword ở cuối tạo chênh lệch xác suất rõ hơn ở đầu/giữa. Đây chỉ là quan sát trên câu nhân tạo, chưa chứng minh vanishing gradient. Trong error analysis, mẫu 3 và 5 mất lời khen do truncation; mẫu 4 vẫn giữ lời chê nhưng dự đoán sai nên cần kiểm tra thêm.

- So với Average Word2Vec, recurrent models giữ state theo thứ tự thay vì lấy trung bình embedding. Đổi lại, chúng xử lý tuần tự, tốn chi phí và vẫn khó giữ thông tin xa. Attention giúp truy cập trực tiếp context ở các vị trí khác.
