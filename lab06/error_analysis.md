# Phân tích lỗi

Sáu ví dụ lấy từ test set trong `examples.csv`: ba dự đoán đúng và ba dự đoán sai. “Token nổi bật” là token có **trọng số attention trung bình** lớn nhất qua các query thật. Review dài hơn 80 token bị cắt trước khi dự đoán. Những nhận xét dưới đây chỉ là giả thuyết cần kiểm tra thêm.

| Mẫu | Nhãn thật → dự đoán | P(positive) | Token nổi bật | Nhận xét |
|---|---|---:|---|---|
| 1 | negative → negative | 0,460 | makes | Dự đoán đúng nhưng sát ngưỡng 0,5. Review bị cắt; một token không đủ giải thích kết quả. |
| 2 | positive → positive | 0,915 | and | Review có lời khen như “good”, “best”, nhưng token nổi bật là từ nối. Có thể mô hình sử dụng cả ngữ cảnh. |
| 3 | positive → positive | 0,764 | who | Review bị cắt và phần đầu có từ tích cực. Token nổi bật không phải từ chỉ cảm xúc rõ ràng. |
| 4 | negative → positive | 0,759 | some | Phần đầu vừa khen diễn viên vừa chê phim; cách diễn đạt pha trộn có thể góp phần gây lỗi. Truncation có thể làm mất lời chê phía sau. |
| 5 | negative → positive | 0,696 | avigdor's | Đầu review có “zero rating”, nhưng token nổi bật là tên riêng hiếm. Biểu diễn từ hiếm yếu là một khả năng, chưa được chứng minh. |
| 6 | negative → positive | 0,673 | why | Đầu review chê rõ (“trilogy of trash”). Việc chú ý đến “why” tự nó không giải thích được lỗi. |

Ở câu đối lập “bad, but ... excellent”, mô hình cũng dự đoán sai: P(positive)=0,015 và token nổi bật là “bad”. Điều này phù hợp với giả thuyết mô hình xử lý tương phản chưa tốt; cần thử bỏ hoặc thay từng token để kiểm tra. Câu phủ định ngắn “I do not like this movie” lại được dự đoán đúng, nên chưa thể kết luận mô hình luôn xử lý phủ định kém.
