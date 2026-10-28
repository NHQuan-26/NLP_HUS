# Dự đoán trước thí nghiệm

1. **Dot product lớn hơn:** trọng số attention cao hơn. Lý do: softmax giữ nguyên thứ tự score.
2. **Các score bằng nhau:** trọng số chia đều. Lý do: các giá trị sau hàm mũ bằng nhau.
3. **Một score vượt trội:** attention tập trung vào vị trí đó. Lý do: giá trị sau hàm mũ của nó chiếm ưu thế.
4. **Từ 10 lên 1000 token:** chi phí full self-attention tăng mạnh. Lý do: kích thước ma trận score tăng theo bình phương độ dài.
5. **Token con người thấy quan trọng:** attention không phải lúc nào cũng chọn đúng token đó. Lý do: trọng số phụ thuộc vào query và mô hình đã học.
