# TỔNG KẾT VÀ TỰ ĐÁNH GIÁ (REFLECTION & LEARNING CHECK) — LAB 03

Môn học: Xử lý ngôn ngữ tự nhiên và ứng dụng  
Học kỳ: Học kỳ I - 2026  
Sinh viên: Nguyễn Hồng Quân - 23001921  
Giảng viên / TA: Phạm Ngọc Hải  
Chủ đề: Word Representations and Embeddings

---

## 1. Bảng so sánh các phương pháp biểu diễn từ (Mục 27)

| Representation                              | Context-dependent? | Sparse/Dense | Một từ có nhiều vector? |
| :------------------------------------------ | :----------------: | :----------: | :---------------------: |
| **TF-IDF (Lab 01)**                         |       Không        |    Sparse    |          Không          |
| **Co-occurrence Matrix (Lab 03)**           |       Không        |    Sparse    |          Không          |
| **Word2Vec (Lab 03)**                       |       Không        |    Dense     |          Không          |
| **Contextual Embedding (BERT/Transformer)** |         Có         |    Dense     |           Có            |

### Giải thích ngắn gọn:

- **TF-IDF:** Là vector thưa (Sparse) có kích thước bằng cả từ điển. Nó chỉ tính tần suất xuất hiện thống kê trong văn bản chứ không thay đổi theo ngữ cảnh của câu.
- **Co-occurrence Matrix:** Là ma trận đếm số lần hai từ đi cạnh nhau trong cửa sổ ngữ cảnh. Kích thước ma trận rất lớn và thưa (hơn 98% là số 0), mỗi từ chỉ có một hàng vector đếm cố định.
- **Word2Vec:** Đã nén biểu diễn xuống vector dày đặc (Dense, ví dụ 50, 100, 300 chiều). Tuy nhiên, sau khi huấn luyện xong thì mỗi từ chỉ được gán đúng một vector tĩnh duy nhất trong bảng tra từ điển.
- **Contextual Embedding:** Mô hình sinh vector động dựa trên ngữ cảnh xung quanh thông qua cơ chế Attention, nên cùng một từ khi đứng ở các câu có nghĩa khác nhau sẽ có các vector khác nhau.

---

## 2. Tại sao từ `bank` cần contextual representation? (Mục 26 & 27)

Xét hai câu ví dụ:

1. _"I deposited money in the bank."_ (bank mang nghĩa tổ chức tài chính / ngân hàng).
2. _"We sat on the river bank."_ (bank mang nghĩa bờ sông).

### Giới hạn của Word2Vec tĩnh:

- Từ `bank` là từ đa nghĩa, nhưng mô hình tĩnh như Word2Vec chỉ cấp duy nhất một vector $v_{\text{bank}}$ cho từ này.
- Khi huấn luyện, vector của `bank` vừa bị kéo về nhóm từ tài chính (`money`, `deposit`), vừa bị kéo về nhóm từ sông nước (`river`, `water`). Kết quả là vector của nó nằm lơ lửng ở giữa hai nghĩa, không phản ánh chính xác nghĩa nào cả.
- Khi người dùng tìm kiếm cụm từ liên quan đến ngân hàng, hệ thống có thể bị lẫn sang các bài viết về bờ sông và ngược lại.

Vì vậy, ta cần biểu diễn phụ thuộc ngữ cảnh (Contextual Representation) để mô hình dựa vào các từ xung quanh trong từng câu cụ thể mà sinh ra đúng vector tương ứng với nghĩa đó.

---

## 3. Bản khai báo sử dụng AI (AI Assistance Statement — Mục 28)

- **Tool:** Google Antigravity Assistant.
- **Mục đích sử dụng:**
    - Hỗ trợ cú pháp tối ưu hóa ma trận thưa trong `cooccurrence.py` và cách gọi thư viện Word2Vec trong gensim.
- **Nội dung do AI hỗ trợ:**
    - Gợi ý cấu trúc tính dot product nhanh giữa vector và ma trận thưa `csr_matrix`.
- **Nội dung sinh viên tự làm và chỉnh sửa:**
    - Tự viết lại cấu trúc code và đặt tên hàm/biến theo phong cách riêng trong `cooccurrence.py`.
    - Tự đưa ra các dự đoán trước thực nghiệm trong `prediction.md`.
    - Tự phân tích các trường hợp đúng, sai và rút ra nguyên nhân trong `error_analysis.md`.

---

## 4. Trả lời câu hỏi kiểm tra miệng (Individual Learning Check — Mục 29)

- **Câu 1: Distributional hypothesis là gì?**  
  _Trả lời:_ Là giả thuyết cho rằng những từ xuất hiện trong các ngữ cảnh tương tự nhau thì thường có ý nghĩa tương tự nhau ("You shall know a word by the company it keeps").

- **Câu 2: Tại sao doctor và physician có thể gần nhau?**  
  _Trả lời:_ Vì hai từ này cùng chỉ bác sĩ, thường xuyên xuất hiện chung với các từ ngữ cảnh y tế như `hospital`, `patient`, `treatment`, `medicine`, nên mô hình học được vector của chúng nằm gần nhau.

- **Câu 3: CBOW khác Skip-gram ở đâu?**  
  _Trả lời:_ CBOW dùng các từ ngữ cảnh xung quanh để dự đoán từ mục tiêu ở giữa. Ngược lại, Skip-gram lấy từ mục tiêu ở giữa để dự đoán từng từ ngữ cảnh xung quanh.

- **Câu 4: Tại sao tăng context window có thể vừa tốt vừa xấu?**  
  _Trả lời:_ Tốt vì cửa sổ lớn giúp bắt được các từ liên quan về mặt chủ đề rộng trong đoạn văn. Xấu vì nếu cửa sổ quá lớn thì nhiều từ không liên quan hoặc từ rác bị lọt vào, làm mờ đi quan hệ cú pháp và các từ đồng nghĩa thay thế trực tiếp.

- **Câu 5: Tại sao Word2Vec không phân biệt được hai nghĩa của bank?**  
  _Trả lời:_ Vì Word2Vec chỉ lưu đúng một vector tĩnh duy nhất cho mỗi từ. Khi một từ có nhiều nghĩa, mô hình bị ép phải gom tất cả các nghĩa đó vào chung một vector.

- **Câu 6: Tại sao TF-IDF không phải word embedding?**  
  _Trả lời:_ Vì TF-IDF là biểu diễn thưa (sparse) dựa trên đếm số lần từ xuất hiện trong từng văn bản cụ thể. Nó không nén thành vector số thực dày đặc (dense) và không tự động kéo các từ đồng nghĩa lại gần nhau trong không gian vector.

---

## 5. Mạch kiến thức kết nối 3 bài Lab đầu tiên (Mục 32)

- **Lab 01:** Dùng phương pháp đếm từ (TF-IDF, vector thưa) để biểu diễn văn bản và làm công cụ tìm kiếm cơ bản. Điểm yếu là không hiểu được từ đồng nghĩa.
- **Lab 02:** Dùng tần suất đếm cụm từ (N-gram) để tính xác suất của câu và dự đoán từ tiếp theo. Điểm yếu là không gian tổ hợp bị thưa và không chuyển giao được nghĩa giữa các từ tương tự.
- **Lab 03:** Chuyển từ biểu diễn thưa sang biểu diễn vector dày đặc (Co-occurrence $\rightarrow$ Word2Vec) để hai từ có nghĩa giống nhau sẽ nằm gần nhau trong không gian vector. Giới hạn là mỗi từ chỉ có 1 vector cố định, mở đường cho mô hình Transformer và Contextual Embeddings ở các bài lab sau.
