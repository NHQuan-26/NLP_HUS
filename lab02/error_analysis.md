# BÁO CÁO PHÂN TÍCH LỖI (ERROR ANALYSIS) — LAB 02

Môn học: Xử lý ngôn ngữ tự nhiên và ứng dụng  
Học kỳ: Học kỳ I - 2026  
Sinh viên: Nguyễn Hồng Quân - 23001921  
Giảng viên / TA: Phạm Ngọc Hải  
Chủ đề: N-gram Language Models, Smoothing and Perplexity

---

Theo yêu cầu tại Mục 22 của tài liệu hướng dẫn `W2.pdf`, sinh viên lựa chọn **2 trường hợp mô hình dự đoán đúng** và **2 trường hợp mô hình dự đoán sai** từ kết quả chạy thực nghiệm trên tập dữ liệu C4, sau đó tiến hành mổ xẻ nguyên nhân kỹ thuật chuyên sâu.

---

## 1. Hai trường hợp dự đoán ĐÚNG (Correct Predictions)

### Trường hợp 1: Dự đoán cụm thuật ngữ cố định

- **Context:** `"natural language"`
- **Model prediction:** `"processing"`
- **Expected (Thực tế quan sát):** `"processing"`
- **Probability:** $P(\text{processing} | \text{natural language}) \approx 0.00005$ (đứng Top 1 áp đảo)
- **Xác định nguyên nhân thành công:**
    1. **Strong Collocation (Cụm từ cố định có liên kết chặt chẽ):** Cụm từ `"natural language processing"` là một thuật ngữ khoa học tiêu chuẩn (fixed collocation). Khi hai từ `"natural language"` xuất hiện cùng nhau, xác suất để từ tiếp theo là `"processing"` cao hơn gấp nhiều lần so với bất kỳ từ nào khác trong toàn bộ từ điển.
    2. **Low Conditional Entropy:** Ngữ cảnh 2 từ này mang lượng thông tin định hướng rất mạnh, làm giảm mạnh độ bất định (entropy) của từ tiếp theo.
    3. **Độ bao phủ dữ liệu (Data Coverage):** Dù kho dữ liệu C4 là văn bản web tổng hợp, các bài viết công nghệ chứa cụm từ này xuất hiện đủ nhiều và nhất quán, giúp mô hình Trigram MLE và Laplace học được phân phối xác suất sắc nét (peaked distribution).

### Trường hợp 2: Dự đoán cụm danh từ chuyên ngành

- **Context:** `"deep neural"`
- **Model prediction:** `"network"`
- **Expected (Thực tế quan sát):** `"network"`
- **Probability:** $P(\text{network} | \text{deep neural}) \approx 0.00005$ (đứng Top 1)
- **Xác định nguyên nhân thành công:**
    1. **Domain Specificity & Lexical Constraint:** Cặp từ `"deep neural"` gần như chỉ đi kèm với một số rất ít danh từ trong tiếng Anh, phổ biến nhất là `"network"` (hoặc `"networks"`). Hầu như không có tính từ hoặc động từ nào chen vào vị trí này.
    2. **Context Length phù hợp:** Với cụm danh từ 3 từ cố định (tri-gram phrase), ngữ cảnh $n=3$ là vừa vặn hoàn hảo để bắt trọn quan hệ phụ thuộc cú pháp ngắn này mà không đòi hỏi mô hình phải ghi nhớ ngữ cảnh xa.

---

## 2. Hai trường hợp dự đoán SAI (Incorrect Predictions)

### Trường hợp 3: Dự đoán sai do lệch phân phối miền dữ liệu cục bộ (Corpus Bias)

- **Context:** `"the cat"`
- **Model prediction:** `"queen"`
- **Expected (Thực tế quan sát trong văn cảnh):** `"eats"` (hoặc các động từ phổ thông như `"was"`, `"is"`, `"sat"`)
- **Probability của Top 1:** $P(\text{queen} | \text{the cat}) \approx 0.00007$ (xác suất của `"eats"` chỉ đạt mức nền làm mịn)
- **Xác định nguyên nhân lỗi:**
    1. **Domain Mismatch & Corpus Bias:** Trong tập con 15.000 văn bản C4 được cào từ Internet, có bài viết chứa cụm từ đặc thù `"the cat queen"` (tên một nhân vật, trò chơi hoặc thương hiệu) lặp lại nhiều lần trong một trang web duy nhất. Điều này tạo ra đột biến tần số giả tạo cho cụm 3 từ hiếm gặp này.
    2. **Sparsity & Long Tail (Độ thưa dữ liệu):** Trong ngôn ngữ tự nhiên thông thường, sau danh từ `"the cat"` có thể là hàng ngàn động từ khác nhau (`sat`, `walked`, `eats`, `jumped`, `meowed`, `was`). Vì số lượng hành động quá đa dạng, xác suất thực tế bị phân tán mỏng (flattened) ra hàng trăm từ khác nhau, mỗi từ chỉ xuất hiện 1-2 lần. Do đó, một từ xuất hiện lặp lại cục bộ 4-5 lần dễ dàng "vượt mặt" các từ tự nhiên hơn.
    3. **Context quá ngắn (Context Limitation):** Ngữ cảnh 2 từ `"the cat"` là hoàn toàn không đủ thông tin để xác định ngữ cảnh ngữ nghĩa (chủ đề bài viết đang nói về dinh dưỡng của mèo hay truyện ngụ ngôn). Mô hình n-gram không có cơ chế chú ý toàn cục (global attention) hay hiểu biết về thế giới thực (commonsense knowledge).

### Trường hợp 4: Dự đoán sai do thiên kiến từ chức năng tần suất cao (Function Word Bias)

- **Context:** `"machine learning"`
- **Model prediction:** `"and"`
- **Expected (Thực tế quan sát):** `"models"` (từ này đứng ở vị trí Top 4 với $P \approx 0.00008$)
- **Probability của Top 1:** $P(\text{and} | \text{machine learning}) \approx 0.00010$
- **Xác định nguyên nhân lỗi:**
    1. **Function Words Over-representation:** Từ liên từ `"and"` là một trong những từ có tần số xuất hiện cao nhất trong toàn bộ tiếng Anh. Trong văn bản, người ta rất hay viết các mệnh đề liệt kê như: `"machine learning and artificial intelligence"`, `"machine learning and data science"`, `"machine learning and its applications"`.
    2. **Thiếu khả năng phân tích cú pháp (No Syntactic Awareness):** Mô hình n-gram chỉ đơn thuần đếm tần suất chuỗi token bề mặt, hoàn toàn không biết vai trò ngữ pháp của từ (Part-of-Speech). Nó không nhận thức được rằng sau cụm danh từ chủ ngữ `"machine learning"`, câu cần một vị ngữ (động từ như `is`, `helps`) hoặc một danh từ ghép bổ nghĩa (`models`, `algorithms`) thay vì một liên từ kết nối câu.
    3. **Smoothing Artifact (Tác dụng phụ của làm mịn):** Vì $V$ lên tới hơn 130.000 từ, việc làm mịn Laplace phân bổ một lượng xác suất đều rất lớn cho không gian từ vựng, khiến mô hình càng có xu hướng thiên vị những từ có số lần quan sát lớn trong tập train để bù trừ lại mẫu số khổng lồ ($C(h) + V$).

---

## 3. Tổng kết bài học rút ra từ phân tích lỗi

| Nguyên nhân chính                   | Biểu hiện trong thực nghiệm                                                                        | Giải pháp kỹ thuật tương ứng                                                                |
| :---------------------------------- | :------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------ |
| **Data Sparsity (Độ thưa dữ liệu)** | 87.58% trigram chỉ xuất hiện 1 lần; nhiều từ tự nhiên bị zero probability                          | Dùng Kneser-Ney smoothing, Word Embeddings                                                  |
| **Context Length Limitation**       | $n=3$ không hiểu chủ đề bao quát của cả đoạn văn                                                   | RNN, LSTM, Transformer (Context window hàng nghìn tokens)                                   |
| **Lack of Semantic Understanding**  | Không phân biệt được mèo ("cat") là động vật thì phải đi với động từ ăn ("eats") hay ngủ ("slept") | Pretrained Foundation Models học biểu diễn ngữ nghĩa dày (Dense Contextual Representations) |
| **Corpus Bias & Function Words**    | Từ nối `"and"` hoặc tên riêng lặp cục bộ chiếm vị trí top dự đoán                                  | Sử dụng mô hình hóa phân cấp cú pháp, Subword tokenization (BPE), Temperature scaling       |
