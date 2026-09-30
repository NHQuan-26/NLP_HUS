# DỰ ĐOÁN TRƯỚC THỰC NGHIỆM (PREDICTION BEFORE EXPERIMENT) — LAB 02

Môn học: Xử lý ngôn ngữ tự nhiên và ứng dụng  
Học kỳ: Học kỳ I - 2026  
Sinh viên: Nguyễn Hồng Quân - 23001921  
Giảng viên / TA: Phạm Ngọc Hải  
Chủ đề: N-gram Language Models, Smoothing and Perplexity  

---

Theo quy định tại Mục 12 của tài liệu hướng dẫn `W2.pdf`, sinh viên bắt buộc phải đưa ra dự đoán và lập luận lý thuyết trước khi tiến hành viết code và chạy thực nghiệm. Dưới đây là 5 dự đoán độc lập của em:

---

## Prediction 1 — Thay đổi kích thước từ vựng (Vocabulary)

- **Câu hỏi:** Khi chuyển từ `unigram → bigram → trigram`, vocabulary có tăng không?
- **Prediction:** **Không tăng.** Kích thước từ vựng ($|V|$) hoàn toàn không đổi.
- **Reason:** Vocabulary (từ vựng) trong xử lý ngôn ngữ tự nhiên được định nghĩa là tập hợp tất cả các đơn vị từ đơn lẻ độc nhất (unique words / tokens) được trích xuất từ kho ngữ liệu. Việc ta xem xét các tổ hợp 2 từ liên tiếp (bigrams) hay 3 từ liên tiếp (trigrams) chỉ làm tăng số lượng các loại tuple (n-gram types), chứ không làm thay đổi tập từ vựng gốc cấu thành nên các chuỗi đó. Nếu tập từ vựng unigram có kích thước $|V|$ thì các từ trong bigram và trigram cũng chỉ lấy từ chính tập $|V|$ đó.
- **Confidence:** **95%**

---

## Prediction 2 — Số lượng n-gram phân biệt (Unique n-grams)

- **Câu hỏi:** Số lượng n-gram (unique n-grams) sẽ thay đổi như thế nào?
- **Prediction:** **Tăng rất mạnh theo cấp số nhân**, theo thứ tự:
$$\#\text{unique trigrams} \gg \#\text{unique bigrams} > \#\text{unique unigrams}$$
Đồng thời, tỉ lệ các n-gram chỉ xuất hiện đúng 1 lần (singletons) sẽ tăng vọt và chiếm đa số tuyệt đối ở mô hình Trigram.
- **Reason:** Về mặt lý thuyết tổ hợp, số lượng bigram có thể tạo ra tối đa là $|V|^2$ và trigram là $|V|^3$. Mặc dù các quy tắc ngữ pháp và ngữ nghĩa tự nhiên giới hạn các từ đi cùng nhau, số lượng cụm 2 từ và 3 từ phân biệt trong một tập dữ liệu lớn (như 10K–30K văn bản C4) vẫn lớn hơn gấp nhiều lần so với số từ đơn. Do hiện tượng "đuôi dài" (Zipf's law / Long-tail distribution), phần lớn các cụm 3 từ chỉ xuất hiện thoáng qua một lần trong toàn bộ ngữ liệu.
- **Confidence:** **90%**

---

## Prediction 3 — Khả năng gặp xác suất bằng 0 (Zero Probability)

- **Câu hỏi:** Mô hình nào có khả năng gặp zero probability nhiều hơn?
- **Prediction:** **Mô hình Trigram có khả năng gặp zero probability cao nhất**, tiếp sau đó là Bigram, và thấp nhất là Unigram.
- **Reason:** Hiện tượng zero probability xảy ra khi mô hình đánh giá một n-gram chưa từng xuất hiện trong tập huấn luyện (unseen n-gram). Không gian của trigram ($|V|^3$) quá rộng lớn so với kích thước hữu hạn của dữ liệu huấn luyện, khiến phần lớn các tổ hợp 3 từ hợp lệ trong thực tế không có mặt trong training corpus. Ngược lại, với Unigram, chỉ cần một từ đã từng xuất hiện ít nhất một lần ở bất kỳ vị trí nào trong tập train thì xác suất của nó đã lớn hơn 0. Do đó, ngữ cảnh càng dài ($n$ càng lớn) thì độ thưa (sparsity) càng cao và khả năng bắt gặp $C = 0$ khi kiểm thử càng trầm trọng.
- **Confidence:** **95%**

---

## Prediction 4 — Perplexity trên Training Set

- **Câu hỏi:** Mô hình nào dự kiến có perplexity thấp hơn trên training set?
- **Prediction:** **Mô hình Trigram sẽ có Perplexity thấp nhất trên Training set**, tiếp theo là Bigram, và cao nhất là Unigram:
$$\text{PPL}_{\text{train}}(\text{Trigram}) < \text{PPL}_{\text{train}}(\text{Bigram}) < \text{PPL}_{\text{train}}(\text{Unigram})$$
- **Reason:** Perplexity đo lường mức độ "bối rối" của mô hình; perplexity càng thấp thì mô hình gán xác suất càng cao cho chuỗi từ. Trên tập dữ liệu huấn luyện, mô hình Trigram có nhiều tham số hơn và sử dụng ngữ cảnh dài hơn ($w_{i-2}, w_{i-1}$) để thu hẹp không gian lựa chọn từ tiếp theo. Điều này giúp mô hình "ghi nhớ" (memorize) rất tốt các mẫu câu có sẵn trong tập train, dẫn đến log-likelihood trên tập train rất cao và Perplexity rất thấp.
- **Confidence:** **90%**

---

## Prediction 5 — Trigram so với Bigram trên Corpus nhỏ

- **Câu hỏi:** Nếu corpus rất nhỏ, trigram có chắc chắn tốt hơn bigram không?
- **Prediction:** **Không chắc chắn.** Thậm chí trên dữ liệu kiểm thử (validation/test set), **Trigram thường sẽ hoạt động kém hơn (perplexity cao hơn nhiều) so với Bigram.**
- **Reason:** Khi corpus nhỏ, dữ liệu không đủ dày để ước lượng đáng tin cậy các xác suất điều kiện của trigram $P(w_i | w_{i-2}, w_{i-1})$. Hầu hết các ngữ cảnh 2 từ sẽ có tần số bằng 0 hoặc 1. Khi đó, mô hình Trigram bị hiện tượng quá khớp (overfitting) nghiêm trọng trên tập train và gặp hàng loạt unseen n-grams trên tập test. Nếu không có kỹ thuật làm mịn (smoothing) hoặc backoff/interpolation cực tốt, Trigram sẽ gán xác suất 0 (hoặc xác suất làm mịn kém) cho câu mới, dẫn đến Perplexity trên test set tăng vọt so với Bigram vốn có độ khái quát hóa tốt hơn.
- **Confidence:** **95%**
