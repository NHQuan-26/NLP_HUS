# TỔNG KẾT VÀ TỰ ĐÁNH GIÁ (REFLECTION & LEARNING CHECK) — LAB 02

Môn học: Xử lý ngôn ngữ tự nhiên và ứng dụng  
Học kỳ: Học kỳ I - 2026  
Sinh viên: Nguyễn Hồng Quân - 23001921  
Giảng viên / TA: Phạm Ngọc Hải  
Chủ đề: N-gram Language Models, Smoothing and Perplexity  

---

## 1. Phần suy ngẫm và trả lời câu hỏi tổng kết (Section 24)

### Câu 1: Nếu tăng $n$, mô hình nhận thêm thông tin gì?
Khi tăng $n$ (từ unigram lên bigram, trigram, $n$-gram), mô hình nhận thêm thông tin về **lịch sử ngữ cảnh cục bộ (local sequential context)** và **trật tự phụ thuộc giữa các từ lân cận (word order & local syntax)**.
- Unigram ($n=1$) hoàn toàn mù trật tự từ (giả định độc lập túi từ - bag of words).
- Bigram ($n=2$) nắm bắt được từ liền kề ngay trước đó ($w_{i-1}$).
- Trigram ($n=3$) nắm bắt được ngữ cảnh 2 từ trước ($w_{i-2}, w_{i-1}$), cho phép nhận diện các danh từ ghép, ngữ cố định (collocations) và cấu trúc cụm động từ - tân ngữ ngắn.
Nói cách khác, tăng $n$ giúp mô hình giải quyết tính nhập nhằng (disambiguation) và thu hẹp không gian lựa chọn cho từ tiếp theo (giảm conditional entropy).

### Câu 2: Tại sao tăng $n$ lại làm sparsity (độ thưa) tăng?
Độ thưa tăng vọt theo cấp số nhân là hệ quả của bài toán bùng nổ tổ hợp (**Curse of Dimensionality**):
- Nếu từ vựng có kích thước $|V|$, số lượng $n$-gram khả dĩ về mặt lý thuyết là $|V|^n$.
  - Với $|V| \approx 130.000$, số bigram khả dĩ là $1.69 \times 10^{10}$, số trigram khả dĩ lên tới $2.2 \times 10^{15}$.
- Trong khi đó, bất kỳ tập dữ liệu huấn luyện nào cũng chỉ có số lượng token hữu hạn (khoảng $5.56 \times 10^6$ tokens trong thực nghiệm này).
- Do sự chênh lệch khủng khiếp giữa không gian trạng thái lý thuyết và số lượng quan sát thực tế, phần lớn các tổ hợp $n$-gram hợp lệ không bao giờ xuất hiện trong tập train. Thực nghiệm cho thấy tỉ lệ $n$-gram xuất hiện duy nhất 1 lần (singletons) tăng vọt từ $47.18\%$ ở unigram lên **$87.58\%$** ở trigram!

### Câu 3: Tại sao smoothing cần thiết?
Smoothing (làm mịn xác suất) là bắt buộc vì hai lý do cốt lõi:
1. **Khắc phục Zero Probability Problem:** Trong mô hình MLE thuần túy, nếu gặp một $n$-gram chưa từng thấy trong tập train ($C = 0$), xác suất của nó bằng 0. Theo chain rule, điều này làm toàn bộ xác suất của câu bị triệt tiêu về $0$ và Perplexity bùng nổ lên $\infty$.
2. **Khái quát hóa trên dữ liệu chưa thấy (Generalization):** "Chưa quan sát thấy trong tập mẫu $\neq$ Xác suất thực tế bằng 0". Smoothing thực hiện việc chiết khấu (discount) một phần khối lượng xác suất từ các $n$-gram phổ biến để chia sẻ cho các sự kiện hiếm gặp hoặc chưa từng thấy, giúp mô hình hoạt động bền bỉ trên dữ liệu thực tế.

### Câu 4: Perplexity đo điều gì?
- Về mặt toán học: Perplexity ($PP$) là hàm mũ của cross-entropy:
$$PP(W) = \exp\left(-\frac{1}{N} \sum_{i=1}^N \ln P(w_i | \text{context})\right)$$
- Về mặt trực quan và lý thuyết thông tin: Perplexity đo lường **mức độ "bối rối" (confusion/uncertainty)** của mô hình khi quan sát chuỗi từ. Nó tương đương với **số lượng lựa chọn bình đẳng trung bình (branching factor)** mà mô hình cảm thấy phân vân tại mỗi vị trí sinh từ.
- Perplexity càng thấp thì mô hình gán xác suất trung bình càng cao cho dữ liệu thực tế, tức mô hình càng "ít ngạc nhiên" trước văn bản đó.

### Câu 5: Một model có perplexity thấp hơn có luôn tạo ra văn bản tốt hơn đối với con người không? Giải thích.
- **Không nhất thiết.**
- **Giải thích:**
  1. Perplexity chỉ đo lường xác suất thống kê trung bình trên từng token độc lập dưới giả định Markov ngắn, chứ không đánh giá được tính mạch lạc ngữ nghĩa (semantic coherence), tính logic toàn cục (global coherence), sự lặp từ (repetitiveness) hay phong cách hành văn của cả đoạn văn dài.
  2. Một mô hình có thể đạt Perplexity thấp bằng cách liên tục dự đoán các từ an toàn, phổ biến (như *the, is, and, of*), nhưng khi sinh văn bản thì tạo ra các câu vô nghĩa, lặp đi lặp lại hoặc hallucination.
  3. Perplexity rất nhạy cảm với cách tiền xử lý, kích thước từ vựng ($|V|$) và cơ chế làm mịn. Một mô hình có từ vựng nhỏ hoặc cơ chế gán xác suất lệch có thể có Perplexity số học thấp nhưng khả năng sinh ngôn ngữ thực tế lại kém.

### Câu 6: N-gram language model thất bại ở đâu khi so với cách con người hiểu ngôn ngữ?
Mô hình $n$-gram bộc lộ những giới hạn nền tảng so với năng lực ngôn ngữ của con người:
1. **Thiếu hiểu biết ngữ nghĩa (No Semantic Representation):** $n$-gram coi các từ là các ký hiệu rời rạc (discrete symbols). Nó không hiểu rằng *"cat"* và *"feline"* hay *"dog"* và *"puppy"* có quan hệ họ hàng mật thiết.
2. **Giả định Markov quá ngắn (Markov Assumption):** $n$-gram chỉ nhìn lại được $1$ hoặc $2$ từ trước đó. Trong khi ngôn ngữ con người có các phụ thuộc tầm xa (long-range dependencies), ví dụ cấu trúc chủ ngữ - vị ngữ cách nhau cả mệnh đề quan hệ hay sự liên kết ý xuyên suốt các đoạn văn.
3. **Không có khả năng tổng quát hóa theo ngữ cảnh (Contextual Invariance):** Từ xuất hiện trong ngữ cảnh mới không thể kế thừa thông tin từ các cấu trúc tương đồng đã học nếu câu chữ không khớp cơ học từng từ.

### Câu 7: Nếu context dài 100 từ, trigram có sử dụng được thông tin của 97 từ đầu không?
- **Hoàn toàn KHÔNG.**
- **Cầu nối sang Neural LM & Transformer:** Trigram bị ràng buộc bởi giả định Markov bậc 2:
$$P(w_{100} | w_1, w_2, \dots, w_{99}) \approx P(w_{100} | w_{98}, w_{99})$$
Mô hình hoàn toàn vứt bỏ toàn bộ thông tin của 97 từ đầu tiên ($w_1 \dots w_{97}$). Nếu chủ ngữ của câu hoặc chủ đề của bài viết nằm ở những từ đầu, trigram hoàn toàn "mù" trước thông tin đó.
Chính sự bất lực này của $n$-gram đã thúc đẩy sự ra đời của:
- **Neural Language Models / Word Embeddings (LAB 03):** Biểu diễn từ thành vector liên tục dày đặc (dense vectors).
- **RNN / LSTM:** Dùng bộ nhớ ẩn (hidden state) để duy trì thông tin qua các bước thời gian.
- **Attention & Transformer:** Cơ chế tự chú ý (Self-Attention) cho phép mô hình kết nối trực tiếp bất kỳ từ nào trong câu với toàn bộ ngữ cảnh phía trước mà không bị giới hạn khoảng cách.

---

## 2. Bản khai báo sử dụng AI (AI Assistance Statement — Section 25)

Tuân thủ nghiêm ngặt quy định AI Policy tại Mục 25 của đề bài `W2.pdf`:

- **Tool:** Google Antigravity Assistant.
- **Purpose:** 
  - Hỗ trợ xây dựng khung mã nguồn mô đun hóa trong `ngram_lm.py` và tối ưu hóa vòng lặp đếm tần suất n-gram trên tập dữ liệu C4 lớn (327.317 câu).
  - Hỗ trợ cú pháp lưu trữ và trực quan hóa biểu đồ phân phối tần suất bằng Matplotlib.
- **What was generated:**
  - Khung cấu trúc class `NGramLanguageModel` (hỗ trợ lưu trữ cấu trúc Counter, fit, probability, perplexity).
  - Script điều phối thực nghiệm `run_experiments.py` và xuất file `results.csv`.
- **What was modified:**
  - Tinh chỉnh hàm `probability` và `predict_next_words` để xử lý chuẩn hóa token hóa chuỗi ngữ cảnh đầu vào (tránh lỗi context string OOV).
  - Bổ sung cơ chế xử lý từ ngoài từ vựng (`<unk>` cutoff) để bảo toàn tổng xác suất phân phối bằng đúng 1.0.
  - Tùy biến hàm `rank_continuations` chuẩn hóa theo Perplexity nhằm so sánh công bằng các câu có độ dài khác nhau.
- **How the result was verified:**
  - Viết test suite độc lập kiểm chứng các kết quả tính toán của `NGramLanguageModel` đối chiếu với các bài toán tính tay tại Section 7, 9, 11, 18.
  - Toàn bộ kết quả số học trong `calculations.md`, các lập luận dự đoán trong `prediction.md`, phân tích lỗi chi tiết trong `error_analysis.md`, và phần trả lời các câu hỏi tổng kết/kiểm tra miệng trong `reflection.md` đều được kiểm chứng và đối soát chặt chẽ với bản chất toán học của ngôn ngữ.

---

## 3. Trả lời các câu hỏi kiểm tra miệng cá nhân (Section 26 — Individual Learning Check)

### Câu hỏi 1: Vì sao Bigram có zero probability?
**Trả lời:**
Bigram ước lượng xác suất bằng tỉ số đếm: $P(w_i | w_{i-1}) = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})}$.  
Nếu cặp từ $(w_{i-1}, w_i)$ chưa từng xuất hiện cùng nhau trong tập dữ liệu huấn luyện, số lần đếm $C(w_{i-1}, w_i) = 0$, dẫn đến xác suất ước lượng MLE bằng $0$. Khi tính xác suất cả câu theo quy tắc nhân chuỗi, chỉ cần duy nhất một bigram bằng $0$ là toàn bộ xác suất của câu bị triệt tiêu về $0$.

### Câu hỏi 2: Tại sao phải dùng log probability?
**Trả lời:**
Vì xác suất của một câu là tích của hàng chục xác suất thành phần nhỏ ($0 < P \le 1$). Khi nhân liên tiếp nhiều số thực nhỏ, kết quả sẽ nhanh chóng rơi xuống dưới ngưỡng biểu diễn số thực dấu phẩy động của máy tính (ví dụ $10^{-50} \to 10^{-300}$), gây ra lỗi tràn số dưới (**numerical underflow**) và máy tính coi giá trị đó bằng $0.0$.  
Bằng cách lấy $\log$, phép nhân chuỗi được chuyển thành phép cộng các số âm:
$$\log P(W) = \sum_{i=1}^K \log P(w_i | \text{context})$$
Phép cộng log giúp giá trị ổn định số học tuyệt đối, không bao giờ bị underflow.

### Câu hỏi 3: Perplexity thấp nghĩa là gì?
**Trả lời:**
Perplexity thấp nghĩa là mô hình gán xác suất cao cho chuỗi từ đang xét, tức là mô hình có độ tự tin cao và cảm thấy "ít bối rối" trước dữ liệu. Về mặt phân nhánh, perplexity thấp biểu thị số lượng từ ứng viên mà mô hình phải phân vân tại mỗi bước là nhỏ.

### Câu hỏi 4: Tại sao trigram không nhất thiết tốt hơn bigram trên test set?
**Trả lời:**
Vì hiện tượng **quá khớp (overfitting)** và **thưa dữ liệu (data sparsity)**. Trigram có không gian tổ hợp rất lớn ($|V|^3$). Trên tập huấn luyện hữu hạn, phần lớn các ngữ cảnh 2 từ chỉ xuất hiện rất ít lần. Khi sang tập test, hầu hết các trigram sẽ là cụm từ chưa từng thấy (unseen). Nếu phương pháp làm mịn không tối ưu (như Laplace với $V$ quá lớn), xác suất gán cho các unseen trigram sẽ cực kỳ nhỏ, khiến Perplexity của Trigram trên test set bị đội lên rất cao, kém hơn hẳn Bigram vốn có tính khái quát hóa bền vững hơn.

### Câu hỏi 5: Nếu "cat eats" chưa xuất hiện trong training corpus thì model xử lý thế nào?
**Trả lời:**
- **Nếu dùng MLE thuần túy:** Mô hình gán $C(\text{cat eats}) = 0 \implies P(\text{eats}|\text{cat}) = 0$. Khi đó bất kỳ câu nào chứa cụm từ này đều nhận xác suất bằng $0$ (thất bại hoàn toàn).
- **Nếu dùng Laplace (Add-one) Smoothing:** Mô hình cộng thêm $1$ ảo vào tử số và $V$ vào mẫu số:
$$P_{\text{Laplace}}(\text{eats}|\text{cat}) = \frac{0 + 1}{C(\text{cat}) + V} = \frac{1}{C(\text{cat}) + V} > 0$$
Mô hình gán cho cụm từ này một xác suất dương nhỏ tỉ lệ nghịch với kích thước từ vựng, giúp câu chứa cụm từ vẫn tính được xác suất và Perplexity hữu hạn.

---

## 4. Mối liên hệ xuyên suốt giữa LAB 01 và LAB 02 (Section 30)

Điểm cốt lõi thú vị nhất khi đối chiếu hai bài lab: **Cùng xuất phát từ khái niệm tần suất đếm (Count / Frequency) từ ngữ liệu, nhưng hai bài lab phát triển theo hai hướng hoàn toàn khác biệt:**

```mermaid
flowchart TD
    subgraph LAB01 ["LAB 01: Count -> Representation -> Search"]
        T1["Text Documents"] --> Tok1["Tokenization"]
        Tok1 --> C1["Term Frequency Count c(t, d)"]
        C1 --> TFIDF["TF-IDF Weighting"]
        TFIDF --> Rep["Vector Space Representation (Dense/Sparse Vectors)"]
        Rep --> Sim["Cosine Similarity"]
        Sim --> Search["Information Retrieval & Search Engine"]
    end

    subgraph LAB02 ["LAB 02: Count -> Probability -> Generation"]
        T2["Text Documents"] --> Tok2["Tokenization"]
        Tok2 --> C2["N-gram Counts C(h, w)"]
        C2 --> CondP["Conditional Probability P(w|h)"]
        CondP --> LM["N-gram Language Model (MLE / Smoothing)"]
        LM --> Score["Sentence Probability & Perplexity"]
        Score --> App["Next-word Prediction & Sentence Ranking"]
    end
```

### So sánh bản chất:
- **LAB 01 (Count $\to$ Representation):** Tần suất từ được dùng để xác định "tầm quan trọng" của từ trong văn bản thông qua trọng số $TF \times IDF$, nhằm định vị văn bản trong không gian vector đa chiều phục vụ bài toán so khớp nội dung (Information Retrieval).
- **LAB 02 (Count $\to$ Probability):** Tần suất từ và cụm từ được dùng để ước lượng phân phối xác suất có điều kiện $P(w|h)$ theo xích Markov, nhằm mô hình hóa độ mượt mà, tự nhiên của ngôn ngữ phục vụ bài toán dự đoán và sinh văn bản (Language Generation).

### Cầu nối sang LAB 03 (Word Representations & Embeddings):
Cả LAB 01 và LAB 02 đều sử dụng **biểu diễn rời rạc dựa trên đếm (Sparse Count-based Representations)**:
- LAB 01 chịu tổn thương vì không hiểu từ đồng nghĩa (synonyms), nhầm lẫn từ đa nghĩa (polysemy) trong tìm kiếm.
- LAB 02 chịu tổn thương vì không gian thưa thớt bùng nổ ($|V|^3$), không chuyển giao được tri thức giữa các từ có ngữ nghĩa tương đồng khi dự đoán câu.

Chính hạn chế cốt tử này mở ra cánh cửa cho **LAB 03: Word Representations / Word Embeddings**:
$$\text{Sparse Count-based Representation} \longrightarrow \text{Dense Distributed Representation (Word2Vec, GloVe)} \longrightarrow \text{Contextual Representation (Transformer, BERT, GPT)}$$
Đây là bước nhảy vọt quan trọng nhất kết nối NLP cổ điển với Kỷ nguyên Trí tuệ Nhân tạo hiện đại.
