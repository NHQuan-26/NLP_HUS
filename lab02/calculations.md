# BÀI TẬP TÍNH TOÁN BẰNG TAY (MANUAL CALCULATIONS) — LAB 02

Môn học: Xử lý ngôn ngữ tự nhiên và ứng dụng  
Học kỳ: Học kỳ I - 2026  
Sinh viên: Nguyễn Hồng Quân - 23001921  
Giảng viên / TA: Phạm Ngọc Hải  
Chủ đề: N-gram Language Models, Smoothing and Perplexity  

---

## 1. Bài 1 — Unigram (Mục 7)

Cho corpus gồm 3 câu:
- $D_1$: `"the cat eats fish"`
- $D_2$: `"the cat likes fish"`
- $D_3$: `"the dog eats meat"`

### 1.1. Xác định vocabulary
Tập các từ phân biệt (lexicon) được trích xuất từ corpus:
$$V = \{\text{cat}, \text{dog}, \text{eats}, \text{fish}, \text{likes}, \text{meat}, \text{the}\}$$
- Kích thước từ vựng: $|V| = 7$.

### 1.2. Tính tổng số token
- Câu $D_1$: 4 tokens (`the`, `cat`, `eats`, `fish`)
- Câu $D_2$: 4 tokens (`the`, `cat`, `likes`, `fish`)
- Câu $D_3$: 4 tokens (`the`, `dog`, `eats`, `meat`)
- Tổng số token trong corpus:
$$N = 4 + 4 + 4 = 12 \text{ tokens}$$

### 1.3. Tính $P(w)$ theo Unigram Maximum Likelihood Estimation (MLE)
Công thức:
$$P(w) = \frac{C(w)}{N}$$

Đếm số lần xuất hiện của từng từ:
- $C(\text{the}) = 3 \implies P(\text{the}) = \frac{3}{12} = 0.25$
- $C(\text{cat}) = 2 \implies P(\text{cat}) = \frac{2}{12} = \frac{1}{6} \approx 0.1667$
- $C(\text{fish}) = 2 \implies P(\text{fish}) = \frac{2}{12} = \frac{1}{6} \approx 0.1667$
- $C(\text{dog}) = 1 \implies P(\text{dog}) = \frac{1}{12} \approx 0.0833$

*(Các từ còn lại trong từ vựng: $C(\text{eats}) = 2 \implies P(\text{eats}) = \frac{2}{12} \approx 0.1667$; $C(\text{likes}) = 1 \implies P(\text{likes}) = \frac{1}{12} \approx 0.0833$; $C(\text{meat}) = 1 \implies P(\text{meat}) = \frac{1}{12} \approx 0.0833$)*.

### 1.4. Kiểm tra tổng xác suất
$$\sum_{w \in V} P(w) = \frac{C(\text{the}) + C(\text{cat}) + C(\text{fish}) + C(\text{dog}) + C(\text{eats}) + C(\text{likes}) + C(\text{meat})}{N}$$
$$\sum_{w \in V} P(w) = \frac{3 + 2 + 2 + 1 + 2 + 1 + 1}{12} = \frac{12}{12} = 1.0$$
Tổng xác suất bằng đúng $1.0$, thỏa mãn tiên đề xác suất.

---

## 2. Bài 2 — Bigram (Mục 7)

Công thức MLE cho Bigram:
$$P(w_i | w_{i-1}) = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})}$$

### 2.1. Tính các xác suất có điều kiện cụ thể
1. **Xét context $w_{i-1} = \text{"the"}$ ($C(\text{the}) = 3$):**
   - Bigram `"the cat"` xuất hiện ở $D_1, D_2 \implies C(\text{the cat}) = 2$.
   - Bigram `"the dog"` xuất hiện ở $D_3 \implies C(\text{the dog}) = 1$.
   - Tính:
     $$P(\text{cat}|\text{the}) = \frac{C(\text{the cat})}{C(\text{the})} = \frac{2}{3} \approx 0.6667$$
     $$P(\text{dog}|\text{the}) = \frac{C(\text{the dog})}{C(\text{the})} = \frac{1}{3} \approx 0.3333$$

2. **Xét context $w_{i-1} = \text{"cat"}$ ($C(\text{cat}) = 2$):**
   - Bigram `"cat eats"` xuất hiện ở $D_1 \implies C(\text{cat eats}) = 1$.
   - Bigram `"cat likes"` xuất hiện ở $D_2 \implies C(\text{cat likes}) = 1$.
   - Tính:
     $$P(\text{eats}|\text{cat}) = \frac{C(\text{cat eats})}{C(\text{cat})} = \frac{1}{2} = 0.5$$
     $$P(\text{likes}|\text{cat}) = \frac{C(\text{cat likes})}{C(\text{cat})} = \frac{1}{2} = 0.5$$

### 2.2. Trả lời câu hỏi:
**Tại sao tổng xác suất của các từ đứng sau "the" phải bằng 1 nếu vocabulary và context được xử lý đầy đủ?**

**Trả lời:**
Tổng xác suất của mọi từ $w$ đứng sau `"the"` là:
$$\sum_{w \in V} P(w|\text{the}) = \sum_{w \in V} \frac{C(\text{the}, w)}{C(\text{the})} = \frac{\sum_{w \in V} C(\text{the}, w)}{C(\text{the})}$$
Vì mỗi khi từ `"the"` xuất hiện ở vị trí không phải cuối câu (hoặc khi luôn có token kết thúc câu `</s>`), nó bắt buộc phải được theo sau bởi đúng một từ tiếp theo. Do đó, tổng số lần xuất hiện của tất cả các cặp bigram bắt đầu bằng `"the"` chính bằng tổng số lần từ `"the"` xuất hiện trong vai trò từ đứng trước:
$$\sum_{w \in V} C(\text{the}, w) = C(\text{the})$$
Do đó tỉ số luôn bằng $1$. Đây là điều kiện chuẩn hóa cơ bản để $P(\cdot|\text{the})$ tạo thành một phân phối xác suất có điều kiện hợp lệ (proper probability distribution).

---

## 3. Bài 3 — Xác suất câu (Mục 7)

Xét câu: $S = \text{"the cat eats fish"}$.  
Phân rã xác suất câu theo mô hình Bigram và Chain Rule:
$$P(S) = P(\text{the}) \times P(\text{cat}|\text{the}) \times P(\text{eats}|\text{cat}) \times P(\text{fish}|\text{eats})$$

### 3.1. Tính từng thành phần xác suất:
- $P(\text{the}) = \frac{3}{12} = \frac{1}{4} = 0.25$ (xác suất unigram mở đầu câu)
- $P(\text{cat}|\text{the}) = \frac{2}{3} \approx 0.6667$
- $P(\text{eats}|\text{cat}) = \frac{1}{2} = 0.5$
- Xét $P(\text{fish}|\text{eats})$:
  - $C(\text{eats}) = 2$ (trong $D_1$: `"eats fish"`, trong $D_3$: `"eats meat"`).
  - $C(\text{eats fish}) = 1$ (ở $D_1$).
  - $P(\text{fish}|\text{eats}) = \frac{C(\text{eats fish})}{C(\text{eats})} = \frac{1}{2} = 0.5$.

### 3.2. Tính xác suất toàn bộ câu:
$$P(S) = \frac{1}{4} \times \frac{2}{3} \times \frac{1}{2} \times \frac{1}{2} = \frac{2}{48} = \frac{1}{24} \approx 0.04167 \quad (\approx 4.167\%)$$

### 3.3. Trả lời câu hỏi:
**Nếu thêm một từ vào câu, xác suất của cả câu có thể tăng không?**

**Trả lời:**
- **Không bao giờ tăng.** Xác suất của cả câu chỉ có thể **giảm đi** hoặc **giữ nguyên** (trong trường hợp từ thêm vào có xác suất có điều kiện bằng $1.0$).
- **Giải thích toán học:** Khi mở rộng chuỗi từ $W_{1:k}$ thành $W_{1:k+1}$, theo Chain Rule:
$$P(W_{1:k+1}) = P(W_{1:k}) \times P(w_{k+1} | w_k)$$
Vì mọi xác suất có điều kiện đều nằm trong đoạn $[0, 1]$, việc nhân thêm một thừa số $P(w_{k+1}|w_k) \le 1$ sẽ luôn khiến:
$$P(W_{1:k+1}) \le P(W_{1:k})$$
- **Ý nghĩa sư phạm:** Xác suất toàn chuỗi $P(S)$ tỉ lệ nghịch với độ dài câu. Câu càng dài thì số lượng thừa số nhân vào càng nhiều, làm cho $P(S)$ tự nhiên nhỏ đi rất nhiều. Vì vậy, $P(S)$ không phải là thước đo công bằng để so sánh trực tiếp hai câu có độ dài khác nhau. Để so sánh khách quan, người ta phải chuẩn hóa theo độ dài bằng Perplexity ($PP = P(S)^{-1/N}$) hoặc trung bình log-likelihood trên mỗi token.

---

## 4. Bài 4 — Sentence Ranking (Mục 7)

Cho hai câu:
- $S_1 = \text{"the cat eats fish"}$
- $S_2 = \text{"the dog eats fish"}$

### 4.1. Tính xác suất câu $S_2$:
$$P(S_2) = P(\text{the}) \times P(\text{dog}|\text{the}) \times P(\text{eats}|\text{dog}) \times P(\text{fish}|\text{eats})$$
Các thành phần:
- $P(\text{the}) = \frac{3}{12} = \frac{1}{4}$
- $P(\text{dog}|\text{the}) = \frac{1}{3}$
- Xét $P(\text{eats}|\text{dog})$:
  - $C(\text{dog}) = 1$ (ở câu $D_3$).
  - Bigram `"dog eats"` xuất hiện 1 lần (ở câu $D_3$).
  - $\implies P(\text{eats}|\text{dog}) = \frac{C(\text{dog eats})}{C(\text{dog})} = \frac{1}{1} = 1.0$.
- $P(\text{fish}|\text{eats}) = \frac{1}{2}$ (như đã tính ở Bài 3).

Thay số:
$$P(S_2) = \frac{1}{4} \times \frac{1}{3} \times 1.0 \times \frac{1}{2} = \frac{1}{24} \approx 0.04167$$

### 4.2. So sánh và kết luận:
$$P(S_1) = \frac{1}{24} \approx 0.04167$$
$$P(S_2) = \frac{1}{24} \approx 0.04167$$
- **Kết luận:** Hai câu có **xác suất hoàn toàn bằng nhau**: $P(S_1) = P(S_2)$.
- **Phân tích chiều sâu:** 
  - Mặc dù `"cat"` đi sau `"the"` phổ biến hơn `"dog"` ($P(\text{cat}|\text{the}) = 2/3 > P(\text{dog}|\text{the}) = 1/3$),
  - Nhưng sau `"dog"`, hành động `"eats"` mang tính tất định tuyệt đối với $P(\text{eats}|\text{dog}) = 1.0$, trong khi sau `"cat"` xác suất bị chia sẻ cho cả `"likes"` và `"eats"` ($P(\text{eats}|\text{cat}) = 0.5$).
  - Tích: $\frac{2}{3} \times 0.5 = \frac{1}{3}$, đúng bằng $\frac{1}{3} \times 1.0 = \frac{1}{3}$. Do đó hai câu đạt cùng một mức xác suất tổng thể.

---

## 5. Bài tập suy luận trước khi smoothing (Mục 8 & 9)

Cho corpus:
- `"I like NLP"`
- `"I like AI"`
- `"I study NLP"`

Cần tính: $P(\text{AI}|\text{study})$.

### 5.1. Câu hỏi 1: Count của "study AI" là bao nhiêu?
- Trong corpus, từ `"study"` chỉ đi với `"NLP"` (`"study NLP"`), cụm `"study AI"` không hề xuất hiện lần nào.
- Vậy: $C(\text{study AI}) = 0$.

### 5.2. Câu hỏi 2: Xác suất MLE là bao nhiêu?
$$P_{\text{MLE}}(\text{AI}|\text{study}) = \frac{C(\text{study AI})}{C(\text{study})} = \frac{0}{1} = 0.0$$

### 5.3. Câu hỏi 3: Điều gì xảy ra khi tính xác suất câu chứa bigram này?
- Xét một câu hợp lý như `"I study AI"`.
- Xác suất bigram của câu là:
$$P(\text{"I study AI"}) = P(\text{I}) \times P(\text{study}|\text{I}) \times P(\text{AI}|\text{study})$$
- Vì $P(\text{AI}|\text{study}) = 0$, tích toàn bộ xác suất lập tức sụp đổ về $0$:
$$P(\text{"I study AI"}) = P(\text{I}) \times P(\text{study}|\text{I}) \times 0 = 0$$
- Đây chính là vấn đề số không (**zero-frequency problem** hay **zero probability**). Chỉ một n-gram chưa từng thấy cũng hủy diệt toàn bộ câu, biến một câu hoàn toàn tự nhiên thành một câu bị mô hình coi là bất khả thi.

### 5.4. Câu hỏi 4: Điều này có nghĩa mô hình "biết" rằng câu đó chắc chắn không thể xảy ra không?
- **Không.** Mô hình MLE không hề có tri thức ngôn ngữ thực sự; nó chỉ đếm tần suất cơ học trên một tập mẫu huấn luyện hữu hạn và rất nhỏ.
- Cụm từ `"study AI"` hoàn toàn có nghĩa và phổ biến trong thực tế. Việc $C(\text{study AI}) = 0$ đơn thuần là do dữ liệu huấn luyện chưa đủ lớn để bao phủ (sample sparsity), chứ không phản ánh bản chất của ngôn ngữ.
- **Quy tắc cốt lõi:**
$$\text{Không quan sát thấy trong tập huấn luyện} \neq \text{Xác suất thực tế bằng 0}$$

---

## 6. Bài tập tính Smoothing (Mục 10 & 11)

Cho:
- $C(\text{cat}) = 10$
- $C(\text{cat eats}) = 0$
- Kích thước từ vựng $V = 5$.

Công thức Laplace (Add-one) Smoothing cho Bigram:
$$P_{\text{Laplace}}(w|h) = \frac{C(h, w) + 1}{C(h) + V}$$

### 6.1. Trường hợp 1: $C(\text{cat eats}) = 0$
$$P_{\text{Laplace}}(\text{eats}|\text{cat}) = \frac{0 + 1}{10 + 5} = \frac{1}{15} \approx 0.0667$$

### 6.2. Trường hợp 2: Khi $C(\text{cat eats}) = 3$
- Giữ nguyên số lần quan sát ngữ cảnh $C(\text{cat}) = 10$:
$$P_{\text{Laplace}}(\text{eats}|\text{cat}) = \frac{3 + 1}{10 + 5} = \frac{4}{15} \approx 0.2667$$
- *(Nếu hiểu đề bài là $C(\text{cat})$ tăng thêm 3 lần tương ứng, thành $C(\text{cat}) = 13$: $P_{\text{Laplace}} = \frac{3+1}{13+5} = \frac{4}{18} = \frac{2}{9} \approx 0.2222$)*.

### 6.3. Trả lời câu hỏi:
**Smoothing đã thay đổi xác suất của những bigram khác như thế nào?**

**Trả lời:**
- Laplace smoothing cộng thêm $1$ vào tử số của tất cả các từ trong từ vựng, dẫn đến mẫu số phải tăng thêm một lượng bằng đúng kích thước từ vựng $V$ ($C(h) \to C(h) + V$).
- Vì tổng xác suất có điều kiện luôn phải bằng 1 ($\sum_{w \in V} P(w|h) = 1$), smoothing đã thực hiện việc **chiết khấu (discount)** bớt một phần khối lượng xác suất từ các bigram có tần số cao (seen n-grams, nơi $P_{\text{MLE}} = \frac{C}{C(h)} > \frac{C+1}{C(h)+V}$) để phân phối lại cho các bigram chưa từng xuất hiện (unseen n-grams, đưa xác suất từ $0$ lên $\frac{1}{C(h)+V}$).
- Nhược điểm của Add-one là với $V$ lớn, nó chuyển quá nhiều khối lượng xác suất sang các từ chưa thấy, làm méo mó phân phối xác suất thực nghiệm.

---

## 7. Bài tập tính Perplexity (Mục 17 & 18)

Perplexity ($PP$) của chuỗi từ $W = (w_1, w_2, \dots, w_N)$ gồm $N$ token:
$$PP(W) = P(W)^{-\frac{1}{N}} = \frac{1}{\sqrt[N]{P(W)}} = \exp\left(-\frac{1}{N} \sum_{i=1}^N \ln P(w_i | h_i)\right)$$

Cho chuỗi $W$ gồm $N = 3$ tokens với các xác suất:
- $P(w_1) = 0.5$
- $P(w_2|w_1) = 0.25$
- $P(w_3|w_2) = 0.5$

### 7.1. Tính $P(W)$ và $PP(W)$ ban đầu:
1. Xác suất của chuỗi:
$$P(W) = P(w_1) \times P(w_2|w_1) \times P(w_3|w_2) = 0.5 \times 0.25 \times 0.5 = 0.0625 = \frac{1}{16}$$

2. Perplexity của chuỗi:
$$PP(W) = P(W)^{-\frac{1}{3}} = \left(\frac{1}{16}\right)^{-\frac{1}{3}} = 16^{\frac{1}{3}} = \sqrt[3]{16} \approx 2.5198$$

### 7.2. Tính lại khi $P(w_2|w_1) = 0.1$:
1. Xác suất mới của chuỗi:
$$P(W) = 0.5 \times 0.1 \times 0.5 = 0.025 = \frac{1}{40}$$

2. Perplexity mới:
$$PP(W) = \left(\frac{1}{40}\right)^{-\frac{1}{3}} = 40^{\frac{1}{3}} = \sqrt[3]{40} \approx 3.41995 \approx 3.4200$$

### 7.3. Trả lời câu hỏi:
**Vì sao chỉ một xác suất nhỏ cũng có thể làm perplexity thay đổi đáng kể?**

**Trả lời:**
- Về mặt toán học, Perplexity là nghịch đảo của trung bình nhân xác suất các từ:
$$PP(W) = \frac{1}{\left(\prod_{i=1}^N P(w_i|h_i)\right)^{1/N}}$$
- Phép nhân chuỗi xác suất cực kỳ nhạy cảm với các thừa số có giá trị nhỏ. Khi $P(w_2|w_1)$ giảm từ $0.25$ xuống $0.1$ (giảm $2.5$ lần), toàn bộ tích xác suất $P(W)$ giảm đi $2.5$ lần, khiến Perplexity tăng theo tỉ lệ $\sqrt[3]{2.5} \approx 1.357$ (tăng từ $2.52$ lên $3.42$, tức tăng vọt gần **$35.7\%$**).
- Đặc biệt, nếu có bất kỳ từ nào có xác suất bằng $0$ (zero probability), mẫu số sẽ bằng $0$ và Perplexity sẽ bùng nổ lên **vô cùng ($\infty$)**.
- Perplexity đại diện cho số lượng lựa chọn tương đương (branching factor) mà mô hình cảm thấy phân vân tại mỗi bước. Khi một từ ít có khả năng xuất hiện xảy ra, sự "bối rối" của mô hình tăng vọt.
