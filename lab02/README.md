# LAB 02 — LANGUAGE MODELS: N-GRAM, SMOOTHING & PERPLEXITY

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Học kỳ:** Học kỳ I - 2026  
**Sinh viên:** Nguyễn Hồng Quân - 23001921  
**Giảng viên / TA:** Phạm Ngọc Hải  
**Hạn nộp:** 23:59 - 30/09/2026  

---

## 1. Giới thiệu tổng quan (Overview)

Trong bài thực hành LAB 02, chúng ta chuyển từ biểu diễn văn bản không gian vector phục vụ tìm kiếm (LAB 01) sang **Language Modeling (Mô hình hóa ngôn ngữ)** dưới góc nhìn xác suất. 

Mục tiêu chính là tự xây dựng một hệ thống **N-gram Language Model** từ đầu (không sử dụng thư viện mô hình ngôn ngữ có sẵn), cài đặt các kỹ thuật ước lượng hợp lý cực đại (**MLE**), kỹ thuật làm mịn (**Laplace / Add-one Smoothing**), xử lý ổn định số học qua **Log-probability**, đánh giá chất lượng bằng chỉ số **Perplexity**, và ứng dụng vào bài toán **Dự đoán từ tiếp theo (Next-word Prediction)** và **Xếp hạng câu (Sentence Ranking)**.

---

## 2. Cấu trúc thư mục nộp bài (Deliverables Structure)

Theo đúng quy chuẩn tại Mục 27 của tài liệu `W2.pdf`:

```text
lab02/
├── README.md               # Bản hướng dẫn tổng quan và tóm tắt kết quả thực nghiệm
├── calculations.md         # Lời giải chi tiết các bài tập tính toán bằng tay (Mục 7, 9, 11, 18)
├── prediction.md           # 5 dự đoán độc lập trước khi chạy thực nghiệm (Mục 12)
├── ngram_lm.py             # Module thư viện Core Implementation N-gram Language Model
├── run_experiments.py      # Kịch bản Python tự động thực thi toàn bộ pipeline thực nghiệm
├── experiments.ipynb       # Jupyter Notebook tương tác với biểu đồ và kết quả chạy chi tiết
├── results.csv             # Tệp CSV tổng hợp kết quả thống kê corpus và Perplexity các mô hình
├── error_analysis.md       # Báo cáo phân tích chuyên sâu 2 ca dự đoán đúng và 2 ca dự đoán sai
├── reflection.md           # Trả lời 7 câu hỏi suy ngẫm, AI statement, kiểm tra miệng và liên hệ LAB 01
├── ngram_distribution.png  # Biểu đồ phân phối tần suất Unigram/Bigram/Trigram (Rank vs Frequency)
└── W2.pdf                  # Đề bài gốc của phòng thực hành LAB 02
```

---

## 3. Tóm tắt kết quả thực nghiệm chính (Key Experimental Findings)

### 3.1. Thống kê kho ngữ liệu C4 (15.000 văn bản)
- **Số câu:** 327.317 câu | **Số tokens:** 5.560.592 tokens | **Từ vựng ($|V|$):** 130.136 từ.
- **Hiện tượng đuôi dài (Long-tail / Zipf's Law):**
  - Unigram: 61.395 từ xuất hiện duy nhất 1 lần (**47.18%**).
  - Bigram: 1.187.009 cụm xuất hiện duy nhất 1 lần (**72.70%**).
  - Trigram: 3.046.773 cụm xuất hiện duy nhất 1 lần (**87.58%**).
- *Kết luận:* Khi tăng $n$, độ thưa dữ liệu (sparsity) bùng nổ khủng khiếp.

### 3.2. So sánh MLE vs Laplace Smoothing (Perplexity)

| Model | N-gram | Smoothing | Train PPL | Valid PPL | Test PPL |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Bigram MLE** | 2 | None (MLE) | **174.87** | $\infty$ | $\infty$ |
| **Bigram Laplace** | 2 | Add-one | 3,391.81 | **4,020.89** | **3,994.95** |
| **Trigram MLE** | 3 | None (MLE) | $\infty$ | $\infty$ | $\infty$ |
| **Trigram Laplace** | 3 | Add-one | 16,260.47 | 25,354.57 | 25,297.26 |
| **Unigram Laplace** | 1 | Add-one | 1,749.23 | **1,632.36** | **1,631.44** |

- **Hiện tượng Zero Probability:** Mô hình MLE trên tập Valid/Test đều nhận PPL = $\infty$ do chỉ cần gặp một n-gram chưa từng thấy (unseen) thì xác suất chuỗi rơi về 0.
- **Hiệu quả của Laplace Smoothing:** Giải quyết triệt để lỗi zero probability, đưa PPL về giá trị hữu hạn.
- **Context length có luôn tốt hơn không?** KHÔNG. Trigram Laplace có PPL cao hơn Bigram trên dữ liệu kiểm thử do Add-one smoothing phạt quá nặng các context chưa từng xuất hiện trong không gian từ vựng lớn ($|V| > 130.000$).

### 3.3. Ứng dụng Dự đoán từ tiếp theo & Xếp hạng câu
- **Next-word Prediction:** Dự đoán chính xác các cụm từ kỹ thuật chặt chẽ như `"natural language" \to "processing"` và `"deep neural" \to "network"`.
- **Sentence Ranking:** Khi chuẩn hóa bằng Perplexity, mô hình xếp hạng các câu tự nhiên, có nghĩa (`"machine learning is useful for nlp"` - PPL: 26,648) cao hơn vượt trội so với các câu vô nghĩa (`"machine learning banana computer quickly"` - PPL: 61,274).

---

## 4. Hướng dẫn chạy và tái hiện thực nghiệm (How to Reproduce)

### Yêu cầu môi trường
- Python 3.8+
- Thư viện: `numpy`, `pandas`, `matplotlib` (đã có sẵn trong môi trường chuẩn)

### Lệnh thực thi nhanh toàn bộ thực nghiệm
Tại thư mục gốc của repository hoặc trong thư mục `lab02/`:

```bash
# 1. Chạy kịch bản tự động thực nghiệm (xuất kết quả ra results.csv và biểu đồ png)
python3 lab02/run_experiments.py

# 2. Chạy test suite kiểm tra tính đúng đắn của thư viện N-gram LM
python3 -c "
from lab02.ngram_lm import NGramLanguageModel, build_vocabulary
corpus = ['the cat eats fish', 'the cat likes fish', 'the dog eats meat']
m = NGramLanguageModel(n=2, smoothing='mle').fit(corpus)
assert round(m.sentence_probability('the cat eats fish'), 5) == round(1/24, 5)
print('ALL VERIFICATION CHECKS PASSED!')
"
```

### Mở Jupyter Notebook
Khởi động Jupyter và mở file [lab02/experiments.ipynb](file:///home/chester/Documents/HUS_NLP/lab02/experiments.ipynb) để tương tác trực tiếp với các khối mã nguồn và biểu đồ phân phối.

---

## 5. Bảng kiểm tự đánh giá theo Rubric (Peer Assessment Checklist)

Đối chiếu với Mục 28 & 29 của `W2.pdf`:

- [x] **Calculation (10 điểm):**
  - [x] Unigram probability đúng ($N=12, V=7, \sum P = 1.0$)
  - [x] Bigram probability đúng ($P(\text{cat}|\text{the})=2/3, P(\text{dog}|\text{the})=1/3$)
  - [x] Sentence probability đúng ($P(S_1) = 1/24 \approx 0.04167$)
  - [x] Sentence ranking đúng ($P(S_1) = P(S_2)$)
  - [x] Zero probability & Smoothing calculations đúng ($P = 1/15$)
  - [x] Perplexity đúng ($PP = 2.5198 \to 3.4200$)
- [x] **Prediction (10 điểm):** Đầy đủ 5 dự đoán trước thực nghiệm kèm lý do và độ tin cậy.
- [x] **Core Implementation (20 điểm):** Tự xây dựng class `NGramLanguageModel` từ đầu với đầy đủ các hàm đếm n-gram, tính xác suất điều kiện, log-prob tránh underflow, và perplexity.
- [x] **Smoothing Experiment (15 điểm):** So sánh đầy đủ MLE vs Laplace trên Train/Valid/Test set.
- [x] **Perplexity Evaluation (15 điểm):** Đánh giá Unigram, Bigram, Trigram và phân tích sâu sắc context length.
- [x] **Application (10 điểm):** Đầy đủ 5 context Next-word prediction và xếp hạng câu ứng viên.
- [x] **Error Analysis (10 điểm):** Phân tích kỹ lưỡng 2 ca đúng và 2 ca sai với nguyên nhân hệ thống.
- [x] **Reflection (5 điểm):** Trả lời trọn vẹn 7 câu hỏi suy ngẫm và liên hệ mở rộng sang Neural LM / Transformer.
- [x] **Individual Learning Check (5 điểm):** Trả lời chính xác và tự tin toàn bộ các câu hỏi phỏng vấn miệng.
