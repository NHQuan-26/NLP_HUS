# BÀI TẬP TÍNH TOÁN BẰNG TAY (PART B)

Môn học: Xử lý ngôn ngữ tự nhiên và ứng dụng  
Sinh viên: Chester  

---

## 1. Bài tập 1 — Count Vector

Cho 3 câu văn bản:
- D1: "cat eats fish"
- D2: "dog eats fish"
- D3: "cat likes fish"

Từ vựng sắp xếp theo bảng chữ cái:
[cat, dog, eats, fish, likes] (gồm 5 từ)

Vector đếm số lần xuất hiện của từng từ:
- D1: [1, 0, 1, 1, 0] (có cat, eats, fish)
- D2: [0, 1, 1, 1, 0] (có dog, eats, fish)
- D3: [1, 0, 0, 1, 1] (có cat, fish, likes)

---

## 2. Bài tập 2 — Term Frequency (TF)

Công thức tính TF trong bài giảng:
TF(t, d) = c(t, d) / tổng số từ trong d

Xét văn bản D1 ("cat eats fish"), tổng số từ là 3.
- TF(cat, D1) = 1/3 ≈ 0.3333
- TF(eats, D1) = 1/3 ≈ 0.3333
- TF(fish, D1) = 1/3 ≈ 0.3333
- TF(dog, D1) = 0
- TF(likes, D1) = 0

Kiểm tra lại tổng TF:
1/3 + 1/3 + 1/3 = 1.0 (đúng bằng 1).

---

## 3. Bài tập 3 — Inverse Document Frequency (IDF)

Tổng số văn bản N = 3.  
Số văn bản chứa từng từ (df):
- df(cat) = 2 (có trong D1, D3)
- df(dog) = 1 (có trong D2)
- df(eats) = 2 (có trong D1, D2)
- df(fish) = 3 (có trong cả 3 câu)
- df(likes) = 1 (có trong D3)

Sử dụng công thức IDF chuẩn trong slide:
IDF(t) = ln(N / df(t))

Tính cụ thể:
- IDF(cat) = ln(3/2) = ln(1.5) ≈ 0.4055
- IDF(dog) = ln(3/1) = ln(3) ≈ 1.0986
- IDF(eats) = ln(3/2) = ln(1.5) ≈ 0.4055
- IDF(fish) = ln(3/3) = ln(1) = 0.0
- IDF(likes) = ln(3/1) = ln(3) ≈ 1.0986

Trả lời câu hỏi:
Từ nào có IDF thấp nhất? Vì sao?
- Từ "fish" có IDF thấp nhất (bằng 0).
- Vì "fish" xuất hiện ở tất cả các văn bản (3 trên 3 câu). Khi một từ xuất hiện ở đâu cũng có thì nó không giúp ta phân biệt được văn bản này với văn bản khác, nên giá trị thông tin của nó bằng 0.

---

## 4. Bài tập 4 — TF-IDF

Công thức:
TF-IDF(t, d) = TF(t, d) * IDF(t)

Tính cho văn bản D1:
- TF-IDF(cat, D1) = (1/3) * ln(1.5) ≈ 0.1351
- TF-IDF(eats, D1) = (1/3) * ln(1.5) ≈ 0.1351
- TF-IDF(fish, D1) = (1/3) * 0 = 0.0

Trả lời câu hỏi:
Tại sao fish xuất hiện trong mọi document nhưng TF-IDF của nó bằng 0?
- Vì IDF của "fish" bằng 0 (ln(3/3) = 0). Khi nhân TF với 0 thì kết quả ra 0. Điều này hợp lý vì từ xuất hiện ở khắp mọi nơi sẽ bị coi như từ dừng (stopword) và bị loại bỏ trọng số.

---

## 5. Bài tập 5 — Cosine Similarity

Cho 2 vector:
x = [1, 1, 1]
y = [1, 1, 0]

Tính toán:
1. Tích vô hướng:
x . y = 1*1 + 1*1 + 1*0 = 2

2. Độ dài vector (chuẩn L2):
||x|| = sqrt(1^2 + 1^2 + 1^2) = sqrt(3) ≈ 1.732
||y|| = sqrt(1^2 + 1^2 + 0^2) = sqrt(2) ≈ 1.414

3. Cosine similarity:
cos(x, y) = 2 / (sqrt(3) * sqrt(2)) = 2 / sqrt(6) ≈ 0.8165

Trả lời câu hỏi:
Hai documents có 2 từ giống nhau trên 3 từ, tại sao cosine không bằng 2/3?
- Tỷ lệ 2/3 (≈ 0.667) là phép tính đếm phần trăm từ trùng nhau thông thường.
- Còn Cosine similarity là đo góc hình học giữa hai vector. Công thức chia cho căn bậc hai của tổng bình phương độ dài chứ không chia thẳng theo số lượng từ, nên giá trị góc tính ra là khoảng 0.8165.

---

## 6. Bài tập 6 — Dự đoán trước khi chạy code

Cho các văn bản:
- D1: "medical image classification"
- D2: "medical image analysis"
- D3: "natural language processing"
Query: "medical image classification"

Dự đoán:
1. Document nào có similarity cao nhất?
- D1, vì query giống y hệt câu D1 nên trùng 100% các từ, độ tương đồng đạt cao nhất (bằng 1.0).

2. Document nào có similarity thấp nhất?
- D3, vì không có từ nào trùng với query cả, độ tương đồng bằng 0.

3. Term nào có thể có IDF thấp?
- Hai từ "medical" và "image", vì chúng xuất hiện ở cả D1 và D2 (df = 2), nhiều hơn các từ khác chỉ xuất hiện 1 lần.

4. Nếu bỏ IDF chỉ dùng Count vector thì ranking có đổi không?
- Thứ tự ranking vẫn giữ nguyên là D1 > D2 > D3. Vì D1 trùng cả 3 từ, D2 trùng 2 từ, còn D3 không trùng từ nào. Tuy nhiên điểm số tương đối giữa các câu sẽ khác nhau một chút.
