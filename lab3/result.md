# KẾT QUẢ THỰC NGHIỆM (EXPERIMENTAL RESULTS) — LAB 03

Môn học: Xử lý ngôn ngữ tự nhiên và ứng dụng  
Học kỳ: Học kỳ I - 2026  
Sinh viên: Nguyễn Hồng Quân - 23001921  
Giảng viên / TA: Phạm Ngọc Hải  
Chủ đề: Word Representations and Embeddings  

---

## 1. Thực nghiệm 1 — Ma trận đồng xuất hiện (Co-occurrence Matrix)

### 1.1. Thống kê ma trận theo kích thước cửa sổ (Window Size)

| Chỉ số thống kê | Cửa sổ (Window = 1) | Cửa sổ (Window = 2) | Cửa sổ (Window = 5) |
| :--- | :---: | :---: | :---: |
| **Kích thước từ vựng (Vocab)** | 29.381 từ | 29.381 từ | 29.381 từ |
| **Kích thước ma trận** | 29.381 × 29.381 | 29.381 × 29.381 | 29.381 × 29.381 |
| **Số phần tử khác 0 (nnz)** | 2.099.883 | 4.078.885 | 8.885.616 |
| **Độ thưa (Sparsity)** | 99.7567% | 99.5275% | 98.9707% |
| **Cosine (`doctor` - `physician`)** | 0.8355 | 0.8487 | 0.8940 |

*Nhận xét ngắn:* Khi tăng window size từ 1 lên 5, số lượng phần tử khác 0 tăng lên hơn 4 lần (từ 2 triệu lên gần 8.9 triệu), làm độ thưa giảm nhẹ từ 99.75% xuống 98.97%. Độ tương đồng giữa `doctor` và `physician` cũng tăng nhẹ từ 0.8355 lên 0.8940.

---

### 1.2. Top-5 từ có vector gần nhất (Co-occurrence Model, Window = 5)

| Hạng | `doctor` | `hospital` | `car` | `computer` | `food` |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **1** | child (0.9456) | house (0.9404) | while (0.9744) | business (0.9407) | music (0.9656) |
| **2** | vehicle (0.9248) | public (0.9403) | house (0.9720) | home (0.9370) | water (0.9654) |
| **3** | plan (0.9242) | community (0.9373) | company (0.9693) | budget (0.9349) | local (0.9634) |
| **4** | or (0.9220) | office (0.9360) | after (0.9671) | account (0.9333) | other (0.9630) |
| **5** | home (0.9209) | school (0.9347) | game (0.9667) | journey (0.9327) | performance (0.9629) |

---

## 2. Thực nghiệm 2 — Mô hình Word2Vec (Dense Embeddings)

Cấu hình huấn luyện mô hình: `vector_size=100`, `window=5`, `min_count=5`, `epochs=10`, `sg=1` (Skip-gram), `workers=4`.

### Top-5 từ tương đồng nhất (`model.wv.most_similar`)

| Từ kiểm thử | Hạng 1 | Hạng 2 | Hạng 3 | Hạng 4 | Hạng 5 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`doctor`** | physician (0.7949) | spouse (0.7228) | child (0.7066) | novak (0.6359) | husband (0.6327) |
| **`hospital`** | clinic (0.7094) | indiana (0.6886) | cincinnati (0.6782) | yale (0.6694) | alabama (0.6691) |
| **`patient`** | patients (0.6474) | treatment (0.6458) | medication (0.6290) | diagnosis (0.6044) | child (0.5931) |
| **`disease`** | diseases (0.8057) | diabetes (0.8054) | syndrome (0.8039) | chronic (0.8030) | infection (0.7974) |
| **`computer`** | desktop (0.7840) | browser (0.7522) | device (0.7431) | laptop (0.7316) | server (0.6991) |
| **`football`** | baseball (0.8323) | basketball (0.8104) | soccer (0.7679) | championship (0.7237) | club (0.7149) |
| **`banana`** | peppers (0.8969) | onions (0.8944) | cabbage (0.8928) | mushrooms (0.8926) | pudding (0.8900) |

*Nhận xét ngắn:* Word2Vec cho kết quả ngữ nghĩa tự nhiên hơn hẳn ma trận co-occurrence thô. `doctor` đứng sát `physician`, `disease` gắn liền với các bệnh như `diabetes`, `football` phân cụm rất chuẩn với `baseball`, `basketball`, `soccer`.

---

## 3. Thực nghiệm 3 — Ảnh hưởng của Context Window

Độ tương đồng giữa các cặp từ khi thay đổi kích thước cửa sổ:

| STT | Cặp từ kiểm tra | Window = 2 | Window = 5 | Window = 10 |
| :-: | :--- | :---: | :---: | :---: |
| 1 | `doctor` - `physician` | **0.7542** | 0.6950 | 0.7472 |
| 2 | `doctor` - `hospital` | 0.4390 | 0.4876 | **0.5377** |
| 3 | `cat` - `dog` | 0.5619 | **0.6868** | 0.6441 |

*Nhận xét ngắn:* 
- Cặp từ đồng nghĩa trực tiếp `doctor - physician` có similarity cao nhất khi cửa sổ hẹp (Window = 2).
- Cặp từ liên quan theo chủ đề `doctor - hospital` có similarity tăng dần khi mở rộng cửa sổ (Window = 10 đạt 0.5377).

---

## 4. Thực nghiệm 4 — Ảnh hưởng của Embedding Dimension

Đánh giá thời gian huấn luyện và chất lượng biểu diễn qua các chiều vector khác nhau:

| STT | Số chiều (Dimension) | Thời gian chạy (giây) | Kích thước ma trận vector | Cosine (`doctor` - `physician`) | Phép tính suy luận (`king` - `man` + `woman`) |
| :-: | :---: | :---: | :---: | :---: | :---: |
| 1 | **50** | 136.70 s | 1.469.050 | **0.7983** | `queen` |
| 2 | **100** | 133.61 s | 2.938.100 | 0.7595 | `queen` |
| 3 | **300** | 196.40 s | 8.814.300 | 0.5575 | `queen` |

*Nhận xét ngắn:* Cả 3 cấu hình đều giải đúng phép tương tự từ `king - man + woman = queen`. Tuy nhiên ở 300 chiều, thời gian chạy lâu hơn và độ tương đồng giữa `doctor` và `physician` bị giảm sút (từ 0.7983 xuống 0.5575) do dữ liệu 10.000 bài viết bị thưa so với không gian 300 chiều.

---

## 5. Ứng dụng — Tìm kiếm ngữ nghĩa (Semantic Search)

Thực hiện tìm kiếm tài liệu bằng vector trung bình của câu (Average Word Embedding) cho 3 câu truy vấn:

### Truy vấn 1: `medical treatment`
- **Top 1 (Điểm: 0.6350):**
  > a phase ii pilot study to evaluate use of intravenous lidocaine for opioid refractory pain in cancer patients kerala india status of cancer pain relief and palliative care nurse moral distress and cancer pain management an ethnography of oncology nurses in india
- **Top 2 (Điểm: 0.5933):**
  > not all ultrasound systems are equal created equal truffles vein specialists has philips affinity ultrasound systems the state of the art cutting edge ultrasound system along with award winning experienced vascular staff enables truffles vein specialists to provide the highest level of vein diagnosis and treatment...
- **Top 3 (Điểm: 0.5858):**
  > on january 28 2016 food and drug administration fda has approved merck and co manufactured zepatier a new oral treatment for adult patients with chronic hepatitis c hcv virus genotype 1 and 4 infections...

### Truy vấn 2: `delicious food recipe`
- **Top 1 (Điểm: 0.7173):**
  > located on malop street geelong serving delicious fresh food coffee the restaurant printer is offline
- **Top 2 (Điểm: 0.6955):**
  > gem home made meal simple fast lesson chinese dishes first dish chinese wedding toronto video photo services how to cook customize your mama shrimp thai tum yum instant noodles
- **Top 3 (Điểm: 0.6858):**
  > spicy kung pao noodles with shrimp is inspired by the famous szechuan kung pao chicken the spicy savory sweet and tangy sauce is one of the elements that makes this dish addicting making stir fried noodles with this sauce is no exception

### Truy vấn 3: `implement linear regression`
- **Top 1 (Điểm: 0.7070):**
  > with a combination of analytical techniques we can determine or confirm the solid state molecular structure for a chemical entity these techniques include mass spectrometry nmr spectroscopy infrared spectroscopy elemental analysis and single crystal x ray diffraction...
- **Top 2 (Điểm: 0.6934):**
  > grok is an ai operations aiops platform that proactively resolves it incidents using machine intelligence and automation it senses behaviors that lead to downtime using anomaly detection then triggers actions based on those insights...
