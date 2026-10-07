# BÁO CÁO PHÂN TÍCH LỖI (ERROR ANALYSIS) — LAB 03

Môn học: Xử lý ngôn ngữ tự nhiên và ứng dụng  
Học kỳ: Học kỳ I - 2026  
Sinh viên: Nguyễn Hồng Quân - 23001921  
Giảng viên / TA: Phạm Ngọc Hải  
Chủ đề: Word Representations and Embeddings  

---

Theo yêu cầu tại Mục 25 của tài liệu `W3.pdf`, em chọn ra 3 trường hợp mô hình cho kết quả đúng và 3 trường hợp kết quả sai hoặc bất ngờ để phân tích nguyên nhân.

## 1. Ba cặp Similarity Đúng

### 1. `doctor` → `physician` (Word2Vec)
- **Quan sát được:** Trong Experiment 2, `physician` đứng đầu trong Top-5 từ gần `doctor` nhất với cosine similarity đạt **0.7949**. Trong Experiment 1, độ tương đồng giữa hai từ này cũng rất cao (từ 0.8355 đến 0.8940).
- **Kỳ vọng:** Hai từ này là từ đồng nghĩa trực tiếp đều chỉ bác sĩ y khoa, nên kỳ vọng điểm similarity phải cao nhất (đúng như Prediction 1).
- **Giải thích:** Cả hai từ đều đóng vai trò danh từ và thường xuyên xuất hiện chung với các từ ngữ cảnh y tế như `patient`, `hospital`, `treatment`, `medicine`. Do xuất hiện trong ngữ cảnh tương tự, mô hình Skip-gram đã kéo hai vector này về rất gần nhau.
- **Bằng chứng từ corpus:** Trong tập dữ liệu C4, hai từ này hay đi cùng nhau trong các câu như: *"consult your doctor or physician"*, *"board-certified physician / licensed doctor"*.
- **Nguyên nhân chính:** Tần suất xuất hiện vừa đủ, ngữ cảnh xuất hiện đồng nhất.

---

### 2. `football` → `baseball`, `basketball`, `soccer` (Word2Vec)
- **Quan sát được:** Trong Experiment 2, ba từ gần nhất của `football` lần lượt là: `baseball` (0.8323), `basketball` (0.8104), `soccer` (0.7679).
- **Kỳ vọng:** Đều là các môn thể thao chơi bóng đối kháng, nên vector của chúng phải nằm gần nhau.
- **Giải thích:** Các từ này có chung tập từ ngữ cảnh rất rõ ràng như các động từ `play`, `watch`, các danh từ giải đấu `game`, `team`, `season`, `league`. Vì vậy mô hình phân cụm nhóm thể thao này tách biệt hẳn khỏi các nhóm từ khác.
- **Bằng chứng từ corpus:** Các câu tin tức thể thao hay liệt kê: *"high school football, basketball, and baseball games"*.
- **Nguyên nhân chính:** Cửa sổ ngữ cảnh bao quát được cấu trúc liệt kê, dữ liệu thể thao trong bài viết khá đồng nhất.

---

### 3. `computer` → `desktop`, `device`, `laptop` (Word2Vec)
- **Quan sát được:** Các từ gần `computer` nhất gồm có: `desktop` (0.7840), `device` (0.7431), `laptop` (0.7316).
- **Kỳ vọng:** Các từ liên quan đến thiết bị máy tính và đồ công nghệ phải nằm sát nhau.
- **Giải thích:** Mô hình học được mối liên hệ giữa các thiết bị phần cứng, chúng cùng xuất hiện với các hành động như `install`, `connect`, `reboot`, `system`, `screen`.
- **Bằng chứng từ corpus:** Bài thảo luận công nghệ trong C4: *"booting from a different drive... clone to desktop/laptop"*.
- **Nguyên nhân chính:** Ngữ cảnh công nghệ thông tin rõ nét, ít bị phân tán nghĩa.

---

## 2. Ba cặp Similarity Sai hoặc Bất Ngờ

### 1. `car` → `while` (0.9744), `house` (0.9720), `after` (0.9671) trong Co-occurrence Matrix
- **Quan sát được:** Trong Experiment 1 (ma trận đồng xuất hiện với window = 5), các từ gần `car` nhất lại là các từ nối, giới từ như `while`, `after` và danh từ chung `house` với điểm rất cao (> 0.96).
- **Kỳ vọng:** `car` phải gần các từ về phương tiện như `vehicle`, `automobile`, `driver`, `road`.
- **Giải thích:** Đây là nhược điểm lớn của việc đếm tần suất thô (raw count). Các từ nối như `while`, `after` xuất hiện quá nhiều trong hầu hết mọi câu, khiến cho từ nào cũng vô tình đi cạnh chúng. Khi tính cosine similarity, các từ có tần suất cao này làm lệch góc vector của các từ thông thường.
- **Bằng chứng từ corpus:** Hầu như câu nào kể về việc đi xe cũng có từ nối: *"while driving the car"*, *"after getting out of the car"*.
- **Nguyên nhân chính:** Tần suất quá cao của các từ dừng (stop words) và việc thiếu cơ chế giảm trọng số (như PMI hoặc IDF).

---

### 2. `hospital` → `indiana` (0.6886), `cincinnati` (0.6782), `yale` (0.6694) (Word2Vec)
- **Quan sát được:** Trong Experiment 2, ngoại trừ từ `clinic` (0.7094) ở vị trí đầu, các vị trí tiếp theo gần `hospital` lại là tên các địa danh hoặc trường đại học tại Mỹ (`indiana`, `cincinnati`, `yale`, `alabama`).
- **Kỳ vọng:** `hospital` nên gần các từ y tế như `doctor`, `nurse`, `patient`, `medicine`.
- **Giải thích:** Trong tiếng Anh báo chí ở tập C4, tên các bệnh viện thường có dạng `[Địa danh / Đại học] + Hospital` (ví dụ: *Indiana University Hospital*, *Cincinnati Children's Hospital*, *Yale New Haven Hospital*). Vì các tên riêng này luôn đi liền kề từ `hospital`, mô hình Skip-gram coi chúng có độ liên kết rất mạnh.
- **Bằng chứng từ corpus:** Một bài viết trong kết quả Semantic Search có đoạn: *"the indiana hospital heart institute has one of the largest..."*.
- **Nguyên nhân chính:** Thiên lệch dữ liệu (domain/corpus bias) và tập mẫu 10.000 bài chưa đủ đa dạng để làm loãng tên riêng địa phương.

---

### 3. `doctor` → `spouse` (0.7228), `novak` (0.6359), `husband` (0.6327) (Word2Vec)
- **Quan sát được:** Từ `doctor` xuất hiện các từ chỉ quan hệ gia đình (`spouse`, `husband`) và một tên riêng lạ (`novak`) ở nhóm có similarity khá cao.
- **Kỳ vọng:** Các từ gần nhất nên là các chức danh y tế khác (`nurse`, `surgeon`) hoặc chuyên khoa y tế.
- **Giải thích:** Trong mẫu 10.000 văn bản có các bài viết về tiểu sử đời tư bác sĩ hoặc hướng dẫn bảo hiểm nhắc đến *"doctor and spouse / husband"*, và có bài viết nhắc nhiều lần đến một vị bác sĩ cụ thể tên là Novak (*"Dr. Novak"*).
- **Bằng chứng từ corpus:** *"coverage for the doctor, spouse, and dependent children"*.
- **Nguyên nhân chính:** Corpus mẫu nhỏ dẫn đến tương quan giả định ngẫu nhiên trong một vài bài viết cụ thể.

---

## 3. Nhận xét thêm từ thực nghiệm đối chiếu với Prediction

1. **Khi tăng Embedding Dimension (Experiment 4):**
   - Ở 50 chiều, độ tương đồng giữa `doctor` và `physician` là 0.7983. Nhưng khi tăng lên 300 chiều, điểm này bị tụt xuống còn 0.5575.
   - Điều này đúng như **Prediction 3**: Tập dữ liệu mẫu 10.000 bài viết là quá nhỏ so với không gian 300 chiều, mô hình bị phân tán tham số và học phải nhiễu, làm giảm độ tương đồng của hai từ đồng nghĩa.

2. **Khi thay đổi Context Window (Experiment 3):**
   - Cặp từ đồng nghĩa thay thế trực tiếp `doctor - physician` đạt điểm cao nhất ở cửa sổ hẹp (window = 2, điểm 0.7542).
   - Ngược lại, cặp từ liên quan về mặt chủ đề `doctor - hospital` lại tăng dần khi mở rộng cửa sổ (từ 0.4390 ở window 2 lên 0.5377 ở window 10).
   - Điều này đúng như **Prediction 2**: Cửa sổ nhỏ bắt tốt quan hệ đồng nghĩa trực tiếp, còn cửa sổ lớn bắt tốt quan hệ cùng chủ đề rộng.
