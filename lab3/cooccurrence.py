import numpy as np
from collections import Counter
from scipy.sparse import csr_matrix
from typing import List, Tuple, Dict, Optional

class MyCooccurrence:
    """
    Xây dựng ma trận đồng xuất hiện từ - ngữ cảnh (Word-Context Co-occurrence Matrix)
    và tính toán độ tương đồng Cosine giữa các vector từ.
    """
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.matrix: Optional[csr_matrix] = None
        self.vocab_size: int = 0

    def word2id(self) -> Dict[str, int]:
        return self.word_to_id

    def id2word(self) -> Dict[int, str]:
        return self.id_to_word

    def build_vocabulary(self, corpus: List[List[str]], min_count: int = 1) -> None:
        """
        Tạo từ điển từ corpus, lọc các từ có tần suất xuất hiện >= min_count.
        """
        freq_counter = Counter()
        for doc in corpus:
            freq_counter.update(doc)

        # Lọc từ theo tần suất tối thiểu và sắp xếp giảm dần theo số lần xuất hiện
        filtered_words = [w for w, freq in freq_counter.items() if freq >= min_count]
        filtered_words.sort(key=lambda w: (-freq_counter[w], w))

        # Lưu chỉ số mapping hai chiều
        self.word_to_id = {word: idx for idx, word in enumerate(filtered_words)}
        self.id_to_word = {idx: word for idx, word in enumerate(filtered_words)}
        self.vocab_size = len(filtered_words)

    def build_cooccurrence_matrix(self, corpus: List[List[str]], window_size: int = 2) -> None:
        """
        Đếm số lần đồng xuất hiện của các cặp từ trong cửa sổ ngữ cảnh window_size
        và tạo ma trận thưa csr_matrix.
        """
        if self.vocab_size == 0:
            raise ValueError("Từ điển đang trống. Hãy chạy build_vocabulary() trước.")

        pair_counts = Counter()

        for tokens in corpus:
            # Chuyển các từ hợp lệ trong văn bản thành ID
            token_ids = [self.word_to_id[t] for t in tokens if t in self.word_to_id]
            doc_len = len(token_ids)

            for i, center_id in enumerate(token_ids):
                start_idx = max(0, i - window_size)
                end_idx = min(doc_len, i + window_size + 1)

                for j in range(start_idx, end_idx):
                    if i != j:
                        context_id = token_ids[j]
                        pair_counts[(center_id, context_id)] += 1

        if not pair_counts:
            self.matrix = csr_matrix((self.vocab_size, self.vocab_size), dtype=np.float32)
            return

        # Tạo ma trận thưa CSR từ danh sách đếm
        rows, cols = zip(*pair_counts.keys())
        values = [float(v) for v in pair_counts.values()]

        self.matrix = csr_matrix(
            (values, (rows, cols)),
            shape=(self.vocab_size, self.vocab_size),
            dtype=np.float32
        )

    @staticmethod
    def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
        """
        Tính độ tương đồng Cosine giữa 2 vector u và v: cos(u, v) = (u . v) / (||u|| * ||v||)
        """
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        if norm_u == 0.0 or norm_v == 0.0:
            return 0.0
        return float(np.dot(u, v) / (norm_u * norm_v))

    def most_similar(self, query_word: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Tìm top_k từ có vector biểu diễn gần nhất với query_word theo Cosine similarity.
        """
        if query_word not in self.word_to_id:
            raise KeyError(f"Từ '{query_word}' không có trong từ vựng.")

        target_idx = self.word_to_id[query_word]
        target_vec = self.matrix[target_idx].toarray().ravel()
        target_norm = np.linalg.norm(target_vec)

        if target_norm == 0.0:
            return []

        # Tích vô hướng vector mục tiêu với tất cả các hàng
        dot_products = self.matrix.dot(target_vec)

        # Tính chuẩn L2 của từng hàng trong ma trận
        squared_matrix = self.matrix.multiply(self.matrix)
        row_norms = np.sqrt(np.array(squared_matrix.sum(axis=1)).ravel())

        # Cosine similarity toàn ma trận (thêm epsilon tránh chia cho 0)
        sim_scores = dot_products / (target_norm * row_norms + 1e-9)

        # Lấy các chỉ số có similarity cao nhất
        sorted_indices = np.argsort(-sim_scores)

        results = []
        for idx in sorted_indices:
            if idx == target_idx:
                continue
            results.append((self.id_to_word[idx], float(sim_scores[idx])))
            if len(results) >= top_k:
                break

        return results

