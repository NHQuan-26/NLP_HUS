

from typing import List, Dict, Tuple, Union
import math
import numpy as np


def tokenize(text: str) -> List[str]:
    """Basic whitespace and lowercase tokenizer for text strings."""
    return text.lower().strip().split()


def build_vocabulary(documents: List[Union[str, List[str]]]) -> List[str]:
    """
    Build a sorted vocabulary of unique terms from a collection of documents.

    Args:
        documents: A list of raw text strings or tokenized documents.

    Returns:
        A sorted list of unique terms (vocabulary).
    """
    vocab_set = set()
    for doc in documents:
        tokens = tokenize(doc) if isinstance(doc, str) else [t.lower() for t in doc]
        vocab_set.update(tokens)
    return sorted(list(vocab_set))


def compute_counts(
    documents: List[Union[str, List[str]]], vocabulary: List[str]
) -> np.ndarray:
    """
    Compute term count vectors for all documents given a vocabulary.

    Args:
        documents: List of documents (strings or token lists).
        vocabulary: List of unique terms representing feature dimensions.

    Returns:
        A 2D numpy array of shape (N, V) containing term counts c(t, d).
    """
    N = len(documents)
    V = len(vocabulary)
    vocab_to_idx = {term: idx for idx, term in enumerate(vocabulary)}
    count_matrix = np.zeros((N, V), dtype=np.float64)

    for doc_idx, doc in enumerate(documents):
        tokens = tokenize(doc) if isinstance(doc, str) else [t.lower() for t in doc]
        for token in tokens:
            if token in vocab_to_idx:
                count_matrix[doc_idx, vocab_to_idx[token]] += 1.0

    return count_matrix


def compute_tf(count_matrix: np.ndarray) -> np.ndarray:
    """
    Compute Term Frequency (TF) according to lecture definition:
        tf(t, d) = c(t, d) / sum_{t'} c(t', d)

    Args:
        count_matrix: 2D numpy array of shape (N, V) containing term counts.

    Returns:
        A 2D numpy array of shape (N, V) with normalized term frequencies.
    """
    doc_lengths = np.sum(count_matrix, axis=1, keepdims=True)
    # Avoid division by zero for empty documents
    tf_matrix = np.divide(
        count_matrix,
        doc_lengths,
        out=np.zeros_like(count_matrix, dtype=np.float64),
        where=doc_lengths > 0,
    )
    return tf_matrix


def compute_idf(
    documents: List[Union[str, List[str]]],
    vocabulary: List[str],
    smooth: bool = False,
    add_one: bool = False,
) -> np.ndarray:
    """
    Compute Inverse Document Frequency (IDF) for all terms in the vocabulary.

    Standard formula (Lecture & Lab 01 Section 4.4):
        idf(t) = ln(N / df(t))

    Smoothed formula (scikit-learn convention):
        idf(t) = ln((1 + N) / (1 + df(t))) + 1

    Args:
        documents: List of documents.
        vocabulary: Vocabulary list of length V.
        smooth: Whether to use Laplace smoothing (+1 in num and den).
        add_one: Whether to add 1 to the final idf (used in scikit-learn).

    Returns:
        A 1D numpy array of shape (V,) containing IDF weights.
    """
    N = len(documents)
    V = len(vocabulary)
    vocab_to_idx = {term: idx for idx, term in enumerate(vocabulary)}
    df = np.zeros(V, dtype=np.float64)

    for doc in documents:
        tokens = tokenize(doc) if isinstance(doc, str) else [t.lower() for t in doc]
        seen_terms = set(tokens)
        for term in seen_terms:
            if term in vocab_to_idx:
                df[vocab_to_idx[term]] += 1.0

    if smooth:
        # scikit-learn standard: ln((1 + N) / (1 + df)) + 1
        idf = np.log((1.0 + N) / (1.0 + df))
        if add_one:
            idf += 1.0
    else:
        # Standard NLP formula: ln(N / df)
        # For df == 0, safe handling
        idf = np.divide(
            float(N),
            df,
            out=np.zeros_like(df, dtype=np.float64),
            where=df > 0,
        )
        idf = np.log(idf, out=np.zeros_like(idf), where=idf > 0)

    return idf


def compute_tfidf(
    tf_matrix: np.ndarray,
    idf_vector: np.ndarray,
    norm: Union[str, None] = None,
) -> np.ndarray:
    """
    Compute TF-IDF representation matrix:
        tfidf(t, d) = tf(t, d) * idf(t)

    Args:
        tf_matrix: 2D numpy array of shape (N, V).
        idf_vector: 1D numpy array of shape (V,).
        norm: Optional normalization, e.g., 'l2' (cosine normalization).

    Returns:
        A 2D numpy array of shape (N, V) containing TF-IDF weights.
    """
    tfidf = tf_matrix * idf_vector  # Element-wise broadcasting across rows

    if norm == "l2":
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        tfidf = np.divide(
            tfidf,
            norms,
            out=np.zeros_like(tfidf),
            where=norms > 0,
        )

    return tfidf


def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Compute Cosine Similarity between two vectors:
        cos(x, y) = (x^T y) / (||x||_2 * ||y||_2)

    Args:
        vector1: 1D numpy array.
        vector2: 1D numpy array.

    Returns:
        Cosine similarity as a float in [-1.0, 1.0] (typically [0.0, 1.0] for non-negative TF-IDF).
    """
    v1 = np.asarray(vector1, dtype=np.float64).flatten()
    v2 = np.asarray(vector2, dtype=np.float64).flatten()

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    dot = np.dot(v1, v2)
    return float(dot / (norm1 * norm2))


# ==============================================================================
# Unit Tests & Verification (Part E - Section 8.4 & 8.5)
# ==============================================================================

def run_unit_tests():
    """Execute unit tests on the toy corpus from Lab 01 Section 8.3."""
    print("=" * 70)
    print("RUNNING UNIT TESTS ON TOY CORPUS (Part E)")
    print("=" * 70)

    # 1. Toy Corpus Definition
    D1 = "cat eats fish"
    D2 = "dog eats fish"
    D3 = "cat likes fish"
    corpus = [D1, D2, D3]

    # 2. Test Vocabulary Building
    expected_vocab = ["cat", "dog", "eats", "fish", "likes"]
    vocab = build_vocabulary(corpus)
    assert vocab == expected_vocab, f"Vocab mismatch: {vocab} != {expected_vocab}"
    print("[PASS] build_vocabulary():", vocab)

    # 3. Test Count Matrix
    # D1: [1, 0, 1, 1, 0]
    # D2: [0, 1, 1, 1, 0]
    # D3: [1, 0, 0, 1, 1]
    expected_counts = np.array([
        [1.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 1.0],
    ])
    counts = compute_counts(corpus, vocab)
    assert np.allclose(counts, expected_counts), f"Count mismatch:\n{counts}\nvs\n{expected_counts}"
    print("[PASS] compute_counts(): shape", counts.shape)

    # 4. Test Term Frequency (TF)
    # In D1, length = 3, so TF(cat) = TF(eats) = TF(fish) = 1/3
    tf = compute_tf(counts)
    expected_tf_D1 = np.array([1 / 3, 0.0, 1 / 3, 1 / 3, 0.0])
    assert np.allclose(tf[0], expected_tf_D1), f"TF mismatch: {tf[0]}"
    # Check sum of TF for each document is exactly 1.0
    for i in range(len(corpus)):
        assert abs(np.sum(tf[i]) - 1.0) < 1e-9, f"TF sum != 1 in doc {i}: {np.sum(tf[i])}"
    print("[PASS] compute_tf(): doc 0 sum =", np.sum(tf[0]))

    # 5. Test Inverse Document Frequency (IDF) - Standard Lecture Formula
    # df: cat=2, dog=1, eats=2, fish=3, likes=1
    # idf = ln(3 / df)
    # idf(cat) = ln(1.5) ≈ 0.4054651
    # idf(dog) = ln(3.0) ≈ 1.0986123
    # idf(eats) = ln(1.5) ≈ 0.4054651
    # idf(fish) = ln(1.0) = 0.0
    # idf(likes) = ln(3.0) ≈ 1.0986123
    idf_std = compute_idf(corpus, vocab, smooth=False)
    expected_idf_cat = math.log(3.0 / 2.0)
    expected_idf_fish = 0.0
    assert abs(idf_std[0] - expected_idf_cat) < 1e-7, f"IDF cat mismatch: {idf_std[0]}"
    assert abs(idf_std[3] - expected_idf_fish) < 1e-7, f"IDF fish mismatch: {idf_std[3]}"
    print("[PASS] compute_idf() standard: idf(cat) =", idf_std[0], ", idf(fish) =", idf_std[3])

    # 6. Test TF-IDF Computation
    tfidf_std = compute_tfidf(tf, idf_std)
    # For D1, tfidf(fish) must be 0 because idf(fish) = 0
    assert abs(tfidf_std[0, 3] - 0.0) < 1e-9, f"TFIDF fish mismatch: {tfidf_std[0, 3]}"
    assert abs(tfidf_std[0, 0] - (1 / 3) * expected_idf_cat) < 1e-7
    print("[PASS] compute_tfidf() standard: tfidf(cat, D1) =", tfidf_std[0, 0], ", tfidf(fish, D1) =", tfidf_std[0, 3])

    # 7. Test Cosine Similarity
    x = np.array([1.0, 1.0, 1.0])
    y = np.array([1.0, 1.0, 0.0])
    expected_cos = 2.0 / math.sqrt(6.0)
    cos_val = cosine_similarity(x, y)
    assert abs(cos_val - expected_cos) < 1e-7, f"Cosine similarity mismatch: {cos_val} != {expected_cos}"
    print("[PASS] cosine_similarity([1,1,1], [1,1,0]) =", cos_val, "≈ 2/sqrt(6)")

    # Test Cosine Similarity on TF-IDF vectors
    sim_1_2 = cosine_similarity(tfidf_std[0], tfidf_std[1])
    sim_1_3 = cosine_similarity(tfidf_std[0], tfidf_std[2])
    print(f"[PASS] Corpus TF-IDF similarity D1-D2: {sim_1_2:.4f}, D1-D3: {sim_1_3:.4f}")

    print("\nALL UNIT TESTS PASSED SUCCESSFULLY!\n")


def compare_with_scikit_learn():
    """
    Comparison with scikit-learn reference implementation (Section 8.5).
    Demonstrates and explains the exact differences in conventions:
    1. Count vs normalized TF.
    2. IDF smoothing: ln((1+N)/(1+df)) + 1 vs ln(N/df).
    3. L2 document vector normalization.
    """
    print("=" * 70)
    print("COMPARISON: STUDENT IMPLEMENTATION vs SCIKIT-LEARN REFERENCE")
    print("=" * 70)

    from sklearn.feature_extraction.text import TfidfVectorizer

    corpus = ["cat eats fish", "dog eats fish", "cat likes fish"]
    vocab = build_vocabulary(corpus)

    # 1. Scikit-learn default TfidfVectorizer
    sk_vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    sk_tfidf = sk_vec.fit_transform(corpus).toarray()
    sk_vocab = sk_vec.get_feature_names_out().tolist()

    # 2. Student implementation with scikit-learn convention
    # In scikit-learn, TF is raw counts, IDF is ln((1+N)/(1+df)) + 1, followed by L2 norm
    counts = compute_counts(corpus, vocab)
    idf_sklearn_convention = compute_idf(corpus, vocab, smooth=True, add_one=True)
    student_sklearn_mode = compute_tfidf(counts, idf_sklearn_convention, norm="l2")

    print("Vocabulary match:", vocab == sk_vocab)
    print(f"Scikit-learn IDF values:\n  {sk_vec.idf_}")
    print(f"Student smoothed IDF values:\n  {idf_sklearn_convention}")
    assert np.allclose(sk_vec.idf_, idf_sklearn_convention), "IDF convention mismatch!"
    print("[MATCH] IDF values match exactly with scikit-learn!")

    print(f"\nScikit-learn TF-IDF matrix (D1):\n  {sk_tfidf[0]}")
    print(f"Student TF-IDF matrix (D1 with sklearn mode):\n  {student_sklearn_mode[0]}")
    assert np.allclose(sk_tfidf, student_sklearn_mode), "TF-IDF matrix mismatch in sklearn mode!"
    print("[MATCH] Full TF-IDF matrix matches scikit-learn exactly when using identical conventions!")

    print("\n" + "-" * 70)
    print("CONVENTION DIFFERENCE SUMMARY:")
    print("1. Standard Lecture Formula:")
    print("   - TF = c(t, d) / len(d)")
    print("   - IDF = ln(N / df(t)) -> df=N results in IDF=0")
    print("2. Scikit-learn Formula:")
    print("   - Uses raw counts * smooth_idf where smooth_idf = ln((1+N)/(1+df)) + 1")
    print("   - Always applies L2 normalization across the document vector")
    print("   - No term gets IDF=0 due to Laplace smoothing + 1")
    print("-" * 70)


if __name__ == "__main__":
    run_unit_tests()
    compare_with_scikit_learn()
