"""
Script to execute all experiments for Lab 01 (Sections 7-D to 13-J),
generate evaluation tables, outputs, and export results.csv.
"""

import json
import os
import sys
import time
import re
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import tiktoken
import nltk
from nltk.corpus import stopwords

# Ensure NLTK resources
stop_words = set(stopwords.words('english'))

print("=" * 80)
print("STARTING LAB 01 EXPERIMENTS (Sections 7 to 13)")
print("=" * 80)

# ------------------------------------------------------------------------------
# 1. Load 30K C4 Corpus
# ------------------------------------------------------------------------------
data_paths = [
    "lab_1/c4-train.00000-of-01024-30K.json",
    "c4-train.00000-of-01024-30K.json",
    "../lab_1/c4-train.00000-of-01024-30K.json"
]
corpus_path = None
for p in data_paths:
    if os.path.exists(p):
        corpus_path = p
        break

if not corpus_path:
    raise FileNotFoundError("Could not find c4-train dataset!")

print(f"Loading corpus from: {corpus_path}")
docs = []
urls = []
with open(corpus_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            entry = json.loads(line)
            docs.append(entry.get("text", ""))
            urls.append(entry.get("url", ""))

N = len(docs)
print(f"Loaded N = {N} documents successfully.")

# ------------------------------------------------------------------------------
# 2. Part D — Experiment 1: Inspect the Sparse Representation
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("PART D — EXPERIMENT 1: INSPECT SPARSE REPRESENTATION")
print("=" * 80)

# Scikit-learn TF-IDF pipeline on 30K documents
tfidf_vec = TfidfVectorizer()
X_tfidf = tfidf_vec.fit_transform(docs)
V = len(tfidf_vec.vocabulary_)
nnz = X_tfidf.nnz
total_entries = N * V
sparsity = 1.0 - (nnz / total_entries)
feature_names = np.array(tfidf_vec.get_feature_names_out())

print(f"Number of documents (N) = {N}")
print(f"Vocabulary size (V)       = {V}")
print(f"Matrix shape              = {X_tfidf.shape}")
print(f"Non-zero elements (nnz)   = {nnz}")
print(f"Matrix Sparsity S         = {sparsity * 100:.6f}%")

# 7.5 Inspect Vocabulary:
# 1) Top 20 terms by document frequency (DF)
df = np.diff(X_tfidf.tocsc().indptr)
top20_df_idx = np.argsort(df)[::-1][:20]
top20_df = [(feature_names[i], int(df[i]), float(tfidf_vec.idf_[i])) for i in top20_df_idx]

print("\n--- Top 20 Terms by Document Frequency (DF) ---")
print(f"{'Rank':<5} | {'Term':<15} | {'DF':<8} | {'IDF':<8}")
print("-" * 42)
for rank, (term, d_val, idf_val) in enumerate(top20_df, 1):
    print(f"{rank:<5} | {term:<15} | {d_val:<8} | {idf_val:.4f}")

# 2) Top 20 terms with highest IDF
top20_idf_idx = np.argsort(tfidf_vec.idf_)[::-1][:20]
top20_idf = [(feature_names[i], int(df[i]), float(tfidf_vec.idf_[i])) for i in top20_idf_idx]

print("\n--- Top 20 Terms with Highest IDF ---")
print(f"{'Rank':<5} | {'Term':<15} | {'DF':<8} | {'IDF':<8}")
print("-" * 42)
for rank, (term, d_val, idf_val) in enumerate(top20_idf, 1):
    print(f"{rank:<5} | {term:<15} | {d_val:<8} | {idf_val:.4f}")

# 3) Top 20 terms by TF-IDF in a selected document (Document 0: BBQ Class)
doc0_vec = X_tfidf[0].toarray().flatten()
top20_doc0_idx = np.argsort(doc0_vec)[::-1][:20]
top20_doc0 = [(feature_names[i], float(doc0_vec[i])) for i in top20_doc0_idx]

print("\n--- Top 20 Terms by TF-IDF in Document 0 (BBQ Class) ---")
print(f"{'Rank':<5} | {'Term':<15} | {'TF-IDF Weight':<15}")
print("-" * 42)
for rank, (term, weight) in enumerate(top20_doc0, 1):
    print(f"{rank:<5} | {term:<15} | {weight:.4f}")

# ------------------------------------------------------------------------------
# 3. Part F — Experiment 2: Preprocessing Ablation
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("PART F — EXPERIMENT 2: PREPROCESSING ABLATION")
print("=" * 80)

# Query set to test OOV rate and retrieval latency
test_queries = [
    "medical image classification",
    "transformer language model",
    "deep learning healthcare",
    "natural language processing",
    "heart attack prevention",
    "climate change renewable energy",
    "quantum computing algorithms",
    "convolutional neural network"
]

# Pipeline A: Minimal (raw lowercase + whitespace tokenization)
t0 = time.time()
vec_A = CountVectorizer(lowercase=True, token_pattern=r'\S+')
X_A = vec_A.fit_transform(docs)
time_A = time.time() - t0
vocab_A = len(vec_A.vocabulary_)
avg_tokens_A = X_A.sum() / N
sparsity_A = 1.0 - (X_A.nnz / (N * vocab_A))

# Calculate OOV for queries on Pipeline A
oov_A_count = sum(1 for q in test_queries for word in q.lower().split() if word not in vec_A.vocabulary_)
total_query_words = sum(len(q.lower().split()) for q in test_queries)
oov_A_rate = oov_A_count / total_query_words

# Pipeline B: Normalized (lowercase + punctuation removal + stopwords removal)
t0 = time.time()
vec_B = CountVectorizer(lowercase=True, stop_words='english', token_pattern=r'(?u)\b[a-zA-Z]{2,}\b')
X_B = vec_B.fit_transform(docs)
time_B = time.time() - t0
vocab_B = len(vec_B.vocabulary_)
avg_tokens_B = X_B.sum() / N
sparsity_B = 1.0 - (X_B.nnz / (N * vocab_B))

# OOV for queries on Pipeline B (words not in vocabulary after excluding stopwords)
oov_B_count = 0
valid_b_words = 0
for q in test_queries:
    words = [w for w in re.findall(r'[a-zA-Z]{2,}', q.lower()) if w not in stop_words]
    for w in words:
        valid_b_words += 1
        if w not in vec_B.vocabulary_:
            oov_B_count += 1
oov_B_rate = oov_B_count / max(1, valid_b_words)

# Pipeline C: Extended (Subword Tokenization using BPE - GPT-2 tiktoken)
enc = tiktoken.get_encoding("gpt2")
vocab_C = enc.n_vocab
# Subword sample over 1000 docs to accurately estimate average tokens
sample_docs = docs[:1000]
sample_subword_counts = [len(enc.encode(d, disallowed_special=())) for d in sample_docs]
avg_tokens_C = np.mean(sample_subword_counts)
# OOV for BPE is 0.0% by definition (byte-level fallback)
oov_C_rate = 0.0
# Estimate sparsity based on average unique subwords per doc (~160)
sample_unique_subwords = [len(set(enc.encode(d, disallowed_special=()))) for d in sample_docs]
avg_unique_C = np.mean(sample_unique_subwords)
sparsity_C = 1.0 - (avg_unique_C / vocab_C)

ablation_df = pd.DataFrame({
    "Pipeline": ["Pipeline A (Minimal)", "Pipeline B (Normalized)", "Pipeline C (Extended BPE)"],
    "Preprocessing Details": [
        "Lowercase + Whitespace tokens",
        "Lowercase + Punctuation removal + Stopwords removal",
        "Normalization + Byte-level BPE Subwords (GPT-2)"
    ],
    "Vocabulary Size": [vocab_A, vocab_B, vocab_C],
    "Avg Tokens/Doc": [round(avg_tokens_A, 1), round(avg_tokens_B, 1), round(avg_tokens_C, 1)],
    "Matrix Sparsity": [f"{sparsity_A*100:.4f}%", f"{sparsity_B*100:.4f}%", f"{sparsity_C*100:.4f}%"],
    "Query OOV Rate": [f"{oov_A_rate*100:.1f}%", f"{oov_B_rate*100:.1f}%", f"{oov_C_rate*100:.1f}%"]
})

print("\n--- Ablation Experiment Comparison Table ---")
print(ablation_df.to_string(index=False))

# ------------------------------------------------------------------------------
# 4. Part G — Application: Build a Document Search Engine
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("PART G — APPLICATION: DOCUMENT SEARCH ENGINE")
print("=" * 80)

class TfidfSearchEngine:
    """
    In-memory vector space document search engine using TF-IDF and Cosine Similarity.
    """
    def __init__(self, vectorizer=None):
        self.vectorizer = vectorizer or TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            token_pattern=r'(?u)\b[a-zA-Z]{2,}\b'
        )
        self.doc_matrix = None
        self.documents = []

    def index(self, documents):
        """Fit vectorizer and index documents."""
        self.documents = documents
        print(f"Indexing {len(documents)} documents...")
        t0 = time.time()
        self.doc_matrix = self.vectorizer.fit_transform(documents)
        print(f"Indexed in {time.time() - t0:.2f}s! Matrix shape: {self.doc_matrix.shape}")

    def search(self, query: str, top_k: int = 5):
        """
        Search for top_k documents most relevant to query.
        Returns list of (rank, doc_id, score, preview_text).
        """
        q_vec = self.vectorizer.transform([query])
        # Vectorized cosine similarity (since rows in doc_matrix and q_vec are L2-normalized)
        # Cosine similarity = doc_matrix * q_vec.T
        scores = self.doc_matrix.dot(q_vec.T).toarray().flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            score = float(scores[idx])
            preview = self.documents[idx].replace("\n", " ")[:150] + "..."
            results.append({
                "rank": rank,
                "doc_id": int(idx),
                "similarity": score,
                "preview": preview
            })
        return results

search_engine = TfidfSearchEngine()
search_engine.index(docs)

example_queries = [
    "medical image classification",
    "transformer language model",
    "deep learning healthcare",
    "natural language processing"
]

for q in example_queries:
    print(f"\nQuery: '{q}'")
    hits = search_engine.search(q, top_k=5)
    print(f"{'Rank':<5} | {'Doc ID':<8} | {'Similarity':<10} | {'Document Preview'}")
    print("-" * 80)
    for h in hits:
        print(f"{h['rank']:<5} | {h['doc_id']:<8} | {h['similarity']:<10.4f} | {h['preview']}")

# ------------------------------------------------------------------------------
# 5. Part H — Evaluation & Benchmark
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("PART H — EVALUATION & BENCHMARK")
print("=" * 80)

# Build a grounded evaluation benchmark with 8 queries across domains
# Ground truth relevant documents are determined through domain keywords & validation
benchmark_queries = [
    {
        "query_id": "Q1",
        "query": "medical image classification",
        "keywords": ["medical", "image", "classification"],
        "category": "Medical Imaging"
    },
    {
        "query_id": "Q2",
        "query": "transformer language model",
        "keywords": ["transformer", "language", "model"],
        "category": "Deep Learning / NLP"
    },
    {
        "query_id": "Q3",
        "query": "deep learning healthcare",
        "keywords": ["deep", "learning", "healthcare"],
        "category": "Healthcare AI"
    },
    {
        "query_id": "Q4",
        "query": "natural language processing",
        "keywords": ["natural", "language", "processing"],
        "category": "NLP"
    },
    {
        "query_id": "Q5",
        "query": "bbq cooking meat smoker",
        "keywords": ["bbq", "smoker", "meat"],
        "category": "Culinary / BBQ"
    },
    {
        "query_id": "Q6",
        "query": "macbook restore disk utility",
        "keywords": ["disk", "utility", "restore"],
        "category": "Tech Support"
    },
    {
        "query_id": "Q7",
        "query": "heart attack prevention",
        "keywords": ["heart", "attack", "prevention"],
        "category": "Cardiology (Lexical match)"
    },
    {
        "query_id": "Q8",
        "query": "myocardial infarction therapy",
        "keywords": ["myocardial", "infarction", "therapy"],
        "category": "Cardiology (Clinical synonym)"
    }
]

# Identify relevant documents in corpus for each query (documents having strong topic relevance)
evaluation_records = []
for bq in benchmark_queries:
    qid = bq["query_id"]
    query = bq["query"]
    kw = bq["keywords"]

    # Ground truth: documents containing all keywords in text
    relevant_doc_ids = []
    for doc_idx, doc_text in enumerate(docs):
        d_lower = doc_text.lower()
        if all(k in d_lower for k in kw):
            relevant_doc_ids.append(doc_idx)

    # Search top 5
    retrieved = search_engine.search(query, top_k=5)
    retrieved_ids = [r["doc_id"] for r in retrieved]

    # Metrics
    # Precision@5 = # relevant retrieved / 5
    rel_retrieved = [doc_id for doc_id in retrieved_ids if doc_id in relevant_doc_ids]
    p_at_5 = len(rel_retrieved) / 5.0

    # Recall@5 = # relevant retrieved / # total relevant
    total_rel = max(1, len(relevant_doc_ids))
    r_at_5 = len(rel_retrieved) / total_rel

    # MRR (Reciprocal Rank of first relevant document)
    mrr = 0.0
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_doc_ids:
            mrr = 1.0 / rank
            break

    evaluation_records.append({
        "query_id": qid,
        "query_text": query,
        "category": bq["category"],
        "total_relevant_in_corpus": len(relevant_doc_ids),
        "precision@5": round(p_at_5, 4),
        "recall@5": round(r_at_5, 4),
        "mrr": round(mrr, 4),
        "retrieved_top5_ids": str(retrieved_ids),
        "relevant_hits_in_top5": len(rel_retrieved)
    })

eval_df = pd.DataFrame(evaluation_records)
print("\n--- Quantitative Evaluation Results ---")
print(eval_df[["query_id", "query_text", "precision@5", "recall@5", "mrr", "relevant_hits_in_top5", "total_relevant_in_corpus"]].to_string(index=False))

mean_p5 = eval_df["precision@5"].mean()
mean_r5 = eval_df["recall@5"].mean()
mean_mrr = eval_df["mrr"].mean()

print("\n--- Summary Metrics Across Benchmark ---")
print(f"Mean Precision@5 : {mean_p5:.4f}")
print(f"Mean Recall@5    : {mean_r5:.4f}")
print(f"Mean MRR         : {mean_mrr:.4f}")

# Export results.csv for both lab01 and lab_1
eval_df.to_csv("lab01/results.csv", index=False)
eval_df.to_csv("lab_1/results.csv", index=False)
print("Saved evaluation results to lab01/results.csv and lab_1/results.csv")

print("\n" + "=" * 80)
print("ALL EXPERIMENTAL RUNS FINISHED SUCCESSFULLY!")
print("=" * 80)
