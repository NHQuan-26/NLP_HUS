"""
Script to execute all experiments for Lab 02:
- Experiment 1: Corpus statistics & N-gram distributions
- Experiment 2: MLE vs Laplace Smoothing comparison
- Experiment 3: Perplexity evaluation across Unigram, Bigram, Trigram
- Applications: Next-word prediction & Candidate sentence ranking
Outputs results to results.csv and saves plot images.
"""

import os
import sys
import json
import time
import math
import re
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure local import of ngram_lm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ngram_lm import (
    NGramLanguageModel,
    basic_tokenize,
    build_vocabulary,
    count_ngrams,
)


def load_corpus(max_docs: int = 15000):
    """Load text documents from C4 dataset."""
    possible_paths = [
        "c4-train.00000-of-01024-30K.json",
        "../lab01/c4-train.00000-of-01024-30K.json",
        "lab01/c4-train.00000-of-01024-30K.json",
    ]
    path_found = None
    for p in possible_paths:
        if os.path.exists(p):
            path_found = p
            break

    if not path_found:
        raise FileNotFoundError("Could not find c4-train.00000-of-01024-30K.json")

    print(f"Loading up to {max_docs} documents from: {path_found} ...")
    docs = []
    with open(path_found, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_docs:
                break
            if line.strip():
                item = json.loads(line)
                docs.append(item.get("text", ""))

    print(f"Loaded {len(docs)} documents.")
    return docs


def extract_sentences(docs):
    """Segment documents into sentences and clean tokens."""
    sentences = []
    for doc in docs:
        raw_sents = re.split(r"[.!?\n\r]+", doc)
        for s in raw_sents:
            toks = basic_tokenize(s)
            if len(toks) >= 2:
                sentences.append(toks)
    return sentences


def run_experiment_1_statistics(sentences, num_docs):
    """Compute comprehensive statistics on the corpus (Section 13)."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 1 — CORPUS & N-GRAM STATISTICS (Section 13)")
    print("=" * 80)

    num_sentences = len(sentences)
    total_tokens = sum(len(s) for s in sentences)

    # Count n-grams
    unigram_counter = Counter()
    bigram_counter = Counter()
    trigram_counter = Counter()

    for s in sentences:
        for w in s:
            unigram_counter[w] += 1
        for i in range(len(s) - 1):
            bigram_counter[(s[i], s[i + 1])] += 1
        for i in range(len(s) - 2):
            trigram_counter[(s[i], s[i + 1], s[i + 2])] += 1

    vocab_size = len(unigram_counter)
    num_unigrams = vocab_size
    num_bigrams = len(bigram_counter)
    num_trigrams = len(trigram_counter)

    unigram_singletons = sum(1 for c in unigram_counter.values() if c == 1)
    bigram_singletons = sum(1 for c in bigram_counter.values() if c == 1)
    trigram_singletons = sum(1 for c in trigram_counter.values() if c == 1)

    stats_summary = {
        "Metric": [
            "Documents",
            "Sentences",
            "Tokens",
            "Vocabulary Size (Unique Unigrams)",
            "Unique Bigrams",
            "Unique Trigrams",
            "Unigram Singletons (Count=1)",
            "Bigram Singletons (Count=1)",
            "Trigram Singletons (Count=1)",
            "Unigram Singleton Ratio (%)",
            "Bigram Singleton Ratio (%)",
            "Trigram Singleton Ratio (%)",
        ],
        "Value": [
            num_docs,
            num_sentences,
            total_tokens,
            num_unigrams,
            num_bigrams,
            num_trigrams,
            unigram_singletons,
            bigram_singletons,
            trigram_singletons,
            round(unigram_singletons / num_unigrams * 100, 2) if num_unigrams else 0,
            round(bigram_singletons / num_bigrams * 100, 2) if num_bigrams else 0,
            round(trigram_singletons / num_trigrams * 100, 2) if num_trigrams else 0,
        ],
    }

    df_stats = pd.DataFrame(stats_summary)
    print(df_stats.to_string(index=False))

    # Generate Frequency Distribution Plots (Section 13)
    out_dir = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, counts, title, color in zip(
        axes,
        [unigram_counter, bigram_counter, trigram_counter],
        ["Unigram Frequency (Top 50)", "Bigram Frequency (Top 50)", "Trigram Frequency (Top 50)"],
        ["#1f77b4", "#ff7f0e", "#2ca02c"],
    ):
        top_items = counts.most_common(50)
        freqs = [item[1] for item in top_items]
        ranks = range(1, len(freqs) + 1)
        ax.plot(ranks, freqs, marker="o", color=color, linewidth=2, markersize=4)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Rank", fontsize=10)
        ax.set_ylabel("Frequency (log scale)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "ngram_distribution.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved n-gram frequency distribution plot to: {plot_path}")

    return df_stats, (unigram_counter, bigram_counter, trigram_counter)


def run_experiment_2_mle_vs_smoothing(train_sents, valid_sents, test_sents):
    """Compare MLE vs Laplace smoothing on Train, Valid, Test (Section 16)."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 2 — MLE VS LAPLACE SMOOTHING (Section 16)")
    print("=" * 80)

    # Subsample for evaluation efficiency while ensuring statistical significance
    eval_train = train_sents[:3000]
    eval_val = valid_sents[:3000]
    eval_test = test_sents[:3000]

    configs = [
        ("Bigram MLE", 2, "mle"),
        ("Bigram Laplace", 2, "laplace"),
        ("Trigram MLE", 3, "mle"),
        ("Trigram Laplace", 3, "laplace"),
    ]

    results = []
    models = {}

    for name, n, smoothing in configs:
        print(f"Training {name} (n={n}, smoothing={smoothing}) ...")
        t0 = time.time()
        # Use unk_cutoff=1 for robust vocabulary handling
        model = NGramLanguageModel(n=n, smoothing=smoothing, alpha=1.0, unk_cutoff=1)
        model.fit(train_sents)
        train_time = time.time() - t0
        models[name] = model

        # Evaluate PPL
        train_ppl = model.perplexity(eval_train)
        val_ppl = model.perplexity(eval_val)
        test_ppl = model.perplexity(eval_test)

        train_ppl_str = f"{train_ppl:.2f}" if math.isfinite(train_ppl) else "inf"
        val_ppl_str = f"{val_ppl:.2f}" if math.isfinite(val_ppl) else "inf"
        test_ppl_str = f"{test_ppl:.2f}" if math.isfinite(test_ppl) else "inf"

        print(f" -> {name}: Train PPL = {train_ppl_str} | Valid PPL = {val_ppl_str} | Test PPL = {test_ppl_str} (fit time: {train_time:.2f}s)")

        results.append({
            "Experiment": "Exp2_MLE_vs_Smoothing",
            "Model": name,
            "N": n,
            "Smoothing": smoothing,
            "Train PPL": train_ppl_str,
            "Valid PPL": val_ppl_str,
            "Test PPL": test_ppl_str,
        })

    df_mle_smooth = pd.DataFrame(results)
    return df_mle_smooth, models


def run_experiment_3_perplexity(train_sents, valid_sents, test_sents):
    """Compare Unigram, Bigram, Trigram with Laplace smoothing (Section 19 & 23)."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 3 — PERPLEXITY ACROSS N-GRAMS & CONTEXT LENGTH (Section 19 & 23)")
    print("=" * 80)

    eval_train = train_sents[:3000]
    eval_val = valid_sents[:3000]
    eval_test = test_sents[:3000]

    configs = [
        ("Unigram Laplace", 1),
        ("Bigram Laplace", 2),
        ("Trigram Laplace", 3),
    ]

    results = []
    models = {}

    for name, n in configs:
        print(f"Training {name} (n={n}) ...")
        model = NGramLanguageModel(n=n, smoothing="laplace", alpha=1.0, unk_cutoff=1)
        model.fit(train_sents)
        models[name] = model

        train_ppl = model.perplexity(eval_train)
        val_ppl = model.perplexity(eval_val)
        test_ppl = model.perplexity(eval_test)

        train_ppl_str = f"{train_ppl:.2f}" if math.isfinite(train_ppl) else "inf"
        val_ppl_str = f"{val_ppl:.2f}" if math.isfinite(val_ppl) else "inf"
        test_ppl_str = f"{test_ppl:.2f}" if math.isfinite(test_ppl) else "inf"

        print(f" -> {name}: Train PPL = {train_ppl_str} | Valid PPL = {val_ppl_str} | Test PPL = {test_ppl_str}")

        results.append({
            "Experiment": "Exp3_Ngram_Perplexity",
            "Model": name,
            "N": n,
            "Smoothing": "laplace",
            "Train PPL": train_ppl_str,
            "Valid PPL": val_ppl_str,
            "Test PPL": test_ppl_str,
        })

    df_ppl = pd.DataFrame(results)
    return df_ppl, models


def run_application_next_word(model_bigram, model_trigram):
    """Next-word prediction for at least 5 contexts (Section 20)."""
    print("\n" + "=" * 80)
    print("APPLICATION 1 — NEXT-WORD PREDICTION (Section 20)")
    print("=" * 80)

    contexts = [
        ("the cat", "eats"),
        ("natural language", "processing"),
        ("machine learning", "models"),
        ("artificial intelligence", "and"),
        ("deep neural", "network"),
    ]

    records = []
    for ctx, actual in contexts:
        # Use trigram for prediction
        predictions = model_trigram.predict_next_words(ctx, top_k=5)
        pred_summary = "; ".join([f"{w} ({p:.4f})" for w, p in predictions[:3]])
        top_word = predictions[0][0] if predictions else "None"
        top_prob = predictions[0][1] if predictions else 0.0

        print(f"Context: '{ctx}' | Actual: '{actual}'")
        for rank, (w, p) in enumerate(predictions, 1):
            print(f"  {rank}. {w:<15} P = {p:.5f}")

        records.append({
            "Context": ctx,
            "Top Prediction": top_word,
            "Top Probability": f"{top_prob:.5f}",
            "Top 3 Predictions": pred_summary,
            "Actual Next Word": actual,
        })

    df_next_word = pd.DataFrame(records)
    return df_next_word


def run_application_sentence_ranking(model_trigram):
    """Rank candidate continuations given a context (Section 21)."""
    print("\n" + "=" * 80)
    print("APPLICATION 2 — SENTENCE RANKING (Section 21)")
    print("=" * 80)

    test_cases = [
        {
            "context": "machine learning",
            "candidates": [
                "is useful for nlp",
                "banana computer quickly",
                "studies language models",
            ],
        },
        {
            "context": "natural language",
            "candidates": [
                "processing is a field of artificial intelligence",
                "table jump green potato",
                "understanding has made huge progress",
            ],
        },
    ]

    records = []
    for tc in test_cases:
        ctx = tc["context"]
        cands = tc["candidates"]
        print(f"\nContext: '{ctx}'")
        ranked = model_trigram.rank_continuations(ctx, cands)
        for rank, (cand, score, ppl) in enumerate(ranked, 1):
            print(f"  Rank {rank}: '{cand}' | Score (LogProb): {score:.3f} | PPL: {ppl:.2f}")
            records.append({
                "Context": ctx,
                "Rank": rank,
                "Candidate Continuation": cand,
                "Log Probability Score": f"{score:.4f}",
                "Perplexity": f"{ppl:.2f}" if math.isfinite(ppl) else "inf",
            })

    df_ranking = pd.DataFrame(records)
    return df_ranking


def main():
    print("Starting Lab 02 Experiments Execution ...")
    start_time = time.time()

    # 1. Load data
    docs = load_corpus(max_docs=15000)
    sentences = extract_sentences(docs)
    print(f"Total extracted sentences: {len(sentences)}")

    # 2. Experiment 1: Statistics
    df_stats, ngram_counters = run_experiment_1_statistics(sentences, len(docs))

    # 3. Split into Train / Val / Test (80% / 10% / 10%)
    np.random.seed(42)
    indices = np.random.permutation(len(sentences))
    n_train = int(len(sentences) * 0.8)
    n_val = int(len(sentences) * 0.1)

    train_sents = [sentences[i] for i in indices[:n_train]]
    valid_sents = [sentences[i] for i in indices[n_train : n_train + n_val]]
    test_sents = [sentences[i] for i in indices[n_train + n_val :]]

    print(f"\nSplit sizes: Train={len(train_sents)}, Valid={len(valid_sents)}, Test={len(test_sents)}")

    # 4. Experiment 2: MLE vs Laplace
    df_exp2, models_exp2 = run_experiment_2_mle_vs_smoothing(train_sents, valid_sents, test_sents)

    # 5. Experiment 3: Perplexity across N-grams
    df_exp3, models_exp3 = run_experiment_3_perplexity(train_sents, valid_sents, test_sents)

    # 6. Applications
    tri_model = models_exp2["Trigram Laplace"]
    bi_model = models_exp2["Bigram Laplace"]
    df_app_pred = run_application_next_word(bi_model, tri_model)
    df_app_rank = run_application_sentence_ranking(tri_model)

    # 7. Export combined results to results.csv
    out_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(out_dir, "results.csv")

    # Combine into a clean CSV format with section markers
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("# LAB 02 EXPERIMENTAL RESULTS\n")
        f.write("# Student: Nguyen Hong Quan - 23001921\n\n")

        f.write("=== EXPERIMENT 1: CORPUS STATISTICS ===\n")
        df_stats.to_csv(f, index=False)
        f.write("\n")

        f.write("=== EXPERIMENT 2: MLE VS LAPLACE SMOOTHING ===\n")
        df_exp2.to_csv(f, index=False)
        f.write("\n")

        f.write("=== EXPERIMENT 3: N-GRAM PERPLEXITY EVALUATION ===\n")
        df_exp3.to_csv(f, index=False)
        f.write("\n")

        f.write("=== APPLICATION: NEXT-WORD PREDICTION (TOP 5 CONTEXTS) ===\n")
        df_app_pred.to_csv(f, index=False)
        f.write("\n")

        f.write("=== APPLICATION: SENTENCE RANKING CANDIDATES ===\n")
        df_app_rank.to_csv(f, index=False)

    print(f"\nAll experimental results saved successfully to: {csv_path}")
    print(f"Total experiment execution time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
