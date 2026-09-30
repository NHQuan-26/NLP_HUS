"""
N-gram Language Model Implementation from scratch for Lab 02.
Course: Xử lý ngôn ngữ tự nhiên và ứng dụng (NLP - HUS)
Student: Nguyễn Hồng Quân - 23001921
Instructor / TA: Phạm Ngọc Hải

Features:
- Tokenization & Vocabulary construction
- N-gram counting (Unigram, Bigram, Trigram)
- Estimation via Maximum Likelihood Estimation (MLE)
- Laplace (Add-one) Smoothing
- Sentence probability and Log-probability (preventing underflow)
- Perplexity computation on test sequences
- Next-word distribution & Top-K prediction
- Sentence ranking
"""

from typing import List, Dict, Tuple, Union, Optional
import math
import re
from collections import defaultdict, Counter


def basic_tokenize(text: str) -> List[str]:
    """
    Standard whitespace and punctuation-aware tokenizer.
    Lowercases text and keeps alphanumeric tokens and common punctuation as separate tokens.
    """
    if not isinstance(text, str):
        return []
    text = text.lower().strip()
    tokens = re.findall(r"\b\w+(?:'\w+)?\b", text)
    return tokens


def build_vocabulary(corpus: List[Union[str, List[str]]]) -> List[str]:
    """
    Build a sorted vocabulary of unique tokens from a corpus of sentences/documents.

    Args:
        corpus: List of raw strings or lists of token strings.

    Returns:
        Sorted list of unique vocabulary tokens.
    """
    vocab = set()
    for item in corpus:
        tokens = basic_tokenize(item) if isinstance(item, str) else [t.lower() for t in item]
        vocab.update(tokens)
    return sorted(list(vocab))


def count_ngrams(corpus: List[Union[str, List[str]]], n: int) -> Dict[Tuple[str, ...], int]:
    """
    Count occurrences of n-grams in a corpus.

    Args:
        corpus: List of sentences (raw strings or token lists).
        n: The 'n' in n-gram (1 for unigram, 2 for bigram, 3 for trigram).

    Returns:
        Dictionary mapping n-gram tuple of strings to its frequency count.
    """
    counts = defaultdict(int)
    for item in corpus:
        tokens = basic_tokenize(item) if isinstance(item, str) else [t.lower() for t in item]
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i + n])
            counts[ngram] += 1
    return dict(counts)


class NGramLanguageModel:
    """
    An N-gram Language Model supporting Unigram (n=1), Bigram (n=2), and Trigram (n=3)
    with Maximum Likelihood Estimation (MLE) or Laplace (Add-one) Smoothing.
    """

    def __init__(self, n: int = 2, smoothing: str = "mle", alpha: float = 1.0, unk_cutoff: int = 0):
        """
        Initialize the N-Gram Language Model.

        Args:
            n: Order of the model (1=Unigram, 2=Bigram, 3=Trigram).
            smoothing: Smoothing method ('mle' for Maximum Likelihood, 'laplace' for Add-k smoothing).
            alpha: Smoothing parameter for Laplace/Add-k smoothing (default 1.0 for Add-one).
            unk_cutoff: Words with frequency <= unk_cutoff are replaced with '<unk>' during fit.
                        If 0, all unique tokens in training corpus are kept in vocabulary.
        """
        if n not in (1, 2, 3):
            raise ValueError(f"Model only supports n in [1, 2, 3], got n={n}")
        if smoothing not in ("mle", "laplace"):
            raise ValueError(f"Supported smoothing methods: 'mle', 'laplace', got {smoothing}")

        self.n = n
        self.smoothing = smoothing
        self.alpha = float(alpha)
        self.unk_cutoff = int(unk_cutoff)
        self.unk_token = "<unk>"
        self.vocabulary: List[str] = []
        self.vocab_set: set = set()
        self.vocab_size: int = 0
        self.total_tokens: int = 0

        # Frequency counters
        self.ngram_counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        self.context_counts: Dict[Tuple[str, ...], int] = defaultdict(int)

    def _replace_oov(self, token: str) -> str:
        """Map word to <unk> if OOV and <unk> is active in vocabulary."""
        token = token.lower()
        if token in self.vocab_set:
            return token
        if self.unk_token in self.vocab_set:
            return self.unk_token
        return token

    def fit(self, corpus: List[Union[str, List[str]]]):
        """
        Train the language model on the provided corpus.
        Computes vocabulary, n-gram frequencies, and context frequencies.
        """
        # First pass to collect raw token frequencies
        raw_token_counts = Counter()
        tokenized_corpus = []
        for item in corpus:
            tokens = basic_tokenize(item) if isinstance(item, str) else [t.lower() for t in item]
            tokenized_corpus.append(tokens)
            raw_token_counts.update(tokens)

        # Build vocabulary with optional UNK replacement
        if self.unk_cutoff > 0:
            vocab = {self.unk_token}
            for token, cnt in raw_token_counts.items():
                if cnt > self.unk_cutoff:
                    vocab.add(token)
            self.vocabulary = sorted(list(vocab))
        else:
            self.vocabulary = sorted(list(raw_token_counts.keys()))

        self.vocab_set = set(self.vocabulary)
        self.vocab_size = len(self.vocabulary)

        self.ngram_counts.clear()
        self.context_counts.clear()
        self.total_tokens = 0

        for tokens in tokenized_corpus:
            processed = [t if t in self.vocab_set else self.unk_token for t in tokens] if self.unk_cutoff > 0 else tokens
            self.total_tokens += len(processed)

            # Count unigrams always for context / fallback
            for t in processed:
                self.ngram_counts[(t,)] += 1

            if self.n >= 2:
                for i in range(len(processed) - 1):
                    bg = (processed[i], processed[i + 1])
                    if self.n == 2:
                        self.ngram_counts[bg] += 1
                    self.context_counts[(processed[i],)] += 1

            if self.n == 3:
                for i in range(len(processed) - 2):
                    tg = (processed[i], processed[i + 1], processed[i + 2])
                    self.ngram_counts[tg] += 1
                    self.context_counts[(processed[i], processed[i + 1])] += 1

        return self

    def probability(self, context: Union[str, Tuple[str, ...], List[str]], word: str) -> float:
        """
        Compute conditional probability P(word | context).

        Args:
            context: Context preceding the word.
                     For unigram (n=1), context is ignored or empty.
                     For bigram (n=2), context is 1 word (str or 1-tuple).
                     For trigram (n=3), context is 2 words (2-tuple or list).
            word: Target token.

        Returns:
            Conditional probability P(word | context) in [0.0, 1.0].
        """
        word = self._replace_oov(word)
        if isinstance(context, str):
            ctx_tokens = basic_tokenize(context) if context else []
            ctx_tuple = tuple(self._replace_oov(t) for t in ctx_tokens)
        elif isinstance(context, (list, tuple)):
            ctx_tuple = tuple(self._replace_oov(t) for t in context)
        else:
            ctx_tuple = ()

        # Standardize context length to (n - 1)
        if len(ctx_tuple) > self.n - 1:
            ctx_tuple = ctx_tuple[-(self.n - 1):]

        if self.n == 1:
            # Unigram: P(w) = C(w) / N
            word_count = self.ngram_counts.get((word,), 0)
            if self.smoothing == "laplace":
                return (word_count + self.alpha) / (self.total_tokens + self.alpha * self.vocab_size)
            else:
                return (word_count / self.total_tokens) if self.total_tokens > 0 else 0.0

        elif self.n == 2:
            # If context is empty, fall back to unigram probability
            if len(ctx_tuple) == 0:
                unigram_count = self.ngram_counts.get((word,), 0)
                if self.smoothing == "laplace":
                    return (unigram_count + self.alpha) / (self.total_tokens + self.alpha * self.vocab_size)
                return (unigram_count / self.total_tokens) if self.total_tokens > 0 else 0.0

            ctx_word = ctx_tuple[-1]
            bigram = (ctx_word, word)
            bigram_count = self.ngram_counts.get(bigram, 0)
            ctx_count = self.ngram_counts.get((ctx_word,), 0)

            if self.smoothing == "laplace":
                # P_laplace(w | h) = (C(h, w) + alpha) / (C(h) + alpha * V)
                return (bigram_count + self.alpha) / (ctx_count + self.alpha * self.vocab_size)
            else:
                return (bigram_count / ctx_count) if ctx_count > 0 else 0.0

        elif self.n == 3:
            # Trigram: P(w | w1, w2) = C(w1, w2, w) / C(w1, w2)
            if len(ctx_tuple) == 0:
                unigram_count = self.ngram_counts.get((word,), 0)
                if self.smoothing == "laplace":
                    return (unigram_count + self.alpha) / (self.total_tokens + self.alpha * self.vocab_size)
                return (unigram_count / self.total_tokens) if self.total_tokens > 0 else 0.0
            elif len(ctx_tuple) == 1:
                # 1-word context fallback to bigram
                ctx_word = ctx_tuple[0]
                bg_count = self.ngram_counts.get((ctx_word, word), 0)
                ctx_count = self.ngram_counts.get((ctx_word,), 0)
                if self.smoothing == "laplace":
                    return (bg_count + self.alpha) / (ctx_count + self.alpha * self.vocab_size)
                return (bg_count / ctx_count) if ctx_count > 0 else 0.0

            trigram = (ctx_tuple[-2], ctx_tuple[-1], word)
            trigram_count = self.ngram_counts.get(trigram, 0)
            ctx_pair = (ctx_tuple[-2], ctx_tuple[-1])
            ctx_count = self.context_counts.get(ctx_pair, 0)

            if self.smoothing == "laplace":
                return (trigram_count + self.alpha) / (ctx_count + self.alpha * self.vocab_size)
            else:
                return (trigram_count / ctx_count) if ctx_count > 0 else 0.0

        return 0.0

    def log_probability(self, context: Union[str, Tuple[str, ...], List[str]], word: str) -> float:
        """
        Compute natural log of conditional probability: ln P(word | context).
        Returns -math.inf if probability is 0.0.
        """
        prob = self.probability(context, word)
        if prob <= 0.0:
            return -math.inf
        return math.log(prob)

    def sentence_probability(self, sentence: Union[str, List[str]]) -> float:
        """
        Compute total joint probability of a sentence using the chain rule:
        P(w_1, ..., w_K) = prod_{i=1}^K P(w_i | w_{i-n+1:i-1})
        """
        tokens = basic_tokenize(sentence) if isinstance(sentence, str) else [t.lower() for t in sentence]
        if not tokens:
            return 0.0

        prob = 1.0
        for i, token in enumerate(tokens):
            if self.n == 1:
                ctx = ()
            elif self.n == 2:
                ctx = (tokens[i - 1],) if i >= 1 else ()
            elif self.n == 3:
                if i == 0:
                    ctx = ()
                elif i == 1:
                    ctx = (tokens[0],)
                else:
                    ctx = (tokens[i - 2], tokens[i - 1])

            p_i = self.probability(ctx, token)
            if p_i <= 0.0:
                return 0.0
            prob *= p_i

        return prob

    def sentence_log_probability(self, sentence: Union[str, List[str]]) -> float:
        """
        Compute sum of log probabilities to avoid floating point underflow:
        log P(W) = sum_{i=1}^K log P(w_i | context)
        """
        tokens = basic_tokenize(sentence) if isinstance(sentence, str) else [t.lower() for t in sentence]
        if not tokens:
            return -math.inf

        total_log_prob = 0.0
        for i, token in enumerate(tokens):
            if self.n == 1:
                ctx = ()
            elif self.n == 2:
                ctx = (tokens[i - 1],) if i >= 1 else ()
            elif self.n == 3:
                if i == 0:
                    ctx = ()
                elif i == 1:
                    ctx = (tokens[0],)
                else:
                    ctx = (tokens[i - 2], tokens[i - 1])

            lp_i = self.log_probability(ctx, token)
            if lp_i == -math.inf:
                return -math.inf
            total_log_prob += lp_i

        return total_log_prob

    def perplexity(self, sentences: List[Union[str, List[str]]]) -> float:
        """
        Compute perplexity across a set of sentences:
        PP = exp( - (1 / N_total) * sum_{tokens} log P(w_i | context) )
        """
        total_log_prob = 0.0
        total_tokens = 0

        for sent in sentences:
            tokens = basic_tokenize(sent) if isinstance(sent, str) else [t.lower() for t in sent]
            if not tokens:
                continue

            for i, token in enumerate(tokens):
                total_tokens += 1
                if self.n == 1:
                    ctx = ()
                elif self.n == 2:
                    ctx = (tokens[i - 1],) if i >= 1 else ()
                elif self.n == 3:
                    if i == 0:
                        ctx = ()
                    elif i == 1:
                        ctx = (tokens[0],)
                    else:
                        ctx = (tokens[i - 2], tokens[i - 1])

                lp = self.log_probability(ctx, token)
                if lp == -math.inf:
                    return float("inf")
                total_log_prob += lp

        if total_tokens == 0:
            return float("inf")

        cross_entropy = -total_log_prob / total_tokens
        try:
            return math.exp(cross_entropy)
        except OverflowError:
            return float("inf")

    def next_word_distribution(self, context: Union[str, Tuple[str, ...], List[str]]) -> Dict[str, float]:
        """
        Compute conditional probability distribution over the entire vocabulary for a given context.
        """
        dist = {}
        for word in self.vocabulary:
            dist[word] = self.probability(context, word)
        return dist

    def predict_next_words(
        self, context: Union[str, Tuple[str, ...], List[str]], top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Predict the top-k most probable next words given a context.

        Returns:
            List of (word, probability) tuples sorted descending by probability.
        """
        dist = self.next_word_distribution(context)
        sorted_candidates = sorted(dist.items(), key=lambda item: item[1], reverse=True)
        return sorted_candidates[:top_k]

    def rank_continuations(
        self, context: Union[str, Tuple[str, ...], List[str]], candidates: List[str]
    ) -> List[Tuple[str, float, float]]:
        """
        Score and rank candidate continuation phrases/sentences given a preceding context.

        Args:
            context: Preceding text or tokens.
            candidates: List of candidate continuation strings.

        Returns:
            List of (candidate, total_log_prob, perplexity) tuples sorted descending by score.
        """
        ctx_tokens = basic_tokenize(context) if isinstance(context, str) else list(context)
        ranked = []

        for cand in candidates:
            cand_tokens = basic_tokenize(cand) if isinstance(cand, str) else list(cand)
            full_tokens = ctx_tokens + cand_tokens
            
            # Evaluate log probability solely over the continuation tokens given full history
            cand_log_prob = 0.0
            is_impossible = False
            for idx in range(len(ctx_tokens), len(full_tokens)):
                token = full_tokens[idx]
                if self.n == 1:
                    ctx = ()
                elif self.n == 2:
                    ctx = (full_tokens[idx - 1],) if idx >= 1 else ()
                elif self.n == 3:
                    if idx == 0:
                        ctx = ()
                    elif idx == 1:
                        ctx = (full_tokens[0],)
                    else:
                        ctx = (full_tokens[idx - 2], full_tokens[idx - 1])

                lp = self.log_probability(ctx, token)
                if lp == -math.inf:
                    is_impossible = True
                    break
                cand_log_prob += lp

            if is_impossible or len(cand_tokens) == 0:
                ppl = float("inf")
                score = -math.inf
            else:
                score = cand_log_prob
                ppl = math.exp(-cand_log_prob / len(cand_tokens))

            ranked.append((cand, score, ppl))

        # Rank candidates by normalized Perplexity ascending (lower PPL = more natural continuation)
        ranked.sort(key=lambda x: (x[2], -x[1]))
        return ranked


# ------------------------------------------------------------------------------
# Standalone functions required by Lab 02 specifications (Section 14)
# ------------------------------------------------------------------------------

def train_unigram(corpus: List[Union[str, List[str]]], smoothing: str = "mle") -> NGramLanguageModel:
    """Helper to train a Unigram Language Model."""
    model = NGramLanguageModel(n=1, smoothing=smoothing)
    model.fit(corpus)
    return model


def train_bigram(corpus: List[Union[str, List[str]]], smoothing: str = "mle") -> NGramLanguageModel:
    """Helper to train a Bigram Language Model."""
    model = NGramLanguageModel(n=2, smoothing=smoothing)
    model.fit(corpus)
    return model


def train_trigram(corpus: List[Union[str, List[str]]], smoothing: str = "mle") -> NGramLanguageModel:
    """Helper to train a Trigram Language Model."""
    model = NGramLanguageModel(n=3, smoothing=smoothing)
    model.fit(corpus)
    return model


def probability(model: NGramLanguageModel, context: Union[str, Tuple[str, ...]], word: str) -> float:
    """Standalone wrapper for model conditional probability."""
    return model.probability(context, word)


def sentence_probability(model: NGramLanguageModel, sentence: Union[str, List[str]]) -> float:
    """Standalone wrapper for sentence joint probability."""
    return model.sentence_probability(sentence)


def sentence_log_probability(model: NGramLanguageModel, sentence: Union[str, List[str]]) -> float:
    """Standalone wrapper for sentence log probability."""
    return model.sentence_log_probability(sentence)


def perplexity(model: NGramLanguageModel, sentences: List[Union[str, List[str]]]) -> float:
    """Standalone wrapper for perplexity calculation."""
    return model.perplexity(sentences)
