"""Small NumPy implementation of scaled dot-product attention."""

import numpy as np


def softmax(x, axis=-1):
    x = np.asarray(x, dtype=float)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V):
    """Return scaled scores, attention weights, and context (rows are tokens)."""
    Q, K, V = (np.asarray(a, dtype=float) for a in (Q, K, V))
    if Q.ndim != 2 or K.ndim != 2 or V.ndim != 2:
        raise ValueError("Q, K, V must be 2D matrices")
    if Q.shape[1] != K.shape[1] or K.shape[0] != V.shape[0]:
        raise ValueError("Q/K feature sizes and K/V token counts must match")
    scores = Q @ K.T / np.sqrt(K.shape[1])
    attention_weights = softmax(scores, axis=-1)
    context = attention_weights @ V
    return scores, attention_weights, context
