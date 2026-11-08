"""VectorForge demo: semantic search over a small built-in corpus.

Embeddings are deterministic TF-IDF vectors over the corpus
vocabulary (no external API, no downloads): each word owns one
dimension, TF-IDF weights are applied, and the result is
L2-normalised.  A second demo runs a
synthetic clustered benchmark so the speed/recall numbers are visible.
"""
from __future__ import annotations

import hashlib
import re
import time

import numpy as np

from vectorforge import VectorForge, benchmark
from vectorforge.brute import BruteForceIndex

CORPUS = [
    "How to train a neural network with gradient descent and backpropagation",
    "Convolutional neural networks for image classification and object detection",
    "Transformers and attention mechanisms for natural language processing",
    "Reinforcement learning agents learn from rewards and environment interaction",
    "How to bake sourdough bread with a starter and a Dutch oven",
    "Classic Italian pasta recipes with tomato sauce and fresh basil",
    "Knife skills and seasoning techniques for home cooking",
    "Fermentation basics for kimchi pickles and yogurt at home",
    "Introduction to quantum computing qubits and superposition",
    "Relativity black holes and the expansion of the universe explained",
    "How rockets work orbital mechanics and rocket propulsion systems",
    "The history of the Apollo missions and lunar landing",
    "Investing basics index funds diversification and compound interest",
    "How mortgages work interest rates and amortization schedules",
    "Personal finance budgeting emergency funds and retirement accounts",
    "Stock market valuation earnings reports and price to earnings ratios",
]

_TOKEN_RE = re.compile(r"[a-z]+")


def _stem(tok: str) -> str:
    """Tiny stemmer so 'training'/'train' and 'cooking'/'cook' match."""
    for suf in ("ing", "ed", "es"):
        if tok.endswith(suf) and len(tok) > len(suf) + 2:
            tok = tok[: -len(suf)]
            break
    if tok.endswith("s") and len(tok) > 3:
        tok = tok[:-1]
    return tok


STOPWORDS = {"how", "to", "and", "the", "a", "an", "of", "for", "with", "in", "on",
             "is", "are", "at", "by", "from", "or", "it", "its", "as", "be"}


def _doc_tokens(doc: str) -> set:
    return {_stem(t) for t in _TOKEN_RE.findall(doc.lower()) if t not in STOPWORDS}


# One dimension per vocabulary word: no hash collisions at all, so
# similarity is driven purely by shared content words.
VOCAB = sorted({tok for doc in CORPUS for tok in _doc_tokens(doc)})
DIM = len(VOCAB)
_VOCAB_INDEX = {tok: i for i, tok in enumerate(VOCAB)}


def embed(text: str, dim: int = DIM) -> np.ndarray:
    """Deterministic TF-IDF embedding over the corpus vocabulary.

    Each vocabulary word owns one dimension (no hashing, so no
    collisions); IDF down-weights words that appear in many docs.
    Query words outside the vocabulary are ignored.
    """
    tokens = [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]
    vec = np.zeros(dim, dtype=np.float32)
    for tok in set(tokens):
        if tok not in _VOCAB_INDEX:
            continue
        df = sum(1 for doc in CORPUS if tok in _doc_tokens(doc))
        idf = float(np.log((len(CORPUS) + 1) / (df + 1)) + 1.0)
        vec[_VOCAB_INDEX[tok]] += tokens.count(tok) * idf
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def semantic_demo() -> None:
    print(f"=== Semantic search demo (TF-IDF embeddings, dim={DIM}, vocab={len(VOCAB)}) ===")
    vecs = np.stack([embed(doc) for doc in CORPUS])
    index = VectorForge(dim=DIM, metric="cosine", M=16, ef_construction=64, ef_search=64)
    index.add(vecs)
    for query in ["neural network training", "how to cook pasta", "space rockets and the moon"]:
        idx, scores = index.search(embed(query), k=3)
        print(f"\nQuery: {query!r}")
        for rank, (i, s) in enumerate(zip(idx, scores), 1):
            print(f"  {rank}. (cos sim {1 - float(s):.3f}) {CORPUS[int(i)]}")


def synthetic_demo() -> None:
    print("\n=== Synthetic clustered benchmark (10,000 vectors, dim 128) ===")
    rng = np.random.default_rng(42)
    n, dim, n_clusters = 10_000, 128, 500  # 500 tight clusters x 20 vectors
    centers = rng.normal(size=(n_clusters, dim)).astype(np.float32)
    labels = rng.integers(0, n_clusters, size=n)
    vecs = (centers[labels] + 0.3 * rng.normal(size=(n, dim))).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    q_labels = rng.integers(0, n_clusters, size=100)
    queries = (centers[q_labels] + 0.3 * rng.normal(size=(100, dim))).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    t0 = time.perf_counter()
    index = VectorForge(dim=dim, metric="cosine", M=16, ef_construction=64, ef_search=64)
    index.add(vecs)
    build = time.perf_counter() - t0
    result = benchmark(index, vecs, queries, k=10, name="HNSW (ef=64)", build_time=build)
    print(f"recall@10 {result['recall@k']:.4f} | {result['qps']:.1f} QPS | "
          f"p50 {result['p50_ms']:.3f} ms | p99 {result['p99_ms']:.3f} ms | "
          f"build {result['build_s']:.2f}s | speedup {result['speedup_vs_brute']:.2f}x vs brute force "
          f"({result['brute_qps']:.1f} QPS)")


if __name__ == "__main__":
    semantic_demo()
    synthetic_demo()
