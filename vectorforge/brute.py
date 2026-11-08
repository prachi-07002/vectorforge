"""Exact brute-force nearest-neighbour search.

NumPy-vectorised baseline used both as a standalone index and as the
ground-truth reference for recall measurements of the approximate indexes.
Supports cosine similarity and squared-L2 distance.
"""
from __future__ import annotations

import numpy as np


def _prepare(x: np.ndarray, metric: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    if metric == "cosine":
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (x / norms).astype(np.float32)
    return x


class BruteForceIndex:
    """Exact k-NN search by a full scan of every stored vector.

    Parameters
    ----------
    metric : {"cosine", "l2"}
        ``cosine`` ranks by cosine similarity (vectors are L2-normalised
        on insert); ``l2`` ranks by squared Euclidean distance.
    """

    def __init__(self, metric: str = "cosine") -> None:
        if metric not in ("cosine", "l2"):
            raise ValueError(f"unknown metric {metric!r}")
        self.metric = metric
        self.vectors: np.ndarray | None = None
        self.dim: int | None = None

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            raise ValueError("vectors must be a non-empty 2-D array")
        if self.dim is not None and vectors.shape[1] != self.dim:
            raise ValueError("dimension mismatch")
        self.dim = vectors.shape[1]
        prep = _prepare(vectors, self.metric)
        self.vectors = prep if self.vectors is None else np.vstack([self.vectors, prep])

    def __len__(self) -> int:
        return 0 if self.vectors is None else len(self.vectors)

    def _scores(self, query: np.ndarray) -> np.ndarray:
        """Higher is better for cosine, lower is better for L2."""
        q = _prepare(query, self.metric)[0]
        if self.metric == "cosine":
            return self.vectors @ q  # similarity, higher = nearer
        return np.sum((self.vectors - q) ** 2, axis=1)  # distance, lower = nearer

    def search(self, query: np.ndarray, k: int = 10):
        """Return ``(indices, scores)`` for the top-k neighbours.

        Scores are cosine similarities for ``cosine`` and squared-L2
        distances for ``l2``.
        """
        if self.vectors is None or len(self) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        if k <= 0:
            raise ValueError("k must be positive")
        k = min(k, len(self))
        scores = self._scores(query)
        if self.metric == "cosine":
            idx = np.argpartition(-scores, k - 1)[:k]
            order = np.argsort(-scores[idx])
        else:
            idx = np.argpartition(scores, k - 1)[:k]
            order = np.argsort(scores[idx])
        idx = idx[order]
        return idx.astype(np.int64), scores[idx].astype(np.float32)

    def search_batch(self, queries: np.ndarray, k: int = 10):
        """Batched top-k: returns (indices, scores) of shape (n_queries, k)."""
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[None, :]
        out_idx, out_scores = [], []
        for q in queries:
            idx, sc = self.search(q, k)
            out_idx.append(idx)
            out_scores.append(sc)
        return np.array(out_idx), np.array(out_scores)
