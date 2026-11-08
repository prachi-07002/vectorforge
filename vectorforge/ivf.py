"""IVF (inverted file) index.

Vectors are clustered with k-means into ``nlist`` Voronoi cells; each
cell holds an inverted list of its members.  A query is routed to the
``nprobe`` nearest centroids and only those lists are scanned (with a
NumPy-vectorised scan), so search touches a fraction of the data.
Recall is traded against speed by tuning ``nprobe``.
"""
from __future__ import annotations

import numpy as np

from .quant import _kmeans


class IVFIndex:
    """Inverted-file index with k-means partitioning.

    Parameters
    ----------
    nlist : int
        Number of Voronoi cells (clusters).
    nprobe : int
        Number of cells scanned per query.
    metric : {"cosine", "l2"}
    """

    def __init__(self, nlist: int = 100, nprobe: int = 10, metric: str = "cosine") -> None:
        if metric not in ("cosine", "l2"):
            raise ValueError(f"unknown metric {metric!r}")
        self.nlist = nlist
        self.nprobe = nprobe
        self.metric = metric
        self.centroids: np.ndarray | None = None
        self.lists: list[list[int]] = []
        self.vectors: np.ndarray | None = None
        self.dim: int | None = None

    # -- helpers ---------------------------------------------------------
    def _prep(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        if self.metric == "cosine":
            norms = np.linalg.norm(x, axis=1, keepdims=True)
            return (x / np.where(norms == 0, 1.0, norms)).astype(np.float32)
        return x

    def _assign(self, vectors: np.ndarray) -> np.ndarray:
        if self.metric == "cosine":
            sims = vectors @ self.centroids.T
            return sims.argmax(axis=1)
        d = ((vectors[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2)
        return d.argmin(axis=1)

    # -- public API ------------------------------------------------------
    def train(self, vectors: np.ndarray, seed: int = 0) -> "IVFIndex":
        vectors = self._prep(vectors)
        self.dim = vectors.shape[1]
        self.nlist = min(self.nlist, len(vectors))
        self.centroids = _kmeans(vectors, self.nlist, n_iter=25, seed=seed)
        if self.metric == "cosine":
            norms = np.linalg.norm(self.centroids, axis=1, keepdims=True)
            self.centroids = (self.centroids / np.where(norms == 0, 1.0, norms)).astype(np.float32)
        return self

    def add(self, vectors: np.ndarray) -> None:
        vectors = self._prep(vectors)
        if self.centroids is None:
            self.train(vectors)
        if self.vectors is None:
            self.vectors = vectors
            self.lists = [[] for _ in range(self.nlist)]
            base = 0
        else:
            base = len(self.vectors)
            self.vectors = np.vstack([self.vectors, vectors])
        labels = self._assign(vectors)
        for i, lab in enumerate(labels):
            self.lists[int(lab)].append(base + i)

    def __len__(self) -> int:
        return 0 if self.vectors is None else len(self.vectors)

    def search(self, query: np.ndarray, k: int = 10, nprobe: int | None = None):
        if self.vectors is None or len(self) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        k = min(k, len(self))
        nprobe = min(nprobe or self.nprobe, self.nlist)
        q = self._prep(query)[0]
        if self.metric == "cosine":
            cell_scores = self.centroids @ q
            cells = np.argpartition(-cell_scores, nprobe - 1)[:nprobe]
        else:
            cell_d = np.sum((self.centroids - q) ** 2, axis=1)
            cells = np.argpartition(cell_d, nprobe - 1)[:nprobe]
        cand = np.concatenate([np.array(self.lists[c], dtype=np.int64) for c in cells
                               if self.lists[c]]) if any(self.lists[c] for c in cells) \
            else np.empty(0, dtype=np.int64)
        if len(cand) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        cand_vecs = self.vectors[cand]
        if self.metric == "cosine":
            scores = cand_vecs @ q
            take = min(k, len(cand))
            idx = np.argpartition(-scores, take - 1)[:take]
            order = np.argsort(-scores[idx])
        else:
            scores = np.sum((cand_vecs - q) ** 2, axis=1)
            take = min(k, len(cand))
            idx = np.argpartition(scores, take - 1)[:take]
            order = np.argsort(scores[idx])
        idx = idx[order]
        return cand[idx].astype(np.int64), scores[idx].astype(np.float32)
