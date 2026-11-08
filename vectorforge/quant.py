"""Quantization: compress float32 vectors to 8-bit codes.

Two schemes, both implemented from scratch:

* :class:`ScalarQuantizer` — uniform per-dimension (or per-vector)
  affine quantization to ``uint8``: ``x ~= scale * (code - zero_point)``.
  4x memory reduction (32 -> 8 bits per component).
* :class:`ProductQuantizer` — split a vector into ``m`` sub-vectors and
  learn a small k-means codebook (``k = 256`` centroids) per sub-space.
  Each sub-vector is replaced by one byte, so a ``dim``-vector costs
  ``m`` bytes instead of ``4 * dim``.  Distances are estimated with
  asymmetric distance computation (ADC): the query stays unquantized
  and is compared against codebook centroids via a lookup table.
"""
from __future__ import annotations

import numpy as np


class ScalarQuantizer:
    """Uniform affine quantization to uint8.

    Parameters
    ----------
    per_vector : bool
        If True, each vector gets its own scale/zero-point (tighter
        error bounds, 8 extra bytes per vector).  If False, one global
        scale/zero-point is fitted to the training data.
    """

    def __init__(self, per_vector: bool = False) -> None:
        self.per_vector = per_vector
        self.scale: np.ndarray | float | None = None
        self.zero_point: np.ndarray | float | None = None
        self.dim: int | None = None

    def fit(self, vectors: np.ndarray) -> "ScalarQuantizer":
        vectors = np.asarray(vectors, dtype=np.float32)
        self.dim = vectors.shape[1]
        if self.per_vector:
            lo = vectors.min(axis=1, keepdims=True)
            hi = vectors.max(axis=1, keepdims=True)
        else:
            lo = vectors.min()
            hi = vectors.max()
        rng = np.maximum(hi - lo, 1e-8)
        self.scale = (rng / 255.0).astype(np.float32)
        self.zero_point = lo.astype(np.float32)
        return self

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        if self.scale is None:
            raise RuntimeError("fit() must be called before encode()")
        vectors = np.asarray(vectors, dtype=np.float32)
        codes = np.round((vectors - self.zero_point) / self.scale)
        return np.clip(codes, 0, 255).astype(np.uint8)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        if self.scale is None:
            raise RuntimeError("fit() must be called before decode()")
        codes = np.asarray(codes, dtype=np.float32)
        return (codes * self.scale + self.zero_point).astype(np.float32)

    @property
    def bytes_per_vector(self) -> int:
        extra = 8 if self.per_vector else 0
        return (self.dim or 0) + extra

    def compression_ratio(self) -> float:
        """float32 bytes / quantized bytes per vector."""
        if not self.dim:
            return 0.0
        return (4 * self.dim) / max(self.bytes_per_vector, 1)


def _kmeans(data: np.ndarray, k: int, n_iter: int = 20, seed: int = 0) -> np.ndarray:
    """Small k-means (k-means++ init) used to train PQ codebooks."""
    rng = np.random.default_rng(seed)
    n = len(data)
    if n == 0:
        raise ValueError("empty data for k-means")
    k = min(k, n)
    # k-means++ seeding
    centroids = [data[rng.integers(n)]]
    closest_sq = np.sum((data - centroids[0]) ** 2, axis=1)
    for _ in range(1, k):
        probs = closest_sq / max(closest_sq.sum(), 1e-12)
        idx = rng.choice(n, p=probs)
        centroids.append(data[idx])
        closest_sq = np.minimum(closest_sq, np.sum((data - data[idx]) ** 2, axis=1))
    centroids = np.array(centroids, dtype=np.float32)
    for _ in range(n_iter):
        d = ((data[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = d.argmin(axis=1)
        new = centroids.copy()
        for j in range(k):
            members = data[labels == j]
            if len(members):
                new[j] = members.mean(axis=0)
        if np.allclose(new, centroids):
            break
        centroids = new
    return centroids


class ProductQuantizer:
    """Product quantization with ``m`` sub-quantizers, 256 centroids each.

    Parameters
    ----------
    m : int
        Number of sub-vectors (bytes per encoded vector). ``dim`` must
        be divisible by ``m``.
    n_iter : int
        k-means iterations per sub-quantizer during training.
    """

    def __init__(self, m: int = 8, n_iter: int = 10) -> None:
        self.m = m
        self.n_iter = n_iter
        self.codebooks: np.ndarray | None = None  # (m, 256, dsub)
        self.dim: int | None = None
        self.dsub: int | None = None

    def fit(self, vectors: np.ndarray, seed: int = 0) -> "ProductQuantizer":
        vectors = np.asarray(vectors, dtype=np.float32)
        n, dim = vectors.shape
        if dim % self.m:
            raise ValueError(f"dim {dim} not divisible by m={self.m}")
        self.dim, self.dsub = dim, dim // self.m
        books = []
        for j in range(self.m):
            sub = vectors[:, j * self.dsub:(j + 1) * self.dsub]
            books.append(_kmeans(sub, 256, self.n_iter, seed + j))
        self.codebooks = np.stack(books).astype(np.float32)
        return self

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        if self.codebooks is None:
            raise RuntimeError("fit() must be called before encode()")
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        codes = np.empty((len(vectors), self.m), dtype=np.uint8)
        for j in range(self.m):
            sub = vectors[:, j * self.dsub:(j + 1) * self.dsub]
            d = ((sub[:, None, :] - self.codebooks[j][None, :, :]) ** 2).sum(axis=2)
            codes[:, j] = d.argmin(axis=1)
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        if self.codebooks is None:
            raise RuntimeError("fit() must be called before decode()")
        codes = np.asarray(codes)
        if codes.ndim == 1:
            codes = codes[None, :]
        out = np.empty((len(codes), self.dim), dtype=np.float32)
        for j in range(self.m):
            out[:, j * self.dsub:(j + 1) * self.dsub] = self.codebooks[j][codes[:, j]]
        return out

    def adc_distances(self, query: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """Asymmetric L2^2 distances: raw query vs quantized codes.

        Builds one (m, 256) lookup table of sub-vector distances, then
        sums the looked-up entries — O(m * 256 * dsub + n * m) instead
        of O(n * dim).
        """
        if self.codebooks is None:
            raise RuntimeError("fit() must be called first")
        q = np.asarray(query, dtype=np.float32).ravel()
        codes = np.asarray(codes)
        table = np.empty((self.m, 256), dtype=np.float32)
        for j in range(self.m):
            sub_q = q[j * self.dsub:(j + 1) * self.dsub]
            table[j] = np.sum((self.codebooks[j] - sub_q) ** 2, axis=1)
        return table[np.arange(self.m)[None, :], codes].sum(axis=1)

    @property
    def bytes_per_vector(self) -> int:
        return self.m

    def compression_ratio(self) -> float:
        if not self.dim:
            return 0.0
        return (4 * self.dim) / self.m
