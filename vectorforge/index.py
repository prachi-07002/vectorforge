"""Unified index: HNSW graph + optional quantization.

:class:`VectorForge` is the one-class entry point: add vectors once,
search fast, optionally compress the stored vectors with scalar or
product quantization, and persist the whole index to disk.
"""
from __future__ import annotations

import os

import numpy as np

from .hnsw import HNSWIndex
from .quant import ProductQuantizer, ScalarQuantizer


class VectorForge:
    """Approximate nearest-neighbour index (HNSW, optionally quantized).

    Parameters
    ----------
    dim : int
        Vector dimensionality.
    metric : {"cosine", "l2"}
    M, ef_construction, ef_search : HNSW parameters (see :class:`HNSWIndex`).
    quantizer : {"none", "sq", "pq"}
        ``"sq"`` stores vectors as int8 (4x smaller); ``"pq"`` stores
        ``pq_m`` bytes per vector via product quantization.
    pq_m : int
        Number of PQ sub-quantizers (bytes per vector when ``pq``).
    """

    def __init__(self, dim: int, metric: str = "cosine", M: int = 16,
                 ef_construction: int = 64, ef_search: int = 64,
                 quantizer: str = "none", pq_m: int = 8) -> None:
        if metric not in ("cosine", "l2"):
            raise ValueError(f"unknown metric {metric!r}")
        if quantizer not in ("none", "sq", "pq"):
            raise ValueError(f"unknown quantizer {quantizer!r}")
        self.dim = dim
        self.metric = metric
        self.quantizer = quantizer
        self.pq_m = pq_m
        self._hnsw = HNSWIndex(M=M, ef_construction=ef_construction,
                               ef_search=ef_search, metric=metric)
        self._sq: ScalarQuantizer | None = None
        self._pq: ProductQuantizer | None = None
        self._codes: np.ndarray | None = None  # quantized codes, aligned with vectors
        self._raw: np.ndarray | None = None    # kept only when quantizer == "none"

    # -- building --------------------------------------------------------
    def add(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected shape (n, {self.dim})")
        if len(vectors) == 0:
            return
        self._hnsw.add(vectors)
        if self.quantizer == "none":
            self._raw = vectors if self._raw is None else np.vstack([self._raw, vectors])
        elif self.quantizer == "sq":
            if self._sq is None:
                self._sq = ScalarQuantizer(per_vector=False).fit(vectors)
            codes = self._sq.encode(vectors)
            self._codes = codes if self._codes is None else np.vstack([self._codes, codes])
        else:  # pq
            if self._pq is None:
                self._pq = ProductQuantizer(m=self.pq_m).fit(vectors)
            codes = self._pq.encode(vectors)
            self._codes = codes if self._codes is None else np.vstack([self._codes, codes])

    def __len__(self) -> int:
        return len(self._hnsw)

    # -- searching -------------------------------------------------------
    def search(self, query: np.ndarray, k: int = 10, ef_search: int | None = None):
        """Return ``(indices, scores)``; scores are distances (lower = nearer).

        With quantization enabled, candidates from the HNSW graph are
        re-ranked against the *quantized* vectors (SQ decode / PQ ADC),
        so the returned distances reflect the compressed representation.
        """
        idx, _ = self._hnsw.search(query, k=k, ef_search=ef_search)
        if len(idx) == 0:
            return idx, np.empty(0, dtype=np.float32)
        q = np.asarray(query, dtype=np.float32).ravel()
        if self.quantizer == "none":
            vecs = self._raw[idx] if self._raw is not None else self._hnsw.vectors[idx]
        elif self.quantizer == "sq":
            vecs = self._sq.decode(self._codes[idx])
        else:
            # PQ: asymmetric distance directly from codes, no decode.
            if self.metric == "l2":
                scores = self._pq.adc_distances(q, self._codes[idx])
                return idx, scores.astype(np.float32)
            # cosine on PQ codes: decode then normalise + dot
            vecs = self._pq.decode(self._codes[idx])
        if self.metric == "cosine":
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.where(norms == 0, 1.0, norms)
            qn = q / max(np.linalg.norm(q), 1e-12)
            scores = 1.0 - vecs @ qn
        else:
            scores = np.sum((vecs - q) ** 2, axis=1)
        order = np.argsort(scores)
        return idx[order].astype(np.int64), scores[order].astype(np.float32)

    # -- stats / persistence ----------------------------------------------
    def memory_bytes(self) -> int:
        """Bytes used by the stored vectors (quantized if applicable)."""
        if self.quantizer == "none":
            return 0 if self._raw is None else self._raw.nbytes
        return 0 if self._codes is None else self._codes.nbytes

    def compression_ratio(self) -> float:
        """float32 storage / actual storage for the vectors."""
        raw = 4 * self.dim * max(len(self), 1)
        return raw / max(self.memory_bytes(), 1)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        base = path[:-4] if path.endswith(".npz") else path
        self._hnsw.save(base + ".npz")
        extra = {"metric": np.array(self.metric), "quantizer": np.array(self.quantizer),
                 "dim": np.array(self.dim), "pq_m": np.array(self.pq_m)}
        if self._codes is not None:
            extra["codes"] = self._codes
        if self._sq is not None:
            extra["sq_scale"] = np.asarray(self._sq.scale, dtype=np.float32)
            extra["sq_zero"] = np.asarray(self._sq.zero_point, dtype=np.float32)
        if self._pq is not None:
            extra["pq_codebooks"] = self._pq.codebooks
        if self._raw is not None:
            extra["raw"] = self._raw
        np.savez(base + "_meta.npz", **extra)

    @classmethod
    def load(cls, path: str) -> "VectorForge":
        base = path[:-4] if path.endswith(".npz") else path
        meta = np.load(base + "_meta.npz", allow_pickle=False)
        metric = str(meta["metric"])
        quantizer = str(meta["quantizer"])
        dim = int(meta["dim"])
        vf = cls(dim=dim, metric=metric, quantizer=quantizer, pq_m=int(meta["pq_m"]))
        vf._hnsw = HNSWIndex.load(base + ".npz", metric=metric)
        if "codes" in meta:
            vf._codes = meta["codes"]
        if "raw" in meta:
            vf._raw = meta["raw"]
        if quantizer == "sq" and "sq_scale" in meta:
            sq = ScalarQuantizer(per_vector=False)
            sq.scale = meta["sq_scale"]
            sq.zero_point = meta["sq_zero"]
            sq.dim = dim
            vf._sq = sq
        if quantizer == "pq" and "pq_codebooks" in meta:
            pq = ProductQuantizer(m=int(meta["pq_m"]))
            pq.codebooks = meta["pq_codebooks"]
            pq.dim = dim
            pq.dsub = dim // pq.m
            vf._pq = pq
        return vf
