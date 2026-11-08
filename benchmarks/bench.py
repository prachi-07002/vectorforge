"""Runnable benchmark: 10,000 clustered vectors (dim 128), 100 queries.

Data are 500 tight Gaussian clusters x 20 vectors (a realistic shape
for embedding collections, which cluster by topic).  Builds one HNSW
graph (benchmarked at ef_search 32 and 64), an IVF index, and a
PQ-quantized HNSW variant, and prints a comparison table with
recall@10, QPS, p50/p99 latency, build time, memory and speedup vs
the exact brute-force baseline.

Usage:  python3 benchmarks/bench.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vectorforge import VectorForge, benchmark  # noqa: E402
from vectorforge.benchmark import format_table as _ft  # noqa: E402
from vectorforge.brute import BruteForceIndex  # noqa: E402
from vectorforge.ivf import IVFIndex  # noqa: E402


def make_data(n: int = 10_000, dim: int = 128, n_clusters: int = 500,
              n_queries: int = 100, seed: int = 42):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, dim)).astype(np.float32)
    labels = rng.integers(0, n_clusters, size=n)
    vecs = (centers[labels] + 0.3 * rng.normal(size=(n, dim))).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    q_labels = rng.integers(0, n_clusters, size=n_queries)
    queries = (centers[q_labels] + 0.3 * rng.normal(size=(n_queries, dim))).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    return vecs, queries


def main() -> None:
    vecs, queries = make_data()
    print(f"Dataset: {len(vecs)} vectors, dim {vecs.shape[1]}, "
          f"500 clusters, {len(queries)} queries, cosine, k=10")
    brute = BruteForceIndex(metric="cosine")
    brute.add(vecs)
    gt, _ = brute.search_batch(queries, 10)
    results = []

    # One HNSW graph, benchmarked at two query-time beam widths.
    t0 = time.perf_counter()
    hnsw = VectorForge(dim=128, metric="cosine", M=16, ef_construction=64, ef_search=64)
    hnsw.add(vecs)
    build = time.perf_counter() - t0
    results.append(benchmark(hnsw, vecs, queries, k=10, name="HNSW (ef=64)",
                             build_time=build, ground_truth=gt,
                             search_kwargs={"ef_search": 64}))
    results.append(benchmark(hnsw, vecs, queries, k=10, name="HNSW (ef=32)",
                             build_time=build, ground_truth=gt,
                             search_kwargs={"ef_search": 32}))

    t0 = time.perf_counter()
    ivf = IVFIndex(nlist=100, nprobe=10, metric="cosine")
    ivf.add(vecs)
    results.append(benchmark(ivf, vecs, queries, k=10, name="IVF (nlist=100, nprobe=10)",
                             build_time=time.perf_counter() - t0, ground_truth=gt))

    t0 = time.perf_counter()
    pq = VectorForge(dim=128, metric="cosine", M=16, ef_construction=40, ef_search=64,
                     quantizer="pq", pq_m=16)
    pq.add(vecs)
    r = benchmark(pq, vecs, queries, k=10, name="HNSW + PQ (16 B/vec)",
                  build_time=time.perf_counter() - t0, ground_truth=gt,
                  search_kwargs={"ef_search": 64})
    r["memory_mb"] = round(pq.memory_bytes() / 1e6, 2)
    results.append(r)

    print()
    print(_ft(results, k=10))
    print(f"\nPQ compression: {pq.compression_ratio():.1f}x "
          f"(512 B -> {pq.memory_bytes() // len(vecs)} B per vector)")


if __name__ == "__main__":
    main()
