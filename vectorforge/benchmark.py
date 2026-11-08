"""Benchmark harness: recall, QPS, latency, build time, memory, speedup.

All measurements are timed with ``time.perf_counter`` on the local
machine; recall@k is measured against the exact brute-force ground
truth computed on the same queries.
"""
from __future__ import annotations

import time

import numpy as np

from .brute import BruteForceIndex


def recall_at_k(retrieved: np.ndarray, ground_truth: np.ndarray, k: int = 10) -> float:
    """Mean fraction of the true top-k present in the retrieved top-k."""
    hits = 0
    for ret, gt in zip(retrieved, ground_truth):
        hits += len(set(ret[:k].tolist()) & set(gt[:k].tolist()))
    return hits / max(len(ground_truth) * k, 1)


def _time_queries(search_fn, queries: np.ndarray, k: int) -> tuple[float, list[float]]:
    """Run queries one at a time; return (qps, per-query latencies in ms)."""
    latencies = []
    start = time.perf_counter()
    for q in queries:
        t0 = time.perf_counter()
        search_fn(q, k)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    total = time.perf_counter() - start
    return len(queries) / max(total, 1e-9), latencies


def benchmark(index, vectors: np.ndarray, queries: np.ndarray, k: int = 10,
              name: str = "index", build_time: float = 0.0,
              ground_truth: np.ndarray | None = None,
              search_kwargs: dict | None = None) -> dict:
    """Benchmark one index against brute-force ground truth.

    Returns a dict with recall@k, QPS, p50/p99 latency (ms), build
    time (s), memory (MB, if the index exposes ``memory_bytes``) and
    speedup vs brute force.  Pass a shared ``ground_truth`` (from
    ``BruteForceIndex.search_batch``) when benchmarking several
    indexes on the same queries to avoid recomputing it each time.
    """
    brute = BruteForceIndex(metric=getattr(index, "metric", "cosine"))
    brute.add(vectors)
    if ground_truth is None:
        gt_idx, _ = brute.search_batch(queries, k)
    else:
        gt_idx = ground_truth
    search_kwargs = search_kwargs or {}

    retrieved = np.array([index.search(q, k, **search_kwargs)[0] for q in queries])
    rec = recall_at_k(retrieved, gt_idx, k)

    qps, lat = _time_queries(lambda q, kk: index.search(q, kk, **search_kwargs), queries, k)
    brute_qps, brute_lat = _time_queries(lambda q, kk: brute.search(q, kk), queries, k)

    mem_mb = None
    if hasattr(index, "memory_bytes"):
        mem_mb = index.memory_bytes() / 1e6
    elif getattr(index, "vectors", None) is not None:
        mem_mb = index.vectors.nbytes / 1e6

    return {
        "name": name,
        "recall@k": round(float(rec), 4),
        "qps": round(float(qps), 1),
        "p50_ms": round(float(np.percentile(lat, 50)), 3),
        "p99_ms": round(float(np.percentile(lat, 99)), 3),
        "build_s": round(float(build_time), 2),
        "memory_mb": round(float(mem_mb), 2) if mem_mb is not None else None,
        "brute_qps": round(float(brute_qps), 1),
        "brute_p50_ms": round(float(np.percentile(brute_lat, 50)), 3),
        "speedup_vs_brute": round(float(qps / max(brute_qps, 1e-9)), 2),
    }


def format_table(results: list[dict], k: int = 10) -> str:
    header = f"{'index':<22} {'recall@'+str(k):<10} {'QPS':>8} {'p50 ms':>8} {'p99 ms':>8} {'build s':>8} {'mem MB':>8} {'speedup':>8}"
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(
            f"{r['name']:<22} {r['recall@k']:<10.4f} {r['qps']:>8.1f} {r['p50_ms']:>8.3f} "
            f"{r['p99_ms']:>8.3f} {r['build_s']:>8.2f} "
            f"{r['memory_mb'] if r['memory_mb'] is not None else float('nan'):>8.2f} "
            f"{r['speedup_vs_brute']:>7.2f}x"
        )
    lines.append(f"\nBrute-force baseline: {results[0]['brute_qps']:.1f} QPS, "
                 f"p50 {results[0]['brute_p50_ms']:.3f} ms (recall 1.0000 by definition)")
    return "\n".join(lines)
