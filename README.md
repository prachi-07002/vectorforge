# VectorForge

An approximate nearest-neighbour (ANN) vector search engine built **from scratch** in pure Python + NumPy — no FAISS, no Annoy, no `scipy`, no network calls. The kind of thing that powers semantic search, retrieval-augmented generation (RAG), and recommendation systems: given a query embedding, find the most similar vectors in a large collection without scanning all of them.

## What's inside

| Module | What it implements |
|---|---|
| `vectorforge/brute.py` | Exact brute-force baseline (NumPy-vectorised cosine / L2 top-k) — also the ground truth for recall |
| `vectorforge/hnsw.py` | **Real HNSW graph** (Malkov & Yashunin, 2018): multi-layer navigable small-world graph, geometric level assignment, greedy layer descent, beam search (`ef_construction` / `ef_search`), and the paper's heuristic neighbour selection, which keeps diverse long-range links instead of near-duplicates |
| `vectorforge/ivf.py` | IVF index: k-means clustering (Lloyd's algorithm, k-means++ init) into Voronoi cells, inverted lists, `nprobe` routing |
| `vectorforge/quant.py` | `ScalarQuantizer` (float32 → int8, affine scale/zero-point, 4× smaller) and `ProductQuantizer` (sub-vector codebooks, asymmetric distance computation via lookup tables) |
| `vectorforge/index.py` | `VectorForge` — the unified index: HNSW + optional SQ/PQ quantization, `add` / `search` / `save` / `load`, cosine and L2 |
| `vectorforge/benchmark.py` | Benchmark harness: recall@k vs brute force, QPS, p50/p99 latency, build time, memory, speedup |

## Quickstart

```python
import numpy as np
from vectorforge import VectorForge

rng = np.random.default_rng(0)
vectors = rng.normal(size=(10_000, 128)).astype(np.float32)

index = VectorForge(dim=128, metric="cosine")   # HNSW, M=16, ef_search=64
index.add(vectors)

ids, distances = index.search(vectors[0], k=10)  # distances: 1 - cosine similarity

# Compressed variant: 32x less memory, nearly the same recall
small = VectorForge(dim=128, metric="cosine", quantizer="pq", pq_m=16)
small.add(vectors)
print(f"compression: {small.compression_ratio():.0f}x")

index.save("my_index.npz")
restored = VectorForge.load("my_index.npz")
```

```bash
pip install -r requirements.txt   # numpy only
python3 demo.py                   # semantic search + synthetic benchmark
python3 benchmarks/bench.py       # full benchmark table
python3 -m unittest discover -s tests -v
```

## Benchmarks (measured, not estimated)

Setup: 10,000 vectors, dim 128, 500 tight Gaussian clusters × 20 vectors (the shape real embedding collections have — they cluster by topic), 100 queries, cosine, k=10, measured on this machine with `python3 benchmarks/bench.py`. Recall@10 is measured against the exact brute-force top-10.

| index | recall@10 | QPS | p50 (ms) | p99 (ms) | build (s) | memory | speedup vs brute |
|---|---:|---:|---:|---:|---:|---:|---:|
| HNSW (ef=64) | 0.9460 | 1,121.9 | 0.871 | 1.426 | 19.98 | 5.12 MB | 1.26× |
| HNSW (ef=32) | 0.9160 | 1,905.2 | 0.511 | 0.935 | 19.98 | 5.12 MB | 2.16× |
| IVF (nlist=100, nprobe=10) | 1.0000 | 3,616.0 | 0.277 | 0.331 | 2.20 | 5.12 MB | 4.09× |
| HNSW + PQ (16 B/vec) | 0.9330 | 1,063.8 | 0.928 | 1.457 | 36.66 | 0.16 MB | 1.19× |
| Brute force (exact) | 1.0000 | 893.8 | 1.114 | — | — | 5.12 MB | 1.00× |

Other measured numbers:

- **PQ compression: 32×** — 512 B → 16 B per vector (dim 128 float32 → 16 product-quantization bytes), at recall@10 0.9330 (vs 0.9460 unquantized).
- **Scalar quantization: 4×** smaller (float32 → int8) with per-component reconstruction error bounded by half a quantization step (verified in tests).
- **Demo:** `python3 demo.py` runs semantic search over a 16-document corpus (TF-IDF embeddings, no external model) and the same 10k-vector benchmark — HNSW recall@10 0.9460 at 1,138.7 QPS, p50 0.863 ms.

Notes on the numbers: exact recall and QPS depend on the data distribution and hardware. The headline trade-off is the point — `ef_search` trades recall for QPS (0.916 @ 1,905 QPS vs 0.946 @ 1,122 QPS), IVF trades a little recall for a large speed win by scanning only 10 of 100 clusters, and PQ trades a little recall for 32× less memory. Brute force is a single BLAS matvec, so ANN only wins on speed once the collection is large enough for the graph/cluster routing to beat a full scan — at 10k vectors HNSW is 1.2–2.2× faster; the gap grows with collection size, since brute-force cost grows linearly while HNSW/IVF cost grows sub-linearly.

## Resume bullets

- Built VectorForge, an approximate nearest-neighbour vector search engine from scratch in Python + NumPy (HNSW graph, IVF clustering, scalar/product quantization), with no ANN libraries
- Achieved recall@10 of 0.946 at 1,122 QPS with HNSW (ef=64), 1.26× faster than exact brute-force search (894 QPS) on 10,000 × 128-dim vectors; IVF variant reaches 3,616 QPS (4.09× speedup) at recall@10 1.0000
- Compressed the index 32× with product quantization (512 B → 16 B per vector) while keeping recall@10 at 0.933, and 4× with int8 scalar quantization
- Implemented the HNSW heuristic neighbour-selection algorithm to preserve diverse long-range graph links, plus a benchmark harness reporting recall@k, QPS, and p50/p99 latency against brute-force ground truth; 12/12 unit tests pass

## Design notes

- **Why heuristic neighbour selection matters:** keeping only the *closest* M neighbours fills every node's list with near-duplicates from its own cluster, which disconnects the graph into components and caps recall (~0.66 even at very large `ef_search` on clustered data). The HNSW heuristic keeps a candidate only if it is closer to the node than to any already-selected neighbour, preserving long-range links; that fix took recall@10 from 0.66 to 0.95 on the same data.
- **PQ search uses asymmetric distance computation (ADC):** the query stays unquantized and is compared against codebook centroids through a lookup table, so scoring quantized vectors costs O(m) table lookups per candidate instead of a full decode.
- **Everything is text:** the whole project is pure Python + NumPy, no binary assets, so it pushes to GitHub cleanly.

## Tests

12 unit tests (`tests/test_vectorforge.py`, stdlib `unittest` only) covering brute-force correctness against a naive loop, metric correctness, IVF recall, HNSW recall > 0.90 on clustered synthetic data, quantization round-trip error bounds and ADC correlation, save/load round-trips (plain and quantized), and empty/single-vector edge cases:

```bash
python3 -m unittest discover -s tests -v   # 12/12 pass
```
