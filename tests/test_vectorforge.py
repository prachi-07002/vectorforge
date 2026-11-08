"""Unit tests for VectorForge (stdlib unittest only)."""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vectorforge import BruteForceIndex, HNSWIndex, IVFIndex, VectorForge  # noqa: E402
from vectorforge.benchmark import recall_at_k  # noqa: E402
from vectorforge.quant import ProductQuantizer, ScalarQuantizer  # noqa: E402


def clustered(n=2000, dim=32, clusters=10, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(clusters, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    labels = rng.integers(0, clusters, size=n)
    vecs = (centers[labels] + 0.3 * rng.normal(size=(n, dim))).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    q_labels = rng.integers(0, clusters, size=20)
    queries = (centers[q_labels] + 0.3 * rng.normal(size=(20, dim))).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    return vecs, queries


class TestBruteForce(unittest.TestCase):
    def test_matches_naive_loop_cosine(self):
        rng = np.random.default_rng(1)
        vecs = rng.normal(size=(50, 16)).astype(np.float32)
        idx = BruteForceIndex(metric="cosine")
        idx.add(vecs)
        q = rng.normal(size=16).astype(np.float32)
        got_idx, got_scores = idx.search(q, k=5)
        # naive loop ground truth
        qn = q / np.linalg.norm(q)
        sims = [(float(vecs[i] @ qn / np.linalg.norm(vecs[i])), i) for i in range(len(vecs))]
        sims.sort(reverse=True)
        self.assertEqual(list(got_idx), [i for _, i in sims[:5]])
        for s, (exp, _) in zip(got_scores, sims[:5]):
            self.assertAlmostEqual(float(s), exp, places=5)

    def test_l2_metric_correctness(self):
        idx = BruteForceIndex(metric="l2")
        idx.add(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]], dtype=np.float32))
        got_idx, got_scores = idx.search(np.array([0.1, 0.0], dtype=np.float32), k=2)
        self.assertEqual(list(got_idx), [0, 1])
        self.assertAlmostEqual(float(got_scores[0]), 0.01, places=5)
        self.assertAlmostEqual(float(got_scores[1]), 0.81, places=5)

    def test_empty_and_single_vector(self):
        idx = BruteForceIndex(metric="cosine")
        got_idx, _ = idx.search(np.array([1.0, 0.0], dtype=np.float32), k=5)
        self.assertEqual(len(got_idx), 0)
        idx.add(np.array([[1.0, 0.0]], dtype=np.float32))
        got_idx, got_scores = idx.search(np.array([1.0, 0.0], dtype=np.float32), k=5)
        self.assertEqual(list(got_idx), [0])
        self.assertAlmostEqual(float(got_scores[0]), 1.0, places=5)


class TestIVF(unittest.TestCase):
    def test_recall_reasonable(self):
        vecs, queries = clustered()
        brute = BruteForceIndex(metric="cosine")
        brute.add(vecs)
        gt, _ = brute.search_batch(queries, 10)
        ivf = IVFIndex(nlist=20, nprobe=5, metric="cosine")
        ivf.add(vecs)
        retrieved = np.array([ivf.search(q, 10)[0] for q in queries])
        self.assertGreaterEqual(recall_at_k(retrieved, gt, 10), 0.80)


class TestHNSW(unittest.TestCase):
    def test_recall_above_090_clustered(self):
        vecs, queries = clustered()
        brute = BruteForceIndex(metric="cosine")
        brute.add(vecs)
        gt, _ = brute.search_batch(queries, 10)
        hnsw = HNSWIndex(M=16, ef_construction=100, ef_search=64, metric="cosine")
        hnsw.add(vecs)
        retrieved = np.array([hnsw.search(q, 10)[0] for q in queries])
        self.assertGreaterEqual(recall_at_k(retrieved, gt, 10), 0.90)

    def test_l2_recall(self):
        rng = np.random.default_rng(3)
        vecs = rng.normal(size=(500, 16)).astype(np.float32)
        queries = rng.normal(size=(10, 16)).astype(np.float32)
        brute = BruteForceIndex(metric="l2")
        brute.add(vecs)
        gt, _ = brute.search_batch(queries, 10)
        hnsw = HNSWIndex(M=16, ef_construction=100, ef_search=64, metric="l2")
        hnsw.add(vecs)
        retrieved = np.array([hnsw.search(q, 10)[0] for q in queries])
        self.assertGreaterEqual(recall_at_k(retrieved, gt, 10), 0.85)

    def test_single_vector(self):
        hnsw = HNSWIndex(metric="cosine")
        hnsw.add(np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
        idx, scores = hnsw.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=5)
        self.assertEqual(list(idx), [0])


class TestQuantization(unittest.TestCase):
    def test_scalar_roundtrip_error_bound(self):
        rng = np.random.default_rng(5)
        vecs = rng.normal(size=(200, 32)).astype(np.float32)
        sq = ScalarQuantizer().fit(vecs)
        recon = sq.decode(sq.encode(vecs))
        # uniform 8-bit error per component is at most half a step
        step = float(np.max(sq.scale))
        self.assertLessEqual(np.abs(recon - vecs).max(), step / 2 + 1e-6)
        self.assertAlmostEqual(sq.compression_ratio(), 4.0, places=5)

    def test_product_quantizer_roundtrip_and_adc(self):
        rng = np.random.default_rng(6)
        vecs = rng.normal(size=(500, 32)).astype(np.float32)
        pq = ProductQuantizer(m=8).fit(vecs)
        codes = pq.encode(vecs)
        self.assertEqual(codes.shape, (500, 8))
        self.assertEqual(codes.dtype, np.uint8)
        recon = pq.decode(codes)
        # PQ reconstruction should beat the raw data scale by a wide margin
        rel_err = np.linalg.norm(recon - vecs) / np.linalg.norm(vecs)
        self.assertLess(rel_err, 0.6)
        self.assertAlmostEqual(pq.compression_ratio(), 16.0, places=5)
        # ADC distances should correlate strongly with exact L2 distances
        q = vecs[0]
        adc = pq.adc_distances(q, codes)
        exact = np.sum((vecs - q) ** 2, axis=1)
        self.assertGreater(np.corrcoef(adc, exact)[0, 1], 0.9)


class TestVectorForge(unittest.TestCase):
    def test_save_load_roundtrip(self):
        vecs, queries = clustered(n=500, dim=32)
        vf = VectorForge(dim=32, metric="cosine", ef_search=64)
        vf.add(vecs)
        before = [vf.search(q, 5)[0] for q in queries[:5]]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "index.npz")
            vf.save(path)
            vf2 = VectorForge.load(path)
            self.assertEqual(len(vf2), len(vf))
            for q, b in zip(queries[:5], before):
                idx, _ = vf2.search(q, 5)
                self.assertEqual(list(idx), list(b))

    def test_save_load_quantized(self):
        vecs, _ = clustered(n=300, dim=32)
        vf = VectorForge(dim=32, metric="cosine", quantizer="pq", pq_m=8)
        vf.add(vecs)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "index.npz")
            vf.save(path)
            vf2 = VectorForge.load(path)
            self.assertEqual(len(vf2), 300)
            self.assertAlmostEqual(vf2.compression_ratio(), 16.0, places=5)
            idx, _ = vf2.search(vecs[0], k=5)
            self.assertIn(0, list(idx))

    def test_empty_search(self):
        vf = VectorForge(dim=8, metric="cosine")
        idx, scores = vf.search(np.zeros(8, dtype=np.float32), k=5)
        self.assertEqual(len(idx), 0)
        self.assertEqual(len(scores), 0)


if __name__ == "__main__":
    unittest.main()
