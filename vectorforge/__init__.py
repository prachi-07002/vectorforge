"""VectorForge — an approximate nearest-neighbour vector search engine.

Built from scratch in pure Python + NumPy: exact brute-force search,
IVF clustering, a real HNSW graph, and scalar / product quantization.
"""
from .benchmark import benchmark, recall_at_k
from .brute import BruteForceIndex
from .hnsw import HNSWIndex
from .index import VectorForge
from .ivf import IVFIndex
from .quant import ProductQuantizer, ScalarQuantizer

__all__ = [
    "VectorForge",
    "BruteForceIndex",
    "HNSWIndex",
    "IVFIndex",
    "ScalarQuantizer",
    "ProductQuantizer",
    "benchmark",
    "recall_at_k",
]

__version__ = "0.1.0"
