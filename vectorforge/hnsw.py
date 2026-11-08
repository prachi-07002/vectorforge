"""HNSW: Hierarchical Navigable Small World graph index.

A faithful implementation of Malkov & Yashunin (2018), built from
scratch on NumPy:

* Each vector is inserted at a random level drawn from a geometric
  distribution (``level = floor(-ln(U) * mL)``, ``mL = 1 / ln(M)``),
  so higher layers are exponentially sparser and act as highways.
* Search descends greedily layer by layer (``ef = 1`` above layer 0),
  then runs a beam search of width ``ef_search`` on layer 0.
* Neighbour distances inside one expansion step are computed as a
  single batched NumPy matvec over that node's neighbour list, which
  keeps the pure-Python graph walk fast.
* Neighbour selection uses the HNSW heuristic (keep a candidate only
  if it is closer to the node than to any already-selected neighbour,
  topping up with the closest rest), capped at ``M`` (``2 * M`` on
  layer 0); links are bidirectional and pruned the same way.

Cosine search works on L2-normalised vectors, so ``distance = 1 - dot``;
L2 search uses squared Euclidean distance.
"""
from __future__ import annotations

import heapq
import math

import numpy as np


class HNSWIndex:
    """HNSW approximate k-NN index.

    Parameters
    ----------
    M : int
        Max neighbours per node per layer (``2 * M`` on layer 0).
    ef_construction : int
        Beam width used while inserting.
    ef_search : int
        Beam width used at query time (raise for higher recall).
    metric : {"cosine", "l2"}
    seed : int
    """

    def __init__(self, M: int = 16, ef_construction: int = 64,
                 ef_search: int = 64, metric: str = "cosine", seed: int = 0) -> None:
        if metric not in ("cosine", "l2"):
            raise ValueError(f"unknown metric {metric!r}")
        self.M = M
        self.M_max0 = 2 * M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.metric = metric
        self.rng = np.random.default_rng(seed)
        self._mL = 1.0 / math.log(M)
        self.vectors: np.ndarray | None = None
        self.dim: int | None = None
        # graph[node][layer] -> list[int] of neighbours
        self.graph: list[list[list[int]]] = []
        self.levels: list[int] = []
        self.entry_point: int | None = None
        self.max_level: int = -1

    # -- internals -------------------------------------------------------
    def _prep(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        if self.metric == "cosine":
            norms = np.linalg.norm(x, axis=1, keepdims=True)
            return (x / np.where(norms == 0, 1.0, norms)).astype(np.float32)
        return x

    def _dist_batch(self, q: np.ndarray, ids: list[int] | np.ndarray) -> np.ndarray:
        """Distances from q to vectors[ids]; lower is nearer."""
        ids = np.asarray(ids, dtype=np.int64)
        vecs = self.vectors[ids]
        if self.metric == "cosine":
            return (1.0 - vecs @ q).astype(np.float32)
        return np.sum((vecs - q) ** 2, axis=1).astype(np.float32)

    def _dist_one(self, q: np.ndarray, idx: int) -> float:
        if self.metric == "cosine":
            return float(1.0 - self.vectors[idx] @ q)
        d = self.vectors[idx] - q
        return float(d @ d)

    def _random_level(self) -> int:
        return int(math.floor(-math.log(max(self.rng.random(), 1e-12)) * self._mL))

    def _search_layer(self, q: np.ndarray, entry: int, ef: int, level: int):
        """Beam search on one layer. Returns list of (dist, id), best first."""
        visited = {entry}
        entry_dist = self._dist_one(q, entry)
        candidates = [(entry_dist, entry)]          # min-heap
        results = [(-entry_dist, entry)]            # max-heap via negation
        heapq.heapify(candidates)
        heapq.heapify(results)
        while candidates:
            dist, cur = heapq.heappop(candidates)
            worst = -results[0][0]
            if dist > worst and len(results) >= ef:
                break
            nbrs = self.graph[cur][level]
            fresh = [n for n in nbrs if n not in visited]
            if not fresh:
                continue
            visited.update(fresh)
            dists = self._dist_batch(q, fresh)
            for nb, d in zip(fresh, dists):
                d = float(d)
                if len(results) < ef or d < -results[0][0]:
                    heapq.heappush(candidates, (d, nb))
                    heapq.heappush(results, (-d, nb))
                    if len(results) > ef:
                        heapq.heappop(results)
        out = sorted((-neg_d, i) for neg_d, i in results)
        return out

    def _select_neighbours(self, candidates, cap: int, qvec: np.ndarray) -> list[int]:
        """HNSW heuristic neighbour selection (Malkov & Yashunin, Alg. 4).

        Candidates are scanned closest-first. A candidate is kept only
        if it is closer to the query node than to every neighbour
        already selected -- this preserves diverse, long-range links
        instead of filling the list with near-duplicates from one
        cluster (which would disconnect the graph into components and
        cap recall). If the heuristic yields too few neighbours, the
        list is topped up with the closest remaining candidates.
        """
        ordered = sorted(candidates)
        # One batched pairwise-distance matrix for all candidates, so
        # the heuristic loop below is pure Python over a small array
        # instead of one NumPy call per candidate.
        cand_ids = [cid for _, cid in ordered]
        cand_vecs = self.vectors[np.asarray(cand_ids, dtype=np.int64)]
        if self.metric == "cosine":
            pairwise = (1.0 - cand_vecs @ cand_vecs.T).tolist()
        else:
            sq = np.sum(cand_vecs ** 2, axis=1)
            pairwise = (sq[:, None] + sq[None, :] - 2.0 * (cand_vecs @ cand_vecs.T)).tolist()
        # Pure-Python check over the (small) pairwise matrix: a NumPy
        # call per candidate here costs more than the arithmetic itself.
        selected: list[int] = []
        selected_pos: list[int] = []
        for pos, (d, cid) in enumerate(ordered):
            if len(selected) >= cap:
                break
            row = pairwise[pos]
            if any(row[sp] <= d for sp in selected_pos):
                continue
            selected.append(cid)
            selected_pos.append(pos)
        if len(selected) < cap:
            chosen = set(selected)
            for _, cid in ordered:
                if cid not in chosen:
                    selected.append(cid)
                    chosen.add(cid)
                    if len(selected) >= cap:
                        break
        return selected

    def _connect(self, node: int, neighbours: list[int], level: int) -> None:
        cap = self.M_max0 if level == 0 else self.M
        neighbours = neighbours[:cap]
        self.graph[node][level] = list(neighbours)
        for nb in neighbours:
            # add the reverse link, pruning the neighbour if over cap
            links = self.graph[nb][level]
            if node not in links:
                links.append(node)
                if len(links) > cap:
                    # Cheap closest-only pruning for the reverse link:
                    # the new node's *own* list (selected heuristically
                    # above) already holds its diverse, long-range
                    # links, so dropping the farthest reverse link
                    # here does not remove that diversity from the
                    # graph -- it only avoids re-running the full
                    # heuristic for every neighbour on every insert.
                    q = self.vectors[nb]
                    dists = self._dist_batch(q, links)
                    order = np.argsort(dists)[:cap]
                    self.graph[nb][level] = [links[j] for j in order]

    # -- public API ------------------------------------------------------
    def add(self, vectors: np.ndarray) -> None:
        vectors = self._prep(vectors)
        if self.dim is None:
            self.dim = vectors.shape[1]
        elif vectors.shape[1] != self.dim:
            raise ValueError("dimension mismatch")
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        start = len(self.graph)
        for local, vec in enumerate(vectors):
            idx = start + local
            level = self._random_level()
            self.levels.append(level)
            self.graph.append([[] for _ in range(level + 1)])
            if self.entry_point is None:
                self.entry_point = idx
                self.max_level = level
                continue
            entry = self.entry_point
            # Greedy descent from the top layer to level+1.
            cur_level = self.max_level
            while cur_level > level:
                best = self._search_layer(vec, entry, ef=1, level=cur_level)
                entry = best[0][1]
                cur_level -= 1
            # Beam search + connect on every layer down to 0.
            for layer in range(min(level, self.max_level), -1, -1):
                ef = self.ef_construction
                cands = self._search_layer(vec, entry, ef=ef, level=layer)
                cap = self.M_max0 if layer == 0 else self.M
                chosen = self._select_neighbours(cands, cap, vec)
                self._connect(idx, chosen, layer)
                entry = cands[0][1]
            if level > self.max_level:
                self.entry_point = idx
                self.max_level = level

    def __len__(self) -> int:
        return len(self.graph)

    def search(self, query: np.ndarray, k: int = 10, ef_search: int | None = None):
        """Return ``(indices, scores)`` — scores are distances (lower = nearer).

        For cosine, the returned score is ``1 - cosine_similarity``;
        for L2 it is the squared Euclidean distance.
        """
        if self.entry_point is None or len(self) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        k = min(k, len(self))
        ef = max(ef_search or self.ef_search, k)
        q = self._prep(query)[0]
        entry = self.entry_point
        for layer in range(self.max_level, 0, -1):
            best = self._search_layer(q, entry, ef=1, level=layer)
            entry = best[0][1]
        results = self._search_layer(q, entry, ef=ef, level=0)[:k]
        idx = np.array([i for _, i in results], dtype=np.int64)
        scores = np.array([d for d, _ in results], dtype=np.float32)
        return idx, scores

    # -- persistence (padded adjacency, npz) ------------------------------
    def save(self, path: str) -> None:
        n = len(self)
        max_deg = max((len(links) for node in self.graph for links in node), default=0)
        n_layers = self.max_level + 1
        adj = np.full((n, n_layers, max_deg), -1, dtype=np.int32)
        for i, node in enumerate(self.graph):
            for layer, links in enumerate(node):
                adj[i, layer, :len(links)] = links
        np.savez(path, vectors=self.vectors, levels=np.array(self.levels, dtype=np.int32),
                 adj=adj, entry_point=np.array(self.entry_point if self.entry_point is not None else -1),
                 max_level=np.array(self.max_level),
                 meta=np.array([self.M, self.ef_construction, self.ef_search]))

    @classmethod
    def load(cls, path: str, metric: str = "cosine") -> "HNSWIndex":
        data = np.load(path, allow_pickle=False)
        meta = data["meta"]
        idx = cls(M=int(meta[0]), ef_construction=int(meta[1]), ef_search=int(meta[2]), metric=metric)
        idx.vectors = data["vectors"]
        idx.dim = idx.vectors.shape[1]
        idx.levels = [int(x) for x in data["levels"]]
        adj = data["adj"]
        idx.graph = []
        for i in range(len(idx.levels)):
            node = []
            for layer in range(idx.levels[i] + 1):
                node.append([int(x) for x in adj[i, layer] if x >= 0])
            idx.graph.append(node)
        idx.entry_point = int(data["entry_point"]) if int(data["entry_point"]) >= 0 else None
        idx.max_level = int(data["max_level"])
        return idx
