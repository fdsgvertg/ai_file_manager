"""
models/embedding_client.py — Sentence-Transformer embeddings + FAISS-backed
semantic clustering. Determines cluster count automatically via elbow method.
"""

from __future__ import annotations
import threading
from typing import List, Optional, Tuple

import numpy as np

from utils.config import load_config
from utils.logger import get_logger

log = get_logger("embedding_client")


class EmbeddingClient:
    """
    Thin wrapper around sentence-transformers for generating text embeddings.
    Runs on CPU to preserve VRAM for vision / LLM.
    """

    _instance: Optional["EmbeddingClient"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._cfg = load_config().models.embedding
        self._model = None
        self._model_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "EmbeddingClient":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError("sentence-transformers not installed: pip install sentence-transformers")

        model_id = self._cfg.model_id
        device = str(self._cfg.device)
        log.info(f"Loading embedding model: {model_id} on {device}")
        self._model = SentenceTransformer(model_id, device=device)
        log.info("Embedding model loaded ✓")

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts. Returns float32 array of shape (N, D).
        Handles batching internally.
        """
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)

        with self._model_lock:
            self._load_model()
            batch_size = int(self._cfg.batch_size)
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        return embeddings.astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


# ─── Clustering ───────────────────────────────────────────────────────────────

class SemanticClusterer:
    """
    Clusters file embeddings using KMeans (default) or DBSCAN.
    Auto-determines cluster count via elbow method when n_clusters_auto=True.
    """

    def __init__(self):
        cfg = load_config().clustering
        self.algorithm: str = str(cfg.algorithm)
        self.n_clusters_auto: bool = bool(cfg.n_clusters_auto)
        self.min_cluster_size: int = int(cfg.min_cluster_size)
        self.eps: float = float(cfg.eps)
        self._embedder = EmbeddingClient.get_instance()

    def cluster_texts(self, texts: List[str]) -> List[str]:
        """
        Given a list of text descriptions, return a list of cluster labels
        of the same length (e.g. ["cluster_0", "cluster_1", ...]).
        """
        if len(texts) < 2:
            return ["cluster_0"] * len(texts)

        embeddings = self._embedder.embed(texts)
        return self._run_clustering(embeddings, len(texts))

    def cluster_embeddings(self, embeddings: np.ndarray) -> List[str]:
        """Cluster pre-computed embeddings directly."""
        if embeddings.shape[0] < 2:
            return ["cluster_0"] * embeddings.shape[0]
        return self._run_clustering(embeddings, embeddings.shape[0])

    def _run_clustering(self, embeddings: np.ndarray, n: int) -> List[str]:
        if self.algorithm == "dbscan":
            return self._dbscan(embeddings)
        return self._kmeans(embeddings, n)

    def _kmeans(self, embeddings: np.ndarray, n: int) -> List[str]:
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import normalize
        except ImportError:
            raise RuntimeError("scikit-learn not installed: pip install scikit-learn")

        emb = normalize(embeddings)  # Already normalized but ensure it

        if self.n_clusters_auto:
            k = _elbow_k(emb, min_k=2, max_k=min(12, n // 2 + 1))
        else:
            k = max(2, min(8, n // 3))

        k = min(k, n)  # Can't have more clusters than samples

        log.info(f"KMeans clustering: k={k}, n={n}")
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(emb)
        return [f"cluster_{int(l)}" for l in labels]

    def _dbscan(self, embeddings: np.ndarray) -> List[str]:
        try:
            from sklearn.cluster import DBSCAN
        except ImportError:
            raise RuntimeError("scikit-learn not installed")

        db = DBSCAN(
            eps=self.eps,
            min_samples=self.min_cluster_size,
            metric="cosine",
        )
        labels = db.fit_predict(embeddings)
        # -1 = noise → map to "misc"
        return [f"cluster_{int(l)}" if l >= 0 else "misc" for l in labels]


# ─── FAISS Similarity Search ──────────────────────────────────────────────────

class FAISSIndex:
    """
    Build a FAISS flat-L2 index from embeddings for nearest-neighbour lookup.
    Used for 'relation' mode to find the most similar existing cluster.
    """

    def __init__(self, embeddings: np.ndarray, labels: List[str]):
        try:
            import faiss
        except ImportError:
            raise RuntimeError("faiss-cpu not installed: pip install faiss-cpu")

        import faiss

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)  # Inner product (cosine on normalized)
        self._index.add(embeddings.astype(np.float32))
        self._labels = labels

    def search(self, query: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
        """Return top-k (label, score) for a query embedding."""
        q = query.reshape(1, -1).astype(np.float32)
        scores, indices = self._index.search(q, k)
        return [
            (self._labels[int(i)], float(s))
            for s, i in zip(scores[0], indices[0])
            if i >= 0
        ]


# ─── Elbow method helper ──────────────────────────────────────────────────────

def _elbow_k(embeddings: np.ndarray, min_k: int = 2, max_k: int = 12) -> int:
    """
    Determine optimal k for KMeans via the elbow method (inertia derivative).
    Falls back to a simple heuristic if sklearn is unavailable.
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        return max(2, embeddings.shape[0] // 5)

    n = embeddings.shape[0]
    max_k = min(max_k, n)
    if max_k <= min_k:
        return min_k

    inertias = []
    ks = list(range(min_k, max_k + 1))

    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        km.fit(embeddings)
        inertias.append(km.inertia_)

    # Find the elbow via second derivative
    if len(inertias) < 3:
        return ks[0]

    diffs = np.diff(inertias)
    second_diffs = np.diff(diffs)
    elbow_idx = int(np.argmax(second_diffs)) + 1  # +1 offset from two diffs
    optimal_k = ks[min(elbow_idx, len(ks) - 1)]

    log.debug(f"Elbow method selected k={optimal_k} from range {min_k}–{max_k}")
    return optimal_k
