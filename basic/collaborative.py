"""Training-only sparse interaction embeddings for LightGBM features."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd


class InteractionEmbedding:
    """Factorize log-count/IDF/row-normalized interactions without densifying R."""

    def __init__(self, components: int, seed: int = 42):
        self.components = components
        self.seed = seed

    def fit(self, users: pd.Series, items: pd.Series) -> "InteractionEmbedding":
        started = time.perf_counter()
        self.users = pd.Index(users.dropna().unique())
        self.items = pd.Index(items.dropna().unique())
        rows, columns = self.users.get_indexer(users), self.items.get_indexer(items)
        present = (rows >= 0) & (columns >= 0)
        matrix = csr_matrix((np.ones(int(present.sum()), dtype=np.float32),
                             (rows[present], columns[present])),
                            shape=(len(self.users), len(self.items)), dtype=np.float32)
        matrix.sum_duplicates()
        matrix.data = np.log1p(matrix.data)
        support = np.bincount(matrix.indices, minlength=matrix.shape[1])
        inverse_frequency = (np.log((1 + matrix.shape[0]) / (1 + support)) + 1).astype(np.float32)
        matrix.data *= inverse_frequency[matrix.indices]
        rank = max(0, min(self.components, matrix.shape[0] - 1, matrix.shape[1] - 1))
        if rank:
            matrix = normalize(matrix, norm="l2", axis=1, copy=False)
            left, singular_values, right = randomized_svd(
                matrix, n_components=rank, n_iter=3, random_state=self.seed,
            )
            scale = np.sqrt(singular_values).astype(np.float32)
            self.user_factors = (left * scale).astype(np.float32)
            self.item_factors = (right.T * scale).astype(np.float32)
        else:
            self.user_factors = np.zeros((matrix.shape[0], 0), dtype=np.float32)
            self.item_factors = np.zeros((matrix.shape[1], 0), dtype=np.float32)
        self.metadata = {"shape": list(matrix.shape), "nonzero_pairs": int(matrix.nnz),
                         "requested_rank": self.components, "rank": rank,
                         "seed": self.seed, "iterations": 3,
                         "fit_seconds": time.perf_counter() - started}
        return self

    def transform(self, users: pd.Series, items: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        rows, columns = self.users.get_indexer(users), self.items.get_indexer(items)
        present = (rows >= 0) & (columns >= 0)
        dot = np.zeros(len(users), dtype=np.float32)
        cosine = np.zeros(len(users), dtype=np.float32)
        if self.metadata["rank"]:
            # Bound temporary factor gathers even when transforming all training rows.
            for start in range(0, len(users), 200_000):
                selected = np.flatnonzero(present[start:start + 200_000]) + start
                if not len(selected):
                    continue
                left, right = self.user_factors[rows[selected]], self.item_factors[columns[selected]]
                products = np.einsum("ij,ij->i", left, right)
                lengths = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
                dot[selected] = products
                cosine[selected] = np.divide(products, lengths, out=np.zeros_like(products), where=lengths > 0)
        return dot, cosine


class CollaborativeFeatures:
    feature_names = ["song_embedding_dot", "song_embedding_cosine",
                     "artist_embedding_dot", "artist_embedding_cosine"]

    def __init__(self, song_components: int = 32, artist_components: int = 16, seed: int = 42):
        self.song = InteractionEmbedding(song_components, seed)
        self.artist = InteractionEmbedding(artist_components, seed)

    def fit(self, frame: pd.DataFrame, artists: pd.Series) -> "CollaborativeFeatures":
        self.song.fit(frame["msno"], frame["song_id"])
        self.artist.fit(frame["msno"], artists)
        self.metadata = {"weighting": "log1p_counts_then_item_idf_then_row_l2",
                         "song": self.song.metadata, "artist": self.artist.metadata}
        return self

    def transform(self, frame: pd.DataFrame, artists: pd.Series) -> dict[str, np.ndarray]:
        values = (*self.song.transform(frame["msno"], frame["song_id"]),
                  *self.artist.transform(frame["msno"], artists))
        return dict(zip(self.feature_names, values))
