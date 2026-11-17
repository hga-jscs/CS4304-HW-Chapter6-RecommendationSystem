"""Smoothed label statistics with out-of-fold training features.

Validation and prediction use only the labels passed to ``fit_transform``.
Row-order training features use preceding blocks, not future label averages.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def _index(values: pd.Series) -> pd.Index:
    # A consistent nullable string representation also makes missing categories
    # match between Arrow-backed CSV input and small in-memory prediction batches.
    return pd.Index(pd.array(values, dtype="string[pyarrow]"))


def _factorize(values: pd.Series) -> tuple[np.ndarray, pd.Index]:
    codes, levels = pd.factorize(_index(values), sort=False, use_na_sentinel=False)
    return codes.astype(np.int32), levels


@dataclass
class _Statistics:
    levels: tuple[pd.Index, ...]
    pairs: pd.Index | None
    counts: np.ndarray
    sums: np.ndarray

    def lookup(self, columns: tuple[pd.Series, ...]) -> np.ndarray:
        first = self.levels[0].get_indexer(_index(columns[0]))
        if self.pairs is None:
            return first
        second = self.levels[1].get_indexer(_index(columns[1]))
        present = (first >= 0) & (second >= 0)
        codes = np.full(len(first), -1, dtype=np.int64)
        packed = (first[present].astype(np.uint64) * len(self.levels[1])
                  + second[present].astype(np.uint64))
        codes[present] = self.pairs.get_indexer(packed)
        return codes


class TargetEncoder:
    """Five label-derived features; fitted state stores no OOF row arrays."""

    feature_names = ["target_msno", "target_song_id", "target_artist_name",
                     "target_user_source_system_tab", "target_user_source_type"]

    def __init__(self, smoothing: float = 20, folds: int = 5, seed: int = 42):
        if not np.isfinite(smoothing) or smoothing <= 0 or folds < 2:
            raise ValueError("smoothing must be positive and folds at least two")
        self.smoothing = smoothing
        self.folds = folds
        self.seed = seed

    @staticmethod
    def _fields(frame: pd.DataFrame, artists: pd.Series):
        yield "target_msno", (frame["msno"],)
        yield "target_song_id", (frame["song_id"],)
        yield "target_artist_name", (artists,)
        yield "target_user_source_system_tab", (frame["msno"], frame["source_system_tab"])
        yield "target_user_source_type", (frame["msno"], frame["source_type"])

    def _blocks(self, frame: pd.DataFrame, split_mode: str) -> list[np.ndarray]:
        count = len(frame)
        if split_mode == "row_order":
            column = "global_row_id" if "global_row_id" in frame else "row_id"
            order = np.argsort(frame[column].to_numpy(), kind="stable")
            return list(np.array_split(order, min(self.folds, count)))
        if count == 1:
            return [np.array([0], dtype=np.int64)]
        folds = KFold(n_splits=min(self.folds, count), shuffle=True, random_state=self.seed)
        return [held_out for _, held_out in folds.split(np.empty(count, dtype=np.int8))]

    @staticmethod
    def _keys(columns: tuple[pd.Series, ...]) -> tuple[np.ndarray, tuple[pd.Index, ...], pd.Index | None]:
        first, first_levels = _factorize(columns[0])
        if len(columns) == 1:
            return first, (first_levels,), None
        second, second_levels = _factorize(columns[1])
        # Numeric keys avoid materializing millions of Python string tuples.
        packed = first.astype(np.uint64) * len(second_levels) + second.astype(np.uint64)
        codes, pairs = pd.factorize(packed, sort=False)
        return codes.astype(np.int32), (first_levels, second_levels), pd.Index(pairs)

    def _values(self, codes: np.ndarray, counts: np.ndarray, sums: np.ndarray,
                prior: float) -> np.ndarray:
        values = np.full(len(codes), prior, dtype=np.float32)
        present = codes >= 0
        keys = codes[present]
        values[present] = (sums[keys] + self.smoothing * prior) / (counts[keys] + self.smoothing)
        return values

    def fit_transform(self, frame: pd.DataFrame, artists: pd.Series,
                      split_mode: str = "random") -> dict[str, np.ndarray]:
        if not len(frame) or len(artists) != len(frame):
            raise ValueError("Training rows must be nonempty and align with artist rows")
        if split_mode not in {"random", "row_order"}:
            raise ValueError("split_mode must be random or row_order")
        labels = frame["target"].to_numpy(dtype=np.float64)
        blocks = self._blocks(frame, split_mode)
        label_sum = float(labels.sum())
        self.statistics = {}
        output = {}
        for name, columns in self._fields(frame, artists):
            codes, levels, pairs = self._keys(columns)
            groups = len(levels[0]) if pairs is None else len(pairs)
            counts = np.bincount(codes, minlength=groups)
            sums = np.bincount(codes, weights=labels, minlength=groups)
            values = np.empty(len(frame), dtype=np.float32)
            if split_mode == "random":
                for held_out in blocks:
                    remaining = len(frame) - len(held_out)
                    prior = (label_sum - labels[held_out].sum()) / remaining if remaining else 0.5
                    fold_counts = counts - np.bincount(codes[held_out], minlength=groups)
                    fold_sums = sums - np.bincount(codes[held_out], weights=labels[held_out], minlength=groups)
                    values[held_out] = self._values(codes[held_out], fold_counts, fold_sums, float(prior))
            else:
                preceding_counts = np.zeros(groups, dtype=np.int64)
                preceding_sums = np.zeros(groups, dtype=np.float64)
                preceding_rows, preceding_label_sum = 0, 0.0
                for block in blocks:
                    prior = preceding_label_sum / preceding_rows if preceding_rows else 0.5
                    values[block] = self._values(codes[block], preceding_counts, preceding_sums, prior)
                    preceding_counts += np.bincount(codes[block], minlength=groups)
                    preceding_sums += np.bincount(codes[block], weights=labels[block], minlength=groups)
                    preceding_rows += len(block)
                    preceding_label_sum += float(labels[block].sum())
            self.statistics[name] = _Statistics(levels, pairs, counts, sums)
            output[name] = values
        self.prior = label_sum / len(frame)
        self.metadata = {"strategy": "preceding_row_blocks" if split_mode == "row_order" else "random_kfold",
                         "fit_rows": len(frame), "folds": len(blocks), "prior": self.prior,
                         "split_mode": split_mode, "smoothing": self.smoothing, "seed": self.seed}
        if split_mode == "row_order":
            self.metadata["first_block_prior"] = 0.5
        return output

    def transform(self, frame: pd.DataFrame, artists: pd.Series) -> dict[str, np.ndarray]:
        if len(artists) != len(frame):
            raise ValueError("Prediction rows must align with artist rows")
        output = {}
        for name, columns in self._fields(frame, artists):
            statistics = self.statistics[name]
            codes = statistics.lookup(columns)
            output[name] = self._values(codes, statistics.counts, statistics.sums, self.prior)
        return output
