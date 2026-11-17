"""Equivalent target caching used only by the official downstream data loader.

This module never builds user representations. It turns the official calculator's
membership tests into byte matrices, then returns fresh float32 targets in the
unchanged client and class order on every Dataset access.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_target_cache(client_ids: np.ndarray, target_df: pd.DataFrame,
                        column: str | None = None, classes: np.ndarray | None = None) -> np.ndarray:
    clients = pd.Index(client_ids)
    if clients.has_duplicates:
        raise ValueError("Cached data loading requires unique client IDs")
    rows = clients.get_indexer(target_df["client_id"])
    if column is None:
        # Official churn target is 1 iff there is no target-window purchase.
        result = np.ones((len(clients), 1), dtype=np.uint8)
        result[rows[rows >= 0], 0] = 0
    else:
        if classes is None or len(classes) == 0:
            raise ValueError("Propensity classes must be provided in official order")
        class_index = pd.Index(classes)
        if class_index.has_duplicates:
            raise ValueError("Official propensity classes must be unique")
        columns = class_index.get_indexer(target_df[column])
        valid = (rows >= 0) & (columns >= 0)
        result = np.zeros((len(clients), len(classes)), dtype=np.uint8)
        # Repeated purchases remain binary, matching unique() followed by isin().
        result[rows[valid], columns[valid]] = 1
    return result


class CachedBehavioralDataset:
    """Map-style Dataset with the same samples as the official BehavioralDataset."""

    def __init__(self, embeddings, client_ids, target_df, target_calculator):
        from training_pipeline.target_calculators import ChurnTargetCalculator, PropensityTargetCalculator

        self.embeddings = embeddings
        self.client_ids = client_ids
        self.target_df = target_df
        self.target_calculator = target_calculator
        started = time.perf_counter()
        if isinstance(target_calculator, ChurnTargetCalculator):
            self.targets = create_target_cache(client_ids, target_df)
        elif isinstance(target_calculator, PropensityTargetCalculator):
            self.targets = create_target_cache(client_ids, target_df,
                                               target_calculator._propensity_type,
                                               target_calculator._propensity_targets)
        else:
            raise TypeError("Target cache supports only the pinned churn and propensity calculators")
        logger.info("Cached targets: clients=%d classes=%d bytes=%d seconds=%.3f",
                    len(client_ids), self.targets.shape[1], self.targets.nbytes,
                    time.perf_counter() - started)

    def __len__(self):
        return self.client_ids.shape[0]

    def __getitem__(self, index):
        return self.embeddings[index], self.targets[index].astype(np.float32)


def install() -> None:
    """Replace only the Dataset reference used by the unchanged official module."""
    import training_pipeline.data_module as official_data_module

    official_data_module.BehavioralDataset = CachedBehavioralDataset
    logger.warning("Using verified target-cache data loader; model, targets, order and metrics are unchanged")
