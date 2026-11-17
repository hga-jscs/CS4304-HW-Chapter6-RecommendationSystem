"""Audit full target memberships and sample the original official calculators."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from extension.evaluate import check_official_code, OFFICIAL_COMMIT, TASKS
from extension.target_cache import create_target_cache


def verify(official_directory: Path, data_directory: Path, output: Path,
           samples: int = 2048) -> dict:
    check_official_code(official_directory)
    sys.path.insert(0, str(official_directory.resolve()))
    from training_pipeline.target_calculators import ChurnTargetCalculator, PropensityTargetCalculator
    from training_pipeline.tasks import parse_task

    relevant = np.load(data_directory / "input" / "relevant_clients.npy", allow_pickle=False)
    active = np.load(data_directory / "target" / "active_clients.npy", allow_pickle=False)
    rng = np.random.default_rng(42)
    records = []
    for window in ("train", "validation"):
        frame = pd.read_parquet(data_directory / "target" / f"{window}_target.parquet")
        for task in TASKS:
            ids = relevant[np.isin(relevant, active)] if task == "churn" else relevant
            classes = None if task == "churn" else np.load(
                data_directory / "target" / f"{task}.npy", allow_pickle=True)
            column = None if task == "churn" else "category" if task == "propensity_category" else "sku"
            calculator = ChurnTargetCalculator() if column is None else PropensityTargetCalculator(
                task=parse_task(task), propensity_targets=classes)
            start = time.perf_counter()
            cached = create_target_cache(ids, frame, column, classes)
            build_seconds = time.perf_counter() - start
            # An independent join/set construction checks every positive membership,
            # including omitted clients/classes. The complement proves all zero cells.
            if column is None:
                _, expected, _ = np.intersect1d(ids, frame["client_id"].unique(), return_indices=True)
                actual = np.flatnonzero(cached[:, 0] == 0)
                if not np.array_equal(np.sort(expected), actual):
                    raise AssertionError(f"Full churn membership mismatch: {window}")
                positive_keys = np.flatnonzero(cached[:, 0] == 1)
            else:
                client_order = pd.DataFrame({"client_id": ids, "row": np.arange(len(ids))})
                class_order = pd.DataFrame({column: classes, "column": np.arange(len(classes))})
                pairs = frame[["client_id", column]].drop_duplicates().merge(
                    client_order, on="client_id", how="inner", validate="many_to_one").merge(
                    class_order, on=column, how="inner", validate="many_to_one")
                expected = np.sort((pairs["row"].to_numpy() * len(classes) + pairs["column"].to_numpy()))
                positive_keys = np.flatnonzero(cached)
                if not np.array_equal(expected, positive_keys):
                    raise AssertionError(f"Full positive-set mismatch: {window}/{task}")
            sums = cached.sum(axis=1)
            indices = {0, len(ids) - 1}
            for mask in (sums == 0, sums > 0, sums > 1):
                group = np.flatnonzero(mask)
                if len(group):
                    indices.update(rng.choice(group, min(256, len(group)), replace=False).tolist())
            if classes is not None and len(positive_keys):
                _, first = np.unique(positive_keys % len(classes), return_index=True)
                indices.update((positive_keys[first] // len(classes)).tolist())
            repeated = frame["client_id"].value_counts()
            repeated_rows = pd.Index(ids).get_indexer(repeated[repeated > 1].index[:128])
            indices.update(repeated_rows[repeated_rows >= 0].tolist())
            indices.update(rng.choice(len(ids), min(samples, len(ids)), replace=False).tolist())
            indices = sorted(indices)
            original_start = time.perf_counter()
            original = [calculator.compute_target(int(ids[index]), frame) for index in indices]
            original_seconds = time.perf_counter() - original_start
            cached_start = time.perf_counter()
            loaded = [cached[index].astype(np.float32) for index in indices]
            cached_seconds = time.perf_counter() - cached_start
            for left, right in zip(original, loaded):
                if left.dtype != right.dtype or not np.array_equal(left, right):
                    raise AssertionError(f"Original calculator mismatch: {window}/{task}")
            # Explicit boundary frame: no event, repeated event, outside client/class,
            # all classes and reversed client/class orders.
            boundary_ids = np.array([7, 2, 9], dtype=np.int64)
            if column is None:
                boundary = pd.DataFrame({"client_id": [7, 7, 99]})
                boundary_cached = create_target_cache(boundary_ids, boundary)
                boundary_calculator = ChurnTargetCalculator()
            else:
                reversed_classes = classes[::-1]
                sentinel = int(np.max(classes)) + 1
                boundary = pd.DataFrame({"client_id": [7] * len(classes) + [7, 2, 99],
                                         column: list(classes) + [classes[0], sentinel, classes[0]]})
                boundary_cached = create_target_cache(boundary_ids, boundary, column, reversed_classes)
                boundary_calculator = PropensityTargetCalculator(task=parse_task(task), propensity_targets=reversed_classes)
            for index, client in enumerate(boundary_ids):
                np.testing.assert_array_equal(boundary_cached[index].astype(np.float32),
                                              boundary_calculator.compute_target(int(client), boundary))
            record = {"window": window, "task": task, "clients": len(ids),
                      "classes": cached.shape[1], "target_rows": len(frame),
                      "full_membership_cells_verified": int(cached.size),
                      "positive_targets": int(cached.sum()), "original_function_sampled_clients": len(indices),
                      "synthetic_boundary_clients": len(boundary_ids), "mismatches": 0,
                      "class_order_sha256": None if classes is None else hashlib.sha256(
                          json.dumps(classes.tolist(), separators=(",", ":")).encode()).hexdigest(),
                      "build_seconds": round(build_seconds, 6),
                      "original_sample_seconds": round(original_seconds, 6),
                      "cached_sample_seconds": round(cached_seconds, 6),
                      "sample_lookup_speedup": round(original_seconds / max(cached_seconds, 1e-9), 2)}
            records.append(record)
            print(json.dumps(record), flush=True)
    result = {"official_commit": OFFICIAL_COMMIT, "verified": True,
              "method": "Full independent membership sets plus sampled original function calls and explicit edge cases",
              "all_original_functions_exhaustively_called": False, "checks": records}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-directory", type=Path, required=True)
    parser.add_argument("--data-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2048)
    verify(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
