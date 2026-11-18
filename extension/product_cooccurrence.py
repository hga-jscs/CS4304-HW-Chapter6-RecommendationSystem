"""Shared product cooccurrence factors with separate purchase and cart pooling.

This input-only spectral ablation is inspired by the purchase/cart relations in
https://arxiv.org/html/2508.06970v1; it does not implement or reproduce TwHIN.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as parquet
from scipy import linalg, sparse
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from .build_embeddings import DAY_NS, input_cutoff, timestamps_ns


RELATIONS = ("product_buy", "add_to_cart")


def events(path, clients, cutoff, batch_size):
    for batch in parquet.ParquetFile(path).iter_batches(batch_size=batch_size,
                                                       columns=["client_id", "sku", "timestamp"]):
        frame = batch.to_pandas()
        rows = clients.get_indexer(frame.client_id)
        times = timestamps_ns(frame.timestamp)
        selected = (rows >= 0) & (times <= cutoff) & frame.sku.notna().to_numpy()
        yield rows[selected], frame.sku.to_numpy()[selected], times[selected]


def relation_matrices(path, clients, products, cutoff, half_life_days, batch_size, output):
    """Spool edge triples, then merge duplicate user-product edges into CSR."""
    paths = [output / f"{path.stem}_{name}.partial" for name in ("rows", "columns", "weights")]
    retained = 0
    with paths[0].open("wb") as row_stream, paths[1].open("wb") as column_stream, paths[2].open("wb") as weight_stream:
        for rows, skus, times in events(path, clients, cutoff, batch_size):
            columns = products.get_indexer(skus)
            selected = columns >= 0
            ages = (cutoff - times[selected]) / DAY_NS
            rows[selected].astype(np.int32).tofile(row_stream)
            columns[selected].astype(np.int32).tofile(column_stream)
            np.exp2(-ages / half_life_days).astype(np.float32).tofile(weight_stream)
            retained += int(selected.sum())
    rows, columns = [np.fromfile(path, dtype=np.int32) for path in paths[:2]]
    weights = np.fromfile(paths[2], dtype=np.float32)
    shape = (len(clients), len(products))
    counts = sparse.coo_matrix((np.ones(len(rows), np.float32), (rows, columns)), shape=shape).tocsr()
    decayed = sparse.coo_matrix((weights, (rows, columns)), shape=shape).tocsr()
    counts.sort_indices()
    decayed.sort_indices()
    del rows, columns, weights
    for path in paths:
        path.unlink()
    return counts, decayed, retained


def weighted_history(counts):
    """Sublinear term frequency and distinct-user IDF, followed by row L2."""
    counts.sum_duplicates()
    counts.eliminate_zeros()
    active_users = int((np.diff(counts.indptr) > 0).sum())
    document_frequency = np.bincount(counts.indices, minlength=counts.shape[1])
    idf = (1 + np.log((1 + active_users) / (1 + document_frequency))).astype(np.float32)
    matrix = counts.astype(np.float32, copy=True)
    np.log1p(matrix.data, out=matrix.data)
    matrix.data *= idf[matrix.indices]
    if matrix.shape[1]:
        normalize(matrix, norm="l2", axis=1, copy=False)
    return matrix, idf, document_frequency


def product_factors(matrix, dimensions=64, oversamples=16, power_iterations=3, seed=42):
    """Approximate right singular factors without forming A.T @ A or a user U.

    A @ Q is still a temporary users-by-subspace float32 matrix (320 MB for
    one million users and 80 columns). The persistent basis lives on products.
    """
    products = matrix.shape[1]
    factors = np.zeros((products, dimensions), np.float32)
    singular_values = np.zeros(dimensions, np.float32)
    if matrix.nnz == 0 or products == 0:
        return factors, singular_values, 0
    width = min(products, dimensions + oversamples)
    generator = np.random.default_rng(seed)
    basis = generator.standard_normal((products, width), dtype=np.float32)

    def gram_times(values):
        users = matrix @ values
        result = matrix.T @ users
        del users
        return np.asarray(result, dtype=np.float32)

    for iteration in range(power_iterations + 1):
        projected = gram_times(basis)
        basis, _ = linalg.qr(projected, mode="economic", overwrite_a=True, check_finite=False)
        print(f"Product subspace iteration {iteration + 1}/{power_iterations + 1}", flush=True)
    projected = gram_times(basis)
    compressed = (basis.T @ projected).astype(np.float64)
    del projected
    eigenvalues, eigenvectors = linalg.eigh((compressed + compressed.T) / 2, check_finite=False)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    threshold = max(float(eigenvalues[0]) * 1e-7, np.finfo(np.float32).tiny)
    rank = min(dimensions, int((eigenvalues > threshold).sum()))
    factors[:, :rank] = basis @ eigenvectors[:, :rank].astype(np.float32)
    singular_values[:rank] = np.sqrt(np.maximum(eigenvalues[:rank], 0)).astype(np.float32)
    if rank:
        anchors = np.argmax(np.abs(factors[:, :rank]), axis=0)
        signs = np.sign(factors[anchors, np.arange(rank)])
        factors[:, :rank] *= signs
    return factors, singular_values, rank


def pooled_blocks(relations, factors, idf, destination, batch_size):
    dimensions = factors.shape[1]
    clients = relations[0].shape[0]
    output = np.lib.format.open_memmap(destination, mode="w+", dtype=np.float16,
                                       shape=(clients, dimensions * len(relations)))
    coverage = {}
    for number, (name, matrix) in enumerate(zip(RELATIONS, relations)):
        np.log1p(matrix.data, out=matrix.data)
        matrix.data *= idf[matrix.indices]
        history_users = 0
        zero_users = 0
        for start in range(0, clients, batch_size):
            rows = matrix[start:start + batch_size]
            values = np.asarray(rows @ factors, dtype=np.float32)
            norms = np.linalg.norm(values.astype(np.float64), axis=1)
            present = norms > 0
            values[present] /= norms[present, None]
            if not np.isfinite(values).all():
                raise ValueError("Product projection produced non-finite values")
            supported = np.diff(rows.indptr) > 0
            assert np.all(values[~supported] == 0)
            history_users += int(supported.sum())
            zero_users += int((~present).sum())
            output[start:start + len(values), number * dimensions:(number + 1) * dimensions] = values
        coverage[name] = {"users_with_retained_edges": history_users, "zero_projection_users": zero_users}
    output.flush()
    del output
    return coverage


def build_product_cooccurrence(data_directory: str | Path, output_directory: str | Path,
                              dimensions=64, minimum_events=3, maximum_products=0,
                              half_life_days=30.0, batch_size=100_000, oversamples=16,
                              power_iterations=3, seed=42, threads=4, cutoff_utc=None,
                              overwrite=False):
    if not 1 <= dimensions <= 64 or min(minimum_events, batch_size, threads) < 1:
        raise ValueError("Use 1-64 dimensions per relation and positive counts/threads")
    if maximum_products < 0 or oversamples < 0 or power_iterations < 0:
        raise ValueError("Product cap, oversampling and power iterations must be nonnegative")
    if not np.isfinite(half_life_days) or half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    started = time.perf_counter()
    data, output = Path(data_directory), Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    reserved = ("embeddings.npy", "client_ids.npy", "features.json", "manifest.json", "product_factors.npz")
    if not overwrite and any((output / name).exists() for name in reserved):
        raise FileExistsError("Product representation exists; use --overwrite")
    (output / "manifest.json").unlink(missing_ok=True)
    ids = np.load(data / "input" / "relevant_clients.npy", allow_pickle=False)
    clients = pd.Index(ids)
    if ids.ndim != 1 or ids.dtype != np.int64 or len(ids) == 0 or clients.has_duplicates:
        raise ValueError("Input IDs must be unique, nonempty int64 values")
    paths = [data / "input" / f"{name}.parquet" for name in RELATIONS]
    if cutoff_utc is None:
        cutoff = input_cutoff(paths, batch_size)
    else:
        stamp = pd.Timestamp(cutoff_utc)
        cutoff = (stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")).value
    frequency, event_counts = Counter(), {}
    for name, path in zip(RELATIONS, paths):
        count = 0
        for _, skus, _ in events(path, clients, cutoff, batch_size):
            frequency.update(pd.Series(skus).value_counts().to_dict())
            count += len(skus)
        event_counts[name] = {"relevant_events_before_cutoff": count}
    qualified = [(int(item), count) for item, count in frequency.items() if count >= minimum_events]
    qualified_count = len(qualified)
    if maximum_products and len(qualified) > maximum_products:
        qualified = sorted(qualified, key=lambda item: (-item[1], item[0]))[:maximum_products]
    qualified.sort()
    products = pd.Index(np.asarray([item for item, _ in qualified], dtype=np.int64))
    retained_frequency = np.asarray([count for _, count in qualified], dtype=np.int64)
    del qualified, frequency
    print(f"Selected {len(products)} products from {qualified_count} meeting frequency >= {minimum_events}", flush=True)
    relations, combined = [], None
    for name, path in zip(RELATIONS, paths):
        counts, decayed, retained = relation_matrices(path, clients, products, cutoff,
                                                     half_life_days, batch_size, output)
        event_counts[name]["retained_events"] = retained
        event_counts[name]["unique_user_product_edges"] = counts.nnz
        combined = counts if combined is None else combined + counts
        relations.append(decayed)
        print(f"Built sparse relation {name}: {counts.nnz} unique user-product edges", flush=True)
    del counts
    with threadpool_limits(limits=threads):
        history, idf, document_frequency = weighted_history(combined)
        del combined
        factors, singular_values, rank = product_factors(history, dimensions, oversamples, power_iterations, seed)
        sparse_bytes = history.data.nbytes + history.indices.nbytes + history.indptr.nbytes
        del history
        coverage = pooled_blocks(relations, factors, idf, output / "embeddings.npy", min(batch_size, 20_000))
    state_path = output / "product_factors.npz"
    np.savez_compressed(state_path, product_ids=products.to_numpy(dtype=np.int64),
                        event_frequency=retained_frequency, document_frequency=document_frequency,
                        item_factors=factors, idf=idf, singular_values=singular_values)
    np.save(output / "client_ids.npy", ids, allow_pickle=False)
    names = [f"{name}_cooccurrence_{column}" for name in RELATIONS for column in range(dimensions)]
    (output / "features.json").write_text(json.dumps(names, indent=2) + "\n", encoding="utf-8")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Shared product spectral factors with separate purchase/cart decay pooling; not TwHIN",
        "clients": len(ids), "dimensions": dimensions * 2, "dtype": "float16", "standardized": False,
        "blocks": [{"name": f"{name}_cooccurrence", "start": number * dimensions,
                    "stop": (number + 1) * dimensions, "kind": "embedding"}
                   for number, name in enumerate(RELATIONS)],
        "normalization": "Each relation projection is row-L2 normalized; no column centering or whitening; empty histories stay zero",
        "recommended_block_weight": 2.0,
        "fit_matrix": "log1p(purchase_count + cart_count) * smooth distinct-user IDF, then user row-L2",
        "pooling": "log1p(sum of exp2(-age_days / half_life_days)) * fitted IDF, then multiply shared item factors and row-L2",
        "vocabulary_scope": "Only relevant_clients purchase/cart input events at or before cutoff; no downstream class list",
        "minimum_events": minimum_events, "maximum_products": maximum_products,
        "qualified_products": qualified_count, "retained_products": len(products), "effective_rank": rank,
        "half_life_days": half_life_days, "oversamples": oversamples, "power_iterations": power_iterations,
        "seed": seed, "threads": threads, "batch_size": batch_size,
        "input_cutoff_utc": pd.Timestamp(cutoff, tz="UTC").isoformat(),
        "event_counts": event_counts, "coverage": coverage, "fit_csr_bytes": sparse_bytes,
        "subspace_workspace_note": "A@Q uses users * min(products, dimensions+oversamples) float32 values; no dense item-item matrix or saved user U",
        "state_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
        "target_data_used_for_embeddings": False, "official_target_directory_read": False,
        "inspiration": "https://arxiv.org/html/2508.06970v1#S3.T4",
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "pyarrow", "scipy", "scikit-learn")},
        "seconds": time.perf_counter() - started}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--dimensions", type=int, default=64)
    parser.add_argument("--minimum-events", type=int, default=3)
    parser.add_argument("--maximum-products", type=int, default=0, help="0 keeps every frequency-qualified product")
    parser.add_argument("--half-life-days", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--oversamples", type=int, default=16)
    parser.add_argument("--power-iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cutoff-utc")
    parser.add_argument("--overwrite", action="store_true")
    print(json.dumps(build_product_cooccurrence(**vars(parser.parse_args())), indent=2), flush=True)


if __name__ == "__main__":
    main()
