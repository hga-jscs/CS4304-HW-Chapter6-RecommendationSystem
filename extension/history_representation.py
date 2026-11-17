"""Compact historical statistics, PQ co-occurrence factors and sparse user factors.

Only input/ and product properties are read. PQ code positions are categorical;
their co-occurrence factors do not reconstruct the unavailable original codebook.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as parquet
from scipy import sparse
from sklearn.utils.extmath import randomized_svd

from .build_embeddings import (EVENTS, PRODUCT_EVENTS, STATISTICS, CROSS_STATISTICS,
    DAY_NS, parse_query, timestamps_ns, input_cutoff, validate_inputs, hash_buckets,
    add_cross_statistics)
from .standardize import write_embeddings


def batches(path, columns, size, cutoff):
    for batch in parquet.ParquetFile(path).iter_batches(batch_size=size, columns=columns):
        frame = batch.to_pandas()
        if "timestamp" in frame:
            frame = frame.loc[timestamps_ns(frame.timestamp) <= cutoff]
        if len(frame):
            yield frame


def code_matrix(values):
    codes = np.asarray([parse_query(x) if isinstance(x, str) else tuple(x) for x in values], dtype=np.int32)
    if codes.ndim != 2 or codes.shape[1] > 16 or np.any((codes < 0) | (codes > 255)):
        raise ValueError("Expected at most 16 categorical PQ bytes per record")
    return codes + np.arange(codes.shape[1], dtype=np.int32)[None, :] * 256


def add_cooccurrence(counts, keys, weights=None):
    n, width = keys.shape
    values = np.repeat(np.sqrt(np.asarray(weights, dtype=np.float32)), width) if weights is not None else np.ones(n * width, np.float32)
    documents = sparse.csr_matrix((values, (np.repeat(np.arange(n), width), keys.ravel())), shape=(n, 4096))
    counts += (documents.T @ documents).toarray()


def token_factors(counts, dimensions=32, seed=42):
    """Positive PMI of position/code co-occurrences, then U sqrt(S)."""
    np.fill_diagonal(counts, 0)
    marginals = counts.sum(axis=1, dtype=np.float64)
    total = marginals.sum()
    if total == 0:
        return np.zeros((4096, dimensions), np.float32)
    for start in range(0, len(counts), 256):
        block = counts[start:start + 256]
        denominator = marginals[start:start + 256, None] * marginals[None, :]
        ratio = np.divide(block * total, denominator, out=np.zeros_like(block), where=denominator > 0)
        block[:] = np.maximum(np.log(np.maximum(ratio, 1e-20)), 0)
    active = np.flatnonzero(marginals > 0)
    rank = min(dimensions, len(active) - 1)
    left, singular, _ = randomized_svd(counts[np.ix_(active, active)], n_components=rank, n_iter=3, random_state=seed)
    result = np.zeros((4096, dimensions), np.float32)
    result[active, :rank] = left * np.sqrt(singular)
    return result


def rank_counts(counter, limit):
    return [int(key) for key, _ in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def feature_names(top_skus, top_categories, long_tail=False):
    names = [f"{event}.{stat}" for event in EVENTS for stat in STATISTICS]
    names += [f"{event}.has_history" for event in EVENTS] + list(CROSS_STATISTICS)
    price = ["price_bucket_mean", "price_bucket_max", "price_bucket_std", "price_bucket_sum_30_days", "price_observations"]
    price += [f"price_decile_{i}" for i in range(10)]
    for event in PRODUCT_EVENTS:
        names += [f"{event}.{name}" for name in price]
    # Exactly 96 statistics and 256 popular-item columns, even on small fixtures.
    names += [f"popular_sku_{top_skus[i]}" if i < len(top_skus) else f"unused_sku_{i}" for i in range(128)]
    names += [f"popular_category_{top_categories[i]}" if i < len(top_categories) else f"unused_category_{i}" for i in range(128)]
    names += [f"name_cooccurrence_{i}" for i in range(32)] + [f"query_cooccurrence_{i}" for i in range(32)]
    names += [f"collaborative_{i}" for i in range(128)]
    if long_tail:
        names += [f"long_tail_{column}_{i}" for column in ("sku", "category") for i in range(64)]
    return names


def build_history(data_directory, output_directory, batch_size=100_000, graph_skus=4096,
                  graph_urls=2048, long_tail=False, overwrite=False, seed=42):
    started = time.perf_counter()
    data, output = Path(data_directory), Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "embeddings.npy").exists() and not overwrite:
        raise FileExistsError("Output exists; use --overwrite")
    ids, paths = validate_inputs(data)
    cutoff = input_cutoff(paths, batch_size)
    clients = pd.Index(ids)
    properties = pd.read_parquet(data / "product_properties.parquet", dtype_backend="pyarrow").set_index("sku")
    sku_counts, category_counts, url_counts = Counter(), Counter(), Counter()
    for event, path in zip(EVENTS, paths):
        if event not in ("product_buy", "add_to_cart", "page_visit"):
            continue
        column = "url" if event == "page_visit" else "sku"
        for frame in batches(path, [column, "timestamp"], batch_size, cutoff):
            if column == "url":
                url_counts.update(frame.url.value_counts().to_dict())
            else:
                sku_counts.update(frame.sku.value_counts().to_dict())
                category_counts.update(frame.sku.map(properties.category).value_counts().to_dict())
        print(f"Counted historical vocabulary: {event}", flush=True)
    top_skus, top_categories = rank_counts(sku_counts, 128), rank_counts(category_counts, 128)
    graph_values = {"category": sorted(int(x) for x in properties.category.dropna().unique()),
                    "sku": rank_counts(sku_counts, graph_skus), "url": rank_counts(url_counts, graph_urls)}
    del url_counts
    graph_indices, graph_width = {}, 0
    for key, values in graph_values.items():
        graph_indices[key] = (pd.Index(values), graph_width)
        graph_width += len(values)
    names = feature_names(top_skus, top_categories, long_tail)
    scratch = output / "features.partial.npy"
    matrix = np.lib.format.open_memmap(scratch, mode="w+", dtype=np.float32, shape=(len(ids), len(names)), fortran_order=True)
    matrix[:] = 0
    cooccurrence = np.zeros((4096, 4096), np.float32)
    used_skus = [sku for sku in sku_counts if sku in properties.index]
    for start in range(0, len(used_skus), batch_size):
        selected = used_skus[start:start + batch_size]
        keys = code_matrix(properties.loc[selected, "name"].tolist())
        add_cooccurrence(cooccurrence, keys, [sku_counts[sku] for sku in selected])
    name_factors = token_factors(cooccurrence, seed=seed)
    del sku_counts, category_counts, used_skus
    cooccurrence[:] = 0
    pool_weights = np.zeros((len(ids), 2), np.float32)
    graph_paths = [output / f"graph_{name}.partial" for name in ("rows", "columns", "weights")]
    graph_streams = [path.open("wb") for path in graph_paths]

    def graph_add(rows, values, column, weights):
        index, offset = graph_indices[column]
        codes = index.get_indexer(values)
        valid = codes >= 0
        rows[valid].astype(np.int32).tofile(graph_streams[0])
        (codes[valid] + offset).astype(np.int32).tofile(graph_streams[1])
        weights[valid].astype(np.float32).tofile(graph_streams[2])

    try:
        for event_number, (event, path) in enumerate(zip(EVENTS, paths)):
            first = np.full(len(ids), np.iinfo(np.int64).max, np.int64)
            last = np.full(len(ids), np.iinfo(np.int64).min, np.int64)
            payload = "sku" if event in PRODUCT_EVENTS else "url" if event == "page_visit" else "query"
            for frame in batches(path, ["client_id", "timestamp", payload], batch_size, cutoff):
                rows = clients.get_indexer(frame.client_id)
                chosen = rows >= 0
                frame, rows = frame.loc[chosen], rows[chosen]
                if not len(frame):
                    continue
                times = timestamps_ns(frame.timestamp)
                age = (cutoff - times) / DAY_NS
                weights = np.exp2(-age / 30).astype(np.float32)
                base = event_number * 8
                for offset, values in enumerate((np.ones(len(rows)), age <= 1, age <= 7, age <= 30,
                                                np.exp2(-age / 7), weights)):
                    np.add.at(matrix[:, base + offset], rows, values)
                np.minimum.at(first, rows, times)
                np.maximum.at(last, rows, times)
                if event in PRODUCT_EVENTS:
                    metadata = properties.reindex(frame.sku)
                    price = metadata.price.to_numpy(dtype=np.float64, na_value=np.nan)
                    valid = np.isfinite(price) & (price >= 0)
                    r, p, w = rows[valid], price[valid], weights[valid]
                    offset = 51 + event_number * 15
                    np.add.at(matrix[:, offset], r, p)
                    np.maximum.at(matrix[:, offset + 1], r, p)
                    np.add.at(matrix[:, offset + 2], r, p * p)
                    recent = valid & (age <= 30)
                    np.add.at(matrix[:, offset + 3], rows[recent], price[recent])
                    np.add.at(matrix[:, offset + 4], r, 1)
                    np.add.at(matrix, (r, offset + 5 + np.minimum((p // 10).astype(int), 9)), w)
                    if event in ("product_buy", "add_to_cart"):
                        for column, values, exact_values, offset in (
                            ("sku", frame.sku.to_numpy(), top_skus, 96),
                            ("category", metadata.category.to_numpy(), top_categories, 224)):
                            codes = pd.Index(exact_values).get_indexer(values)
                            valid = codes >= 0
                            np.add.at(matrix, (rows[valid], offset + codes[valid]), weights[valid])
                            if long_tail:
                                tail = ~valid & ~pd.isna(values)
                                np.add.at(matrix, (rows[tail], 544 + (0 if column == "sku" else 64) + hash_buckets(values[tail], 64)), weights[tail])
                            graph_add(rows, values, column, weights)
                    if event == "product_buy":
                        valid = metadata["name"].notna().to_numpy()
                        keys = code_matrix(metadata.loc[valid, "name"].tolist())
                        vectors = name_factors[keys].mean(axis=1)
                        np.add.at(matrix[:, 352:384], rows[valid], vectors * weights[valid, None])
                        np.add.at(pool_weights[:, 0], rows[valid], weights[valid])
                elif event == "page_visit":
                    graph_add(rows, frame.url.to_numpy(), "url", weights)
                else:
                    keys = code_matrix(frame["query"].tolist())
                    add_cooccurrence(cooccurrence, keys, weights)
            present = matrix[:, base] > 0
            matrix[:, base + 6] = 366
            matrix[present, base + 6] = (cutoff - last[present]) / DAY_NS
            matrix[present, base + 7] = (last[present] - first[present]) / DAY_NS
            print(f"Aggregated history: {event}", flush=True)
    finally:
        for stream in graph_streams:
            stream.close()
    add_cross_statistics(matrix, names, min(batch_size, 20_000))
    for event_number in range(3):
        offset = 51 + event_number * 15
        count = np.maximum(matrix[:, offset + 4], 1)
        mean = matrix[:, offset] / count
        matrix[:, offset + 2] = np.sqrt(np.maximum(matrix[:, offset + 2] / count - mean * mean, 0))
        matrix[:, offset] = mean
    query_factors = token_factors(cooccurrence, seed=seed)
    del cooccurrence
    for frame in batches(paths[-1], ["client_id", "timestamp", "query"], batch_size, cutoff):
        rows = clients.get_indexer(frame.client_id)
        valid = rows >= 0
        frame, rows = frame.loc[valid], rows[valid]
        if not len(frame):
            continue
        weights = np.exp2(-(cutoff - timestamps_ns(frame.timestamp)) / DAY_NS / 30).astype(np.float32)
        vectors = query_factors[code_matrix(frame["query"].tolist())].mean(axis=1)
        np.add.at(matrix[:, 384:416], rows, vectors * weights[:, None])
        np.add.at(pool_weights[:, 1], rows, weights)
    for i, start in enumerate((352, 384)):
        matrix[:, start:start + 32] /= np.maximum(pool_weights[:, i, None], 1e-10)
    del properties, pool_weights
    if graph_paths[0].stat().st_size:
        rows = np.memmap(graph_paths[0], mode="r", dtype=np.int32)
        columns = np.memmap(graph_paths[1], mode="r", dtype=np.int32)
        weights = np.memmap(graph_paths[2], mode="r", dtype=np.float32)
        graph = sparse.coo_matrix((weights, (rows, columns)), shape=(len(ids), graph_width)).tocsr()
        del rows, columns, weights
        graph.data = np.log1p(graph.data)
        degree = np.sqrt(np.asarray(graph.power(2).sum(axis=1)).ravel())
        graph = sparse.diags(1 / np.maximum(degree, 1e-10)) @ graph
        rank = min(128, min(graph.shape) - 1)
        if rank > 0:
            left, singular, _ = randomized_svd(graph, n_components=rank, n_iter=3, random_state=seed)
            for start in range(0, len(ids), 20_000):
                values = left[start:start + 20_000] * singular
                values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-10)
                values[degree[start:start + len(values)] == 0] = 0
                matrix[start:start + len(values), 416:416 + rank] = values
            del left
        del graph
    else:
        rank = 0
    for path in graph_paths:
        path.unlink()
    for start in range(0, len(ids), 20_000):
        matrix[start:start + 20_000, :352] = np.log1p(matrix[start:start + 20_000, :352])
        if long_tail:
            matrix[start:start + 20_000, 544:] = np.log1p(matrix[start:start + 20_000, 544:])
    # Keep float32 blocks until the final concatenation/standardization.
    matrix.flush()
    del matrix
    scratch.replace(output / "embeddings.npy")
    np.save(output / "client_ids.npy", ids)
    (output / "features.json").write_text(json.dumps(names, indent=2) + "\n", encoding="utf-8")
    manifest = {"method": "historical_compact_statistics_pq_cooccurrence_and_sparse_factors",
        "clients": len(ids), "dimensions": len(names), "dtype": "float32", "graph_rank": rank,
        "graph_columns": {k: len(v) for k, v in graph_values.items()},
        "popular_skus": top_skus, "popular_categories": top_categories,
        "input_cutoff_utc": pd.Timestamp(cutoff, tz="UTC").isoformat(), "seed": seed,
        "target_data_used_for_embeddings": False, "long_tail": long_tail,
        "pq_interpretation": "Separate name/query position-code co-occurrence factors; not recovered source semantic vectors",
        "seconds": round(time.perf_counter() - started, 3),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def combine(history_directory, pretrained_directory, output_directory, overwrite=False):
    history, pretrained, output = map(Path, (history_directory, pretrained_directory, output_directory))
    output.mkdir(parents=True, exist_ok=True)
    if (output / "embeddings.npy").exists() and not overwrite:
        raise FileExistsError("Output exists; use --overwrite")
    ids = np.load(history / "client_ids.npy")
    if not np.array_equal(ids, np.load(pretrained / "client_ids.npy")):
        raise ValueError("Representation rows have different client order")
    a = np.load(history / "embeddings.npy", mmap_mode="r")
    b = np.load(pretrained / "embeddings.npy", mmap_mode="r")
    dimensions = a.shape[1] + b.shape[1]
    if dimensions > 2048:
        raise ValueError("Combined dimensions exceed 2048")
    temporary = output / "combined.partial.npy"
    matrix = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(len(ids), dimensions))
    for start in range(0, len(ids), 20_000):
        matrix[start:start + 20_000] = np.concatenate((b[start:start + 20_000], a[start:start + 20_000]), axis=1)
    normalization = write_embeddings(matrix, output / "embeddings.npy", standardize=True)
    del matrix
    temporary.unlink()
    np.save(output / "client_ids.npy", ids)
    names = [f"pretrained_{i}" for i in range(b.shape[1])] + json.loads((history / "features.json").read_text())
    (output / "features.json").write_text(json.dumps(names, indent=2) + "\n", encoding="utf-8")
    (output / "standardization.json").write_text(json.dumps(normalization, indent=2) + "\n", encoding="utf-8")
    manifest = {"clients": len(ids), "dimensions": dimensions, "dtype": "float16",
                "target_data_used_for_embeddings": False, "standardized": True,
                "history_manifest": json.loads((history / "manifest.json").read_text()),
                "pretrained_manifest": json.loads((pretrained / "manifest.json").read_text())}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    build = actions.add_parser("build")
    build.add_argument("--data-directory", type=Path, required=True)
    build.add_argument("--output-directory", type=Path, required=True)
    build.add_argument("--batch-size", type=int, default=100_000)
    build.add_argument("--graph-skus", type=int, default=4096)
    build.add_argument("--graph-urls", type=int, default=2048)
    build.add_argument("--long-tail", action="store_true")
    build.add_argument("--seed", type=int, default=42)
    merge = actions.add_parser("combine")
    for name in ("history-directory", "pretrained-directory", "output-directory"):
        merge.add_argument("--" + name, type=Path, required=True)
    for action in (build, merge):
        action.add_argument("--overwrite", action="store_true")
    arguments = vars(parser.parse_args())
    action = arguments.pop("action")
    print(json.dumps((build_history if action == "build" else combine)(**arguments), indent=2))


if __name__ == "__main__":
    main()
