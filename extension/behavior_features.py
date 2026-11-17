"""Historical behavior features with a reusable vocabulary and projection basis.

Only input/ and product properties are used. Cart balance is an event-count
proxy, not reconstructed cart contents; independent purchase days are not orders.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.utils.extmath import randomized_svd

from .build_embeddings import (EVENTS, PRODUCT_EVENTS, STATISTICS, CROSS_STATISTICS,
                               DAY_NS, add_cross_statistics, input_cutoff,
                               timestamps_ns, validate_inputs)
from .history_representation import (add_cooccurrence, batches, code_matrix,
                                      rank_counts, token_factors)


STATE_FILE = "feature_state.npz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cutoff(paths: list[Path], cutoff_utc: str | None, batch_size: int) -> int:
    if cutoff_utc is None:
        return input_cutoff(paths, batch_size)
    value = pd.Timestamp(cutoff_utc)
    value = value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
    return value.value


def _fit_vocabulary(paths, properties, cutoff, batch_size, graph_skus, graph_urls, seed):
    purchases, purchase_categories = Counter(), Counter()
    graph_items, graph_categories, urls = Counter(), Counter(), Counter()
    cooccurrence = np.zeros((4096, 4096), np.float32)
    for event, path in zip(EVENTS, paths):
        if event == "remove_from_cart":
            continue
        payload = "sku" if event in PRODUCT_EVENTS else "url" if event == "page_visit" else "query"
        for frame in batches(path, [payload, "timestamp"], batch_size, cutoff):
            if event in ("product_buy", "add_to_cart"):
                counts = frame.sku.value_counts().to_dict()
                categories = frame.sku.map(properties.category).value_counts().to_dict()
                graph_items.update(counts)
                graph_categories.update(categories)
                if event == "product_buy":
                    purchases.update(counts)
                    purchase_categories.update(categories)
            elif event == "page_visit":
                urls.update(frame.url.value_counts().to_dict())
            else:
                weights = np.exp2(-(cutoff - timestamps_ns(frame.timestamp)) / DAY_NS / 30)
                add_cooccurrence(cooccurrence, code_matrix(frame["query"].tolist()), weights)
        print(f"Fitted historical vocabulary: {event}", flush=True)
    query_factors = token_factors(cooccurrence, seed=seed)
    cooccurrence[:] = 0
    used_items = [item for item in graph_items if item in properties.index]
    for start in range(0, len(used_items), batch_size):
        items = used_items[start:start + batch_size]
        names = properties.loc[items, "name"]
        valid = names.notna().to_numpy()
        if valid.any():
            selected = np.asarray(items)[valid]
            add_cooccurrence(cooccurrence, code_matrix(names.loc[valid].tolist()),
                             [graph_items[item] for item in selected])
    name_factors = token_factors(cooccurrence, seed=seed)
    return {"top_skus": np.asarray(rank_counts(purchases, 128), np.int64),
            "top_categories": np.asarray(rank_counts(purchase_categories, 128), np.int64),
            "top_urls": np.asarray(rank_counts(urls, 128), np.int64),
            "graph_skus": np.asarray(rank_counts(graph_items, graph_skus), np.int64),
            "graph_categories": np.asarray(sorted(graph_categories), np.int64),
            "graph_urls": np.asarray(rank_counts(urls, graph_urls), np.int64),
            "name_factors": name_factors, "query_factors": query_factors}


def feature_layout(state: dict) -> tuple[list[str], list[dict]]:
    names = [f"{event}.{stat}" for event in EVENTS for stat in STATISTICS]
    names += [f"{event}.has_history" for event in EVENTS] + list(CROSS_STATISTICS)
    price_names = ["price_bucket_mean", "price_bucket_max", "price_bucket_std",
                   "price_bucket_sum_30_days", "price_observations"]
    price_names += [f"price_decile_{i}" for i in range(10)]
    for event in PRODUCT_EVENTS:
        names.extend(f"{event}.{name}" for name in price_names)
    blocks = [{"name": "statistics", "start": 0, "stop": len(names), "kind": "statistics"}]

    def append(name, kind, columns):
        start = len(names)
        names.extend(columns)
        blocks.append({"name": name, "start": start, "stop": len(names), "kind": kind})

    append("velocity", "statistics", [f"{event}.{ratio}" for event in EVENTS
           for ratio in ("decay_7_to_30_ratio", "count_7_to_30_ratio", "count_30_to_all_ratio")])
    append("purchase_days", "statistics", ["purchase_active_days", "repeat_purchase_day_observed",
           "mean_purchase_day_interval_days", "purchase_day_recency_to_interval_ratio"])
    for event, kind in (("buy_history", "counts"), ("cart_history", "counts"),
                        ("cart_balance_7_days", "signed_count")):
        append(event, kind, [f"{event}.{column}_{values[i]}" if i < len(values)
                            else f"{event}.unused_{column}_{i}"
                            for column, values in (("sku", state["top_skus"]),
                                                   ("category", state["top_categories"]))
                            for i in range(128)])
    append("url_history", "counts", [f"popular_url_{state['top_urls'][i]}" if i < len(state["top_urls"])
                                     else f"unused_url_{i}" for i in range(128)])
    append("name", "embedding", [f"name_cooccurrence_{i}" for i in range(32)])
    append("query", "embedding", [f"query_cooccurrence_{i}" for i in range(32)])
    append("graph", "embedding", [f"collaborative_{i}" for i in range(128)])
    return names, blocks


def build_behavior(data_directory, output_directory, cutoff_utc=None, reference_directory=None,
                   batch_size=100_000, overwrite=False, seed=42, graph_idf=True,
                   graph_event_weights=True, graph_skus=4096, graph_urls=2048):
    started = time.perf_counter()
    data, output = Path(data_directory), Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "embeddings.npy").exists() and not overwrite:
        raise FileExistsError("Output exists; use --overwrite")
    ids, paths = validate_inputs(data)
    cutoff = _cutoff(paths, cutoff_utc, batch_size)
    properties = pd.read_parquet(data / "product_properties.parquet", dtype_backend="pyarrow").set_index("sku")
    if reference_directory is None:
        state = _fit_vocabulary(paths, properties, cutoff, batch_size, graph_skus, graph_urls, seed)
        settings = {"fit_cutoff_utc": pd.Timestamp(cutoff, tz="UTC").isoformat(),
                    "graph_idf": bool(graph_idf), "graph_event_weights": bool(graph_event_weights),
                    "graph_skus_limit": graph_skus, "graph_urls_limit": graph_urls, "seed": seed}
    else:
        reference = Path(reference_directory).resolve()
        if reference == output.resolve():
            raise ValueError("Use a separate output directory when applying a reference")
        with np.load(reference / STATE_FILE, allow_pickle=False) as saved:
            state = {key: saved[key] for key in saved.files if key != "settings"}
            settings = json.loads(str(saved["settings"]))
        if pd.Timestamp(settings["fit_cutoff_utc"]).value > cutoff:
            raise ValueError("Reference was fitted after the requested input cutoff")
    names, blocks = feature_layout(state)
    offsets = {block["name"]: block["start"] for block in blocks}
    clients = pd.Index(ids)
    matrix = np.lib.format.open_memmap(output / "features.partial.npy", mode="w+", dtype=np.float32,
                                      shape=(len(ids), len(names)), fortran_order=True)
    matrix[:] = 0
    pool_weights = np.zeros((len(ids), 2), np.float32)
    top_indices = {name: pd.Index(state["top_" + name]) for name in ("skus", "categories", "urls")}
    graph_indices, graph_width = {}, 0
    for key in ("categories", "skus", "urls"):
        graph_indices[key] = (pd.Index(state["graph_" + key]), graph_width)
        graph_width += len(state["graph_" + key])
    graph_paths = [output / f"graph_{name}.partial" for name in ("rows", "columns", "weights")]
    streams = [path.open("wb") for path in graph_paths]
    purchase_days = []

    def graph_add(rows, values, column, weights, event):
        index, offset = graph_indices[column]
        codes = index.get_indexer(values)
        valid = codes >= 0
        relation_weight = ({"product_buy": 4.0, "add_to_cart": 2.0, "page_visit": 0.5}[event]
                           if settings["graph_event_weights"] else 1.0)
        rows[valid].astype(np.int32).tofile(streams[0])
        (codes[valid] + offset).astype(np.int32).tofile(streams[1])
        (weights[valid] * relation_weight).astype(np.float32).tofile(streams[2])

    try:
        for event_number, (event, path) in enumerate(zip(EVENTS, paths)):
            first = np.full(len(ids), np.iinfo(np.int64).max, np.int64)
            last = np.full(len(ids), np.iinfo(np.int64).min, np.int64)
            payload = "sku" if event in PRODUCT_EVENTS else "url" if event == "page_visit" else "query"
            for frame in batches(path, ["client_id", "timestamp", payload], batch_size, cutoff):
                rows = clients.get_indexer(frame.client_id)
                valid = rows >= 0
                frame, rows = frame.loc[valid], rows[valid]
                if not len(frame):
                    continue
                times = timestamps_ns(frame.timestamp)
                ages = (cutoff - times) / DAY_NS
                weights = np.exp2(-ages / 30).astype(np.float32)
                base = event_number * len(STATISTICS)
                for offset, values in enumerate((np.ones(len(rows)), ages <= 1, ages <= 7, ages <= 30,
                                                np.exp2(-ages / 7), weights)):
                    np.add.at(matrix[:, base + offset], rows, values)
                np.minimum.at(first, rows, times)
                np.maximum.at(last, rows, times)
                if event in PRODUCT_EVENTS:
                    metadata = properties.reindex(frame.sku)
                    price = metadata.price.to_numpy(dtype=np.float64, na_value=np.nan)
                    valid = np.isfinite(price) & (price >= 0)
                    r, p, w = rows[valid], price[valid], weights[valid]
                    price_offset = 51 + event_number * 15
                    np.add.at(matrix[:, price_offset], r, p)
                    np.maximum.at(matrix[:, price_offset + 1], r, p)
                    np.add.at(matrix[:, price_offset + 2], r, p * p)
                    recent = valid & (ages <= 30)
                    np.add.at(matrix[:, price_offset + 3], rows[recent], price[recent])
                    np.add.at(matrix[:, price_offset + 4], r, 1)
                    np.add.at(matrix, (r, price_offset + 5 + np.minimum((p // 10).astype(int), 9)), w)
                    for column, values, delta in (("skus", frame.sku.to_numpy(), 0),
                                                  ("categories", metadata.category.to_numpy(), 128)):
                        codes = top_indices[column].get_indexer(values)
                        present = codes >= 0
                        if event in ("product_buy", "add_to_cart"):
                            start = offsets["buy_history" if event == "product_buy" else "cart_history"] + delta
                            np.add.at(matrix, (rows[present], start + codes[present]), weights[present])
                            graph_add(rows, values, column, weights, event)
                        recent = present & (ages <= 7)
                        direction = 1.0 if event == "add_to_cart" else -1.0
                        np.add.at(matrix, (rows[recent], offsets["cart_balance_7_days"] + delta + codes[recent]), direction)
                    if event == "product_buy":
                        # Compact unique (user, UTC-day) pairs; repeated basket items count once.
                        day = times // DAY_NS
                        purchase_days.append(np.unique(np.column_stack((rows, day)), axis=0))
                        valid = metadata["name"].notna().to_numpy()
                        if valid.any():
                            vectors = state["name_factors"][code_matrix(metadata.loc[valid, "name"].tolist())].mean(axis=1)
                            np.add.at(matrix[:, offsets["name"]:offsets["name"] + 32], rows[valid], vectors * weights[valid, None])
                            np.add.at(pool_weights[:, 0], rows[valid], weights[valid])
                elif event == "page_visit":
                    codes = top_indices["urls"].get_indexer(frame.url)
                    valid = codes >= 0
                    np.add.at(matrix, (rows[valid], offsets["url_history"] + codes[valid]), weights[valid])
                    graph_add(rows, frame.url.to_numpy(), "urls", weights, event)
                else:
                    vectors = state["query_factors"][code_matrix(frame["query"].tolist())].mean(axis=1)
                    np.add.at(matrix[:, offsets["query"]:offsets["query"] + 32], rows, vectors * weights[:, None])
                    np.add.at(pool_weights[:, 1], rows, weights)
            present = matrix[:, base] > 0
            matrix[:, base + 6] = 366
            matrix[present, base + 6] = (cutoff - last[present]) / DAY_NS
            matrix[present, base + 7] = (last[present] - first[present]) / DAY_NS
            start = offsets["velocity"] + event_number * 3
            matrix[:, start] = matrix[:, base + 4] / (matrix[:, base + 5] + 0.1)
            matrix[:, start + 1] = matrix[:, base + 2] / (matrix[:, base + 3] + 1)
            matrix[:, start + 2] = matrix[:, base + 3] / (matrix[:, base] + 1)
            print(f"Aggregated behavior: {event}", flush=True)
    finally:
        for stream in streams:
            stream.close()

    add_cross_statistics(matrix, names, min(batch_size, 20_000))
    for event_number in range(3):
        start = 51 + event_number * 15
        count = np.maximum(matrix[:, start + 4], 1)
        mean = matrix[:, start] / count
        matrix[:, start + 2] = np.sqrt(np.maximum(matrix[:, start + 2] / count - mean * mean, 0))
        matrix[:, start] = mean
    if purchase_days:
        pairs = np.unique(np.concatenate(purchase_days), axis=0)
        count = np.bincount(pairs[:, 0], minlength=len(ids))
        first = np.full(len(ids), np.iinfo(np.int64).max, np.int64)
        last = np.full(len(ids), np.iinfo(np.int64).min, np.int64)
        np.minimum.at(first, pairs[:, 0], pairs[:, 1])
        np.maximum.at(last, pairs[:, 0], pairs[:, 1])
        eligible = count >= 2
        interval = np.zeros(len(ids), np.float32)
        interval[eligible] = (last[eligible] - first[eligible]) / (count[eligible] - 1)
        ratio = np.zeros(len(ids), np.float32)
        ratio[eligible] = (cutoff // DAY_NS - last[eligible]) / (interval[eligible] + 1)
        start = offsets["purchase_days"]
        matrix[:, start:start + 4] = np.column_stack((count, eligible, interval, ratio))
    del purchase_days
    for number, name in enumerate(("name", "query")):
        start = offsets[name]
        matrix[:, start:start + 32] /= np.maximum(pool_weights[:, number, None], 1e-10)
    del properties, pool_weights

    if graph_paths[0].stat().st_size:
        rows = np.memmap(graph_paths[0], mode="r", dtype=np.int32)
        columns = np.memmap(graph_paths[1], mode="r", dtype=np.int32)
        weights = np.memmap(graph_paths[2], mode="r", dtype=np.float32)
        graph = sparse.coo_matrix((weights, (rows, columns)), shape=(len(ids), graph_width)).tocsr()
        del rows, columns, weights
    else:
        graph = sparse.csr_matrix((len(ids), graph_width), dtype=np.float32)
    graph.data = np.log1p(graph.data)
    if reference_directory is None:
        frequency = np.bincount(graph.indices, minlength=graph_width)
        state["graph_idf"] = ((np.log((len(ids) + 1) / (frequency + 1)) + 1).astype(np.float32)
                              if settings["graph_idf"] else np.ones(graph_width, np.float32))
    graph.data *= state["graph_idf"][graph.indices]
    degree = np.sqrt(np.asarray(graph.power(2).sum(axis=1)).ravel())
    graph = sparse.diags(1 / np.maximum(degree, 1e-10)) @ graph
    if reference_directory is None:
        rank = min(128, min(graph.shape) - 1) if graph.nnz else 0
        state["graph_components"] = np.zeros((128, graph_width), np.float32)
        if rank > 0:
            _, _, right = randomized_svd(graph, n_components=rank, n_iter=3, random_state=seed)
            state["graph_components"][:rank] = right
        settings["graph_rank"] = rank
        np.savez(output / STATE_FILE, **state, settings=np.asarray(json.dumps(settings)))
    else:
        shutil.copyfile(reference / STATE_FILE, output / STATE_FILE)
    start_column = offsets["graph"]
    for start in range(0, len(ids), 20_000):
        values = np.asarray(graph[start:start + 20_000] @ state["graph_components"].T)
        values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-10)
        values[degree[start:start + len(values)] == 0] = 0
        matrix[start:start + len(values), start_column:start_column + 128] = values
    del graph
    for path in graph_paths:
        path.unlink()
    for start in range(0, len(ids), 20_000):
        chunk = matrix[start:start + 20_000]
        chunk[:, :96] = np.log1p(chunk[:, :96])
        day_start = offsets["purchase_days"]
        chunk[:, day_start:day_start + 4] = np.log1p(chunk[:, day_start:day_start + 4])
        for name in ("buy_history", "cart_history", "url_history"):
            block = next(block for block in blocks if block["name"] == name)
            chunk[:, block["start"]:block["stop"]] = np.log1p(chunk[:, block["start"]:block["stop"]])
        balance = chunk[:, offsets["cart_balance_7_days"]:offsets["cart_balance_7_days"] + 256]
        balance[:] = np.sign(balance) * np.log1p(np.abs(balance))
    del chunk, balance  # Release memmap views before renaming on Windows.
    matrix.flush()
    del matrix
    (output / "features.partial.npy").replace(output / "embeddings.npy")
    np.save(output / "client_ids.npy", ids)
    (output / "features.json").write_text(json.dumps(names, indent=2) + "\n", encoding="utf-8")
    manifest = {"method": "separate_purchase_cart_balance_velocity_and_frozen_content_graph",
                "clients": len(ids), "dimensions": len(names), "dtype": "float32", "blocks": blocks,
                "input_cutoff_utc": pd.Timestamp(cutoff, tz="UTC").isoformat(),
                **settings, "reference_used": reference_directory is not None,
                "state_sha256": _sha256(output / STATE_FILE),
                "target_data_used_for_embeddings": False,
                "value_transforms": {"statistics": "log1p", "velocity": "none", "purchase_days": "log1p",
                                     "counts": "log1p", "signed_count": "signed_log1p",
                                     "name_query": "time_weighted_mean", "graph": "row_l2"},
                "top_identifiers_selected_from": {"sku_category": "purchase events at or before fit cutoff",
                                                   "url": "page_visit events at or before fit cutoff"},
                "official_target_lists_used_for_vocabulary": False,
                "cart_balance_definition": "7-day cart events minus remove and purchase events; signed log1p; not cart state",
                "purchase_interval_definition": "Separate UTC purchase days; unobserved repeat intervals are zero with an explicit availability flag",
                "projection_definition": "All windows use log1p graph, frozen IDF, row L2, X @ fixed Vt.T, then row L2",
                "pq_interpretation": "Independent name/query position-code co-occurrence bases fitted before fit cutoff",
                "seconds": round(time.perf_counter() - started, 3)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--cutoff-utc")
    parser.add_argument("--reference-directory", type=Path)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--graph-skus", type=int, default=4096)
    parser.add_argument("--graph-urls", type=int, default=2048)
    parser.add_argument("--disable-graph-idf", dest="graph_idf", action="store_false")
    parser.add_argument("--equal-graph-event-weights", dest="graph_event_weights", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    print(json.dumps(build_behavior(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
