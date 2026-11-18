"""Embed recent carts and visited URLs through historical next-purchase baskets.

All time comparisons use the input event timestamps. A basket is the set of
distinct items bought by one user at one timestamp. Attribution to the first
future basket bounds the join while keeping simultaneous purchases together.
"""
import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.parquet as parquet
from scipy import sparse
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from .build_embeddings import DAY_NS, input_cutoff, timestamps_ns
from .product_cooccurrence import product_factors, weighted_history


RELATIONS = (("add_to_cart", "sku"), ("page_visit", "url"))
LINK_TYPE = np.dtype([("node", "<i8"), ("basket", "<i4")])
RECENT_TYPE = np.dtype([("node", "<i8"), ("user", "<i4"), ("weight", "<f4")])


def event_batches(path, field, clients, cutoff, batch_size):
    file = parquet.ParquetFile(path)
    for batch in file.iter_batches(batch_size=batch_size, columns=["client_id", "timestamp", field],
                                   use_threads=False):
        users = clients.get_indexer(batch.column(0).to_numpy(zero_copy_only=False))
        times = timestamps_ns(batch.column(1).to_pandas())
        selected = (users >= 0) & (times <= cutoff) & batch.column(2).is_valid().to_numpy(zero_copy_only=False)
        nodes = batch.column(2).fill_null(0).to_numpy(zero_copy_only=False)
        yield users[selected], times[selected], nodes[selected]


@dataclass
class Baskets:
    offsets: np.ndarray
    times: np.ndarray
    products: sparse.csr_matrix


def purchase_baskets(users, times, items, n_clients):
    order = np.lexsort((times, users))
    users, times, items = users[order], times[order], items[order]
    first = np.ones(len(users), dtype=bool)
    first[1:] = (users[1:] != users[:-1]) | (times[1:] != times[:-1])
    basket_id = np.cumsum(first, dtype=np.int32) - 1
    product_ids, columns = np.unique(items, return_inverse=True)
    n_baskets = int(first.sum())
    matrix = sparse.csr_matrix((np.ones(len(items), np.float32), (basket_id, columns)),
                               shape=(n_baskets, len(product_ids)))
    if matrix.nnz:
        matrix.data.fill(1)
        normalize(matrix, norm="l1", axis=1, copy=False)
    offsets = np.r_[0, np.cumsum(np.bincount(users[first], minlength=n_clients))]
    return Baskets(offsets, times[first], matrix)


def next_baskets(users, times, baskets):
    """Vectorized upper-bound search only within the user's own date range."""
    low, end = baskets.offsets[users], baskets.offsets[users + 1]
    selected = np.flatnonzero(low < end)
    low, end, times = low[selected], end[selected], times[selected]
    high = end.copy()
    pending = np.arange(len(low))
    while len(pending):
        mid = (low[pending] + high[pending]) // 2
        later = baskets.times[mid] > times[pending]
        high[pending[later]] = mid[later]
        low[pending[~later]] = mid[~later] + 1
        pending = pending[low[pending] < high[pending]]
    found = np.flatnonzero(low < end)
    found = found[baskets.times[low[found]] - times[found] <= 7 * DAY_NS]
    return selected[found], low[found]


def write_columns(stream, dtype, columns):
    packed = np.empty(len(columns[0]), dtype=dtype)
    for field, values in zip(dtype.names, columns):
        packed[field] = values
    packed.tofile(stream)


def sparse_from_stream(path, dtype, row_field, node_index, shape):
    """Merge one chunk at a time; historical links are node x basket."""
    result = sparse.csr_matrix(shape, dtype=np.float32)
    retained = 0
    with path.open("rb") as data:
        while len(chunk := np.fromfile(data, dtype=dtype, count=500_000)):
            nodes = node_index.get_indexer(chunk["node"])
            keep = nodes >= 0
            retained += int(keep.sum())
            nodes = nodes[keep]
            rows = chunk[row_field][keep]
            if row_field == "basket":
                values = np.ones(len(nodes), np.float32)
                rows, columns = nodes, rows
            else:
                values = chunk["weight"][keep]
                columns = nodes
            result += sparse.csr_matrix((values, (rows, columns)), shape=shape)
    return result, retained


def build_relation(name, field, directory, clients, baskets, cutoff, temp, source_limit, batch_size):
    link_path, recent_path = temp / (name + "_links.bin"), temp / (name + "_recent.bin")
    frequency = Counter()
    eligible = matches = recent_total = relevant_total = 0
    with link_path.open("wb") as links, recent_path.open("wb") as recent:
        for users, times, nodes in event_batches(directory / (name + ".parquet"), field,
                                                clients, cutoff, batch_size):
            relevant_total += len(users)
            complete = times <= cutoff - 7 * DAY_NS
            eligible += int(complete.sum())
            positions, following = next_baskets(users[complete], times[complete], baskets)
            linked_nodes = nodes[complete][positions]
            matches += len(following)
            unique_nodes, counts = np.unique(linked_nodes, return_counts=True)
            frequency.update(dict(zip(unique_nodes.tolist(), counts.tolist())))
            write_columns(links, LINK_TYPE, (linked_nodes, following))
            latest = times > cutoff - 14 * DAY_NS
            ages = (cutoff - times[latest]) / DAY_NS
            recent_total += int(latest.sum())
            write_columns(recent, RECENT_TYPE, (nodes[latest], users[latest], np.exp2(-ages / 7)))
    raw_linked_nodes = len(frequency)
    ranked = sorted(frequency, key=lambda node: (-frequency[node], node))
    kept_nodes = np.asarray(ranked[:source_limit], dtype=np.int64)
    del frequency, ranked
    vocabulary = pd.Index(kept_nodes)
    edges, included_matches = sparse_from_stream(link_path, LINK_TYPE, "basket", vocabulary,
                                                 (len(vocabulary), len(baskets.times)))
    # Repeated page loads or cart clicks before the same basket count once.
    edges.data.fill(1)
    supported = np.diff(edges.indptr) >= 2
    edges = edges[supported]
    distinct_links = edges.nnz
    transition = (edges @ baskets.products).tocsr()
    del edges
    history, retained_recent = sparse_from_stream(recent_path, RECENT_TYPE, "user",
                                                pd.Index(kept_nodes[supported]),
                                                (len(clients), int(supported.sum())))
    spool_bytes = link_path.stat().st_size + recent_path.stat().st_size
    link_path.unlink()
    recent_path.unlink()
    stats = dict(relevant_input_events=int(relevant_total), sources_with_observable_seven_days=eligible,
                 events_matched_to_next_basket=int(matches), raw_linked_sources=raw_linked_nodes,
                 source_nodes_before_basket_filter=len(vocabulary), retained_source_nodes=int(supported.sum()),
                 matches_before_basket_filter=int(included_matches), retained_distinct_source_baskets=distinct_links,
                 directed_source_product_edges=transition.nnz, recent_events=recent_total,
                 recent_events_with_retained_nodes=retained_recent, recent_edges=history.nnz,
                 users_with_retained_recent_sources=int(np.count_nonzero(np.diff(history.indptr))),
                 temporary_stream_bytes=spool_bytes)
    print(name, json.dumps(stats), flush=True)
    return transition, history, stats


def propagated_embeddings(transitions, recent, destination, dimensions, seed):
    graph = sparse.vstack(transitions, format="csr")
    spectral = weighted_history(graph)[0] if graph.nnz else graph
    del graph
    factors, singular, rank = product_factors(spectral, dimensions=dimensions,
                                              oversamples=12, power_iterations=2, seed=seed)
    del spectral
    projected = []
    for transition, history in zip(transitions, recent):
        if transition.nnz:
            normalize(transition, norm="l1", axis=1, copy=False)
        projected.append(transition @ factors)
        if history.nnz:
            normalize(history, norm="l1", axis=1, copy=False)
    count = recent[0].shape[0]
    output = np.lib.format.open_memmap(destination, mode="w+", dtype=np.float16,
                                       shape=(count, dimensions))
    nonzero = 0
    for start in range(0, count, 8192):
        end = min(count, start + 8192)
        values = np.zeros((end - start, dimensions), dtype=np.float32)
        for matrix, source_vectors in zip(recent, projected):
            values += matrix[start:end] @ source_vectors
        nonzero += int(np.any(values != 0, axis=1).sum())
        output[start:end] = normalize(values, norm="l2", copy=False)
    del output
    return rank, singular.tolist(), nonzero


def build(data_directory, output_directory, dimensions=64, source_limit=65_536,
          batch_size=250_000, threads=2, seed=42, cutoff_utc=None):
    data = Path(data_directory) / "input"
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "embeddings.npy").exists() or (output / "manifest.json").exists():
        raise FileExistsError(output)
    if min(dimensions, source_limit, batch_size, threads) <= 0:
        raise ValueError("Dimensions, source limit, batch size and threads must be positive")
    started = perf_counter()
    paths = [data / (name + ".parquet") for name in ("product_buy", "add_to_cart", "page_visit")]
    cutoff = (input_cutoff(paths, batch_size) if cutoff_utc is None
              else pd.to_datetime(cutoff_utc, utc=True).value)
    ids = np.load(data / "relevant_clients.npy", allow_pickle=False)
    clients = pd.Index(ids)
    if ids.ndim != 1 or ids.dtype != np.int64 or not len(ids) or clients.has_duplicates:
        raise ValueError("Expected unique int64 relevant client IDs")
    purchase_parts = list(event_batches(paths[0], "sku", clients, cutoff, batch_size))
    purchases = [np.concatenate(columns) for columns in zip(*purchase_parts)] if purchase_parts else [
        np.empty(0, dtype=np.int64) for _ in range(3)]
    baskets = purchase_baskets(*purchases, len(ids))
    del purchase_parts, purchases
    relations = {}
    with TemporaryDirectory(dir=output) as stage, threadpool_limits(limits=threads):
        temp = Path(stage)
        transitions, recent = [], []
        for name, field in RELATIONS:
            graph, history, stats = build_relation(name, field, data, clients, baskets, cutoff,
                                                  temp, source_limit, batch_size)
            transitions.append(graph)
            recent.append(history)
            relations[name] = stats
        rank, singular, nonzero = propagated_embeddings(transitions, recent, temp / "embeddings.npy",
                                                       dimensions, seed)
        np.save(temp / "client_ids.npy", ids)
        names = [f"purchase_transition_{i}" for i in range(dimensions)]
        (temp / "features.json").write_text(json.dumps(names), encoding="utf-8")
        report = dict(clients=len(ids), dimensions=dimensions, dtype="float16",
                      blocks=[dict(name="purchase_transition", start=0, stop=dimensions, kind="embedding")],
                      method="Visit URL/cart SKU -> earliest strictly future purchase basket within seven days; each distinct source-basket once, uniform across its distinct SKUs",
                      graph_inputs="History of relevant_clients only; exclude sources lacking a full subsequent seven days before input cutoff",
                      pooling="Past 14 days; 7-day half-life, average per relation, add relations, row-L2 normalize. Empty users remain zero",
                      fit="Stacked source-product counts with log1p, distinct-source IDF, row L2; randomized spectral product factors. Propagation uses the row-L1 transition probabilities",
                      source_selection="Most matched historical events per source ID and event type, descending; ties by ID; then require two distinct baskets",
                      source_limit_per_relation=source_limit, seed=seed, threads=threads,
                      effective_rank=rank, singular_values=singular, nonzero_user_vectors=nonzero,
                      baskets=len(baskets.times), observed_purchase_products=baskets.products.shape[1],
                      distinct_basket_products=baskets.products.nnz, relations=relations,
                      input_cutoff_utc=pd.Timestamp(cutoff, tz="UTC").isoformat(),
                      official_target_directory_read=False, seconds=round(perf_counter()-started,3))
        (temp / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        for name in ("features.json", "client_ids.npy", "embeddings.npy", "manifest.json"):
            (temp / name).replace(output / name)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--dimensions", type=int, default=64)
    parser.add_argument("--source-limit", type=int, default=65_536)
    parser.add_argument("--batch-size", type=int, default=250_000)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cutoff-utc")
    print(json.dumps(build(**vars(parser.parse_args())), indent=2), flush=True)
