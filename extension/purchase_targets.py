"""Train compact product/category propensity heads using only historical purchase windows.

The aligned multiwindow cache supplies pre-decision user profiles. Sparse future
purchases provide labels. All supervised fits, including the probability PCA,
exclude the cache's held-out users. No official target data is opened.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.parquet as parquet
from scipy import sparse
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.utils.extmath import randomized_svd
from threadpoolctl import threadpool_limits

from .build_embeddings import timestamps_ns


def profile_batches(path, size, selected=None):
    """Read row-major NPY by bounded ordinary reads, including on memory-constrained hosts."""
    matrix = np.load(path, mmap_mode="r")
    if matrix.ndim != 2 or not matrix.flags.c_contiguous:
        raise ValueError(f"Expected a row-major profile matrix: {path}")
    rows, width = matrix.shape
    offset, dtype = matrix.offset, matrix.dtype
    del matrix
    with path.open("rb") as stream:
        for begin in range(0, rows, size):
            end = min(begin + size, rows)
            indices = np.arange(begin, end) if selected is None else (
                selected[np.searchsorted(selected, begin):np.searchsorted(selected, end)])
            if not len(indices):
                continue
            stream.seek(offset + begin * width * dtype.itemsize)
            block = np.fromfile(stream, dtype=dtype, count=(end - begin) * width).reshape(-1, width)
            yield indices, block[indices - begin].astype(np.float32, copy=False)


def purchase_labels(data, ids, validation, windows, limits, minimum):
    catalog = parquet.read_table(data / "product_properties.parquet", columns=["sku", "category"],
                                 use_threads=False).to_pandas().set_index("sku")["category"]
    clients = pd.Index(ids)
    events = []
    for batch in parquet.ParquetFile(data / "input" / "product_buy.parquet").iter_batches(
            batch_size=64_000, columns=["client_id", "timestamp", "sku"], use_threads=False):
        chunk = batch.to_pandas()
        rows = clients.get_indexer(chunk.client_id)
        times = timestamps_ns(chunk.timestamp)
        for i, win in enumerate(windows):
            keep = ((rows >= 0) & (times >= pd.Timestamp(win["target_start_inclusive_utc"]).value)
                    & (times <= pd.Timestamp(win["target_end_inclusive_utc"]).value))
            if keep.any():
                skus = chunk.sku.to_numpy()[keep]
                events.append(pd.DataFrame({"row": rows[keep], "sku": skus,
                                            "category": catalog.reindex(skus).to_numpy(), "window": i}))
    if not events:
        raise ValueError("The historical target windows contain no purchases for the supplied users")
    events = pd.concat(events, ignore_index=True)
    training = events.loc[~validation[events.row.to_numpy()]]
    vocabulary, matrices = {}, {}
    for field, limit in limits.items():
        counts = training[field].value_counts().rename_axis("id").reset_index(name="count")
        counts = counts.loc[counts["count"] >= minimum].sort_values(["count", "id"], ascending=[False, True])
        levels = counts.head(limit)["id"].to_numpy(dtype=np.int64)
        if len(levels) < 2:
            raise ValueError(f"Fewer than two eligible historical {field} labels")
        vocabulary[field] = levels
        indices = pd.Index(levels).get_indexer(events[field])
        matrices[field] = []
        for i in range(len(windows)):
            keep = (indices >= 0) & (events.window.to_numpy() == i)
            mat = sparse.csr_matrix((np.ones(keep.sum(), np.float32),
                                    (events.row.to_numpy()[keep], indices[keep])), shape=(len(ids), len(levels)))
            mat.data.fill(1)  # one label per user and item, regardless of repeated purchase lines
            matrices[field].append(mat)
    return vocabulary, matrices, len(training)


def standardize(x, mean, scale):
    return np.clip((x - mean) / scale, -6, 6)


def sample_rows(indices, n, rng):
    return np.sort(rng.choice(indices, min(n, len(indices)), replace=False))


def supervised_basis(paths, matrices, train, mean, scale, centers, size, rank, seed):
    """Find historical feature directions whose class-conditional means differ from the population."""
    cross = np.zeros((matrices[0].shape[1], len(mean)), np.float32)
    support = np.zeros(len(cross), np.float32)
    for path, labels, center in zip(paths, matrices, centers):
        positive_rows = np.flatnonzero(train & (np.diff(labels.indptr) != 0))
        for rows, raw in profile_batches(path, size, positive_rows):
            members = labels[rows].tocoo()
            x = standardize(raw, mean, scale) - center
            np.add.at(cross, members.col, x[members.row])
            support += np.bincount(members.col, minlength=len(support))
    # Shrink rare noisy means while retaining more tail evidence than raw count sums.
    cross /= (support + 32)[:, None] ** 0.8
    width = min(rank, *cross.shape)
    _, _, directions = randomized_svd(cross, n_components=width, n_iter=3, random_state=seed)
    return directions.T.copy()


def fit_heads(x, labels, background, rng):
    """Each model gives equal total logistic-loss weight to positive and sampled negative rows."""
    labels = labels.tocsc()
    weight = np.zeros((x.shape[1], labels.shape[1]), np.float32)
    bias = np.zeros(labels.shape[1], np.float32)
    for col in range(labels.shape[1]):
        positive = labels.indices[labels.indptr[col]:labels.indptr[col + 1]]
        sample_size = min(len(background), max(256, min(4096, 16 * len(positive))))
        candidates = rng.choice(background, size=sample_size, replace=False)
        negative = candidates[~np.isin(candidates, positive)]
        if not len(positive) or not len(negative):
            raise ValueError("Balanced head requires both a positive and a negative training case")
        indices = np.concatenate((positive, negative))
        y = np.arange(len(indices)) < len(positive)
        model = LogisticRegression(C=12 / len(indices), solver="liblinear", max_iter=100,
                                   class_weight="balanced", random_state=42)
        model.fit(x[indices], y)
        weight[:, col] = model.coef_[0]
        bias[col] = model.intercept_[0]
    return weight, bias


def normalized_latent(raw, mean, scale, basis, center, deviation):
    return np.clip((standardize(raw, mean, scale) @ basis - center) / deviation, -8, 8)


def projected_data(paths, rows, size, mean, scale, basis):
    parts = []
    for path, selected in zip(paths, rows):
        parts.append(np.concatenate([standardize(raw, mean, scale) @ basis
                                     for _, raw in profile_batches(path, size, selected)]))
    return np.concatenate(parts)


def fit_block(paths, matrices, train, train_rows, validation_sample, backgrounds,
              mean, scale, centers, size, rank, seed):
    rng = np.random.default_rng(seed)
    basis = supervised_basis(paths, matrices, train, mean, scale, centers, size, rank, seed)
    design = projected_data(paths, train_rows, size, mean, scale, basis)
    background = np.concatenate([np.searchsorted(rows, bg) + offset
                                 for rows, bg, offset in zip(train_rows, backgrounds, [0, len(train_rows[0])])])
    center = design[background].mean(axis=0)
    deviation = np.maximum(design[background].std(axis=0), 0.01)
    design = np.clip((design - center) / deviation, -8, 8)
    train_labels = sparse.vstack([labels[rows] for labels, rows in zip(matrices, train_rows)], format="csr")
    weight, bias = fit_heads(design, train_labels, background, rng)

    # Sampling precedes PCA so no users x all classes array is needed for extraction.
    pca_sample = sample_rows(background, 8192, rng)
    probs = expit(design[pca_sample] @ weight + bias)
    probability_mean = probs.mean(axis=0)
    probs -= probability_mean
    width = min(rank, probs.shape[1], len(probs) - 1)
    _, singular, axes = randomized_svd(probs, n_components=width, n_iter=3, random_state=seed)
    projected_scale = np.maximum(singular / np.sqrt(max(1, len(probs) - 1)), 0.015).astype(np.float32)

    val_x = projected_data(paths, [validation_sample] * len(paths), size, mean, scale, basis)
    val_x = np.clip((val_x - center) / deviation, -8, 8)
    val_labels = sparse.vstack([m[validation_sample] for m in matrices], format="csc")
    scores = []
    for start in range(0, weight.shape[1], 32):
        stop = min(start + 32, weight.shape[1])
        logits = val_x @ weight[:, start:stop] + bias[start:stop]
        truth = val_labels[:, start:stop].toarray()
        kept = (truth.sum(axis=0) > 0) & (truth.sum(axis=0) < len(truth))
        if kept.any():
            scores.extend(np.atleast_1d(roc_auc_score(truth[:, kept], logits[:, kept], average=None)).tolist())
    return dict(basis=basis, center=center, deviation=deviation, weight=weight, bias=bias,
                probability_mean=probability_mean, axes=axes.T.copy(), projected_scale=projected_scale), {
        "classes_with_heldout_positives": len(scores), "heldout_macro_auc_supported": float(np.mean(scores)) if scores else None,
        "heldout_user_window_rows": len(val_x), "training_rows_for_head_pool": len(design),
        "uniform_training_background": len(background), "training_probability_pca_rows": len(probs),
        "probability_pca_explained_fraction": float(np.square(singular).sum() / max(np.square(probs).sum(), 1e-12))}


def build(data, cache, output, sku_limit=2000, category_limit=500, minimum=15,
          rank=64, batch_size=4096, seed=42):
    started = perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "embeddings.npy").exists():
        raise FileExistsError(output / "embeddings.npy")
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    windows = [w for w in manifest["windows"] if w["feature_window"] == w["target_window"]
               and w["forecast_gap_days"] == 0]
    if len(windows) != 2 or windows[0]["target_end_inclusive_utc"] >= windows[1]["target_start_inclusive_utc"]:
        raise ValueError("Expected two ordered, disjoint historical fortnights in the multiwindow cache")
    windows = sorted(windows, key=lambda w: w["target_start_inclusive_utc"])
    limits = {"sku": sku_limit, "category": category_limit}
    ids = np.load(data / "input" / "relevant_clients.npy")
    if not np.array_equal(ids, np.load(cache / "client_ids.npy")):
        raise ValueError("Cache users differ from historical input")
    paths = [cache / w["feature_window"] / "embeddings.npy" for w in windows]
    full = cache / "full" / "embeddings.npy"
    for path in [*paths, full]:
        if not np.array_equal(ids, np.load(path.parent / "client_ids.npy")):
            raise ValueError(f"User order mismatch: {path}")
    train = ~np.load(cache / "validation_users.npy")
    if train.shape != (len(ids),) or train.dtype != bool:
        raise ValueError("Expected a boolean validation split aligned with clients")
    vocabulary, labels, events = purchase_labels(data, ids, ~train, windows, limits, minimum)
    print("Historical training purchase lines:", events, "target counts:", {k: len(v) for k, v in vocabulary.items()}, flush=True)
    rng = np.random.default_rng(seed)
    training_users = np.flatnonzero(train)
    backgrounds = [sample_rows(training_users, 40_000, rng) for _ in paths]
    validation_sample = sample_rows(np.flatnonzero(~train), 10_000, rng)
    train_rows = []
    moments = []
    for i, path in enumerate(paths):
        any_positive = np.logical_or.reduce([np.diff(labels[field][i].indptr) != 0 for field in limits])
        train_rows.append(np.union1d(backgrounds[i], np.flatnonzero(train & any_positive)))
        moment_users = sample_rows(backgrounds[i], 8192, rng)
        moments.append(np.concatenate([x for _, x in profile_batches(path, batch_size, moment_users)]))
    both = np.concatenate(moments)
    mean = both.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(both.std(axis=0, dtype=np.float64), 0.01).astype(np.float32)
    centers = [standardize(x, mean, scale).mean(axis=0) for x in moments]
    del moments, both
    blocks, diagnostics = {}, {}
    for field in limits:
        print(f"Training {field} historical heads", flush=True)
        blocks[field], diagnostics[field] = fit_block(paths, labels[field], train, train_rows,
            validation_sample, backgrounds, mean, scale, centers, batch_size, rank, seed)
        print(field, diagnostics[field], flush=True)
    columns = [f"historical_{field}_probability_component_{i}" for field, b in blocks.items()
               for i in range(b["axes"].shape[1])]
    partial = output / "embeddings.partial.npy"
    vectors = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float16, shape=(len(ids), len(columns)))
    for rows, raw in profile_batches(full, batch_size):
        pieces = []
        for b in blocks.values():
            z = normalized_latent(raw, mean, scale, b["basis"], b["center"], b["deviation"])
            p = expit(z @ b["weight"] + b["bias"])
            scores = ((p - b["probability_mean"]) @ b["axes"]) / b["projected_scale"]
            pieces.append(np.clip(scores, -6, 6))
        values = np.concatenate(pieces, axis=1)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite compressed historical head scores")
        vectors[rows[0]:rows[-1] + 1] = values
    del vectors
    parameters = {f"{field}_{name}": v for field, b in blocks.items() for name, v in b.items()}
    np.savez(output / "heads.npz", mean=mean, scale=scale, **parameters, **vocabulary)
    np.save(output / "client_ids.npy", ids, allow_pickle=False)
    (output / "features.json").write_text(json.dumps(columns), encoding="utf-8")
    record = {"clients": len(ids), "dimensions": len(columns), "dtype": "float16",
        "training_purchase_lines_vocabulary": events, "training_users": int(train.sum()),
        "heldout_users": int((~train).sum()), "min_purchase_lines": minimum, "requested_caps": limits,
        "historical_classes": {k: len(v) for k, v in vocabulary.items()},
        "historical_windows": [{k: w[k] for k in ("feature_window", "input_cutoff_utc",
            "target_start_inclusive_utc", "target_end_inclusive_utc")} for w in windows],
        "method": "Supervised low-rank feature directions, per-class balanced binary logistic heads, centered probability PCA",
        "features": "Existing aligned 1203-dimensional aggregate profiles; earliest history feature definitions frozen",
        "label_selection": "Purchase-event frequency of training users in the two disjoint historical target windows",
        "negatives": "Uniform training-user window background, excluding each class's own positive cases",
        "diagnostics": diagnostics, "official_target_directory_read": False, "seed": seed,
        "seconds": round(perf_counter() - started, 3)}
    (output / "manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    partial.replace(output / "embeddings.npy")
    print(record, flush=True)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("data-directory", "cache-directory", "output-directory"):
        parser.add_argument("--" + argument, type=Path, required=True)
    parser.add_argument("--sku-limit", type=int, default=2000)
    parser.add_argument("--category-limit", type=int, default=500)
    parser.add_argument("--minimum-purchases", type=int, default=15)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    with threadpool_limits(limits=args.threads):
        build(args.data_directory, args.cache_directory, args.output_directory, args.sku_limit,
              args.category_limit, args.minimum_purchases, args.rank, args.batch_size, args.seed)
