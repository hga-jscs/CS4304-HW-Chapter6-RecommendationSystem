"""Compress previously evaluated Synerise blocks using only historical user vectors."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits


# Separate projections retain purchase, cart, and net-cart semantics.
HISTORY_RANKS = {"buy_history": 96, "cart_history": 96, "cart_balance_7_days": 48,
                 "url_history": 29, "graph": 64}
BATCH = 4096


def read_profile(directory):
    names = json.loads((directory / "features.json").read_text(encoding="utf-8-sig"))
    matrix = np.load(directory / "embeddings.npy", mmap_mode="r")
    ids = np.load(directory / "client_ids.npy")
    if matrix.shape != (len(ids), len(names)) or matrix.dtype != np.float16:
        raise ValueError(f"Invalid embedding layout in {directory}")
    return ids, names, matrix


def fit_projection(matrix, start, stop, selected, rank):
    # Accumulate small per-block sufficient statistics rather than a user covariance.
    width = stop - start
    total = np.zeros(width, np.float64)
    cross = np.zeros((width, width), np.float64)
    for offset in range(0, len(selected), BATCH):
        rows = selected[offset:offset + BATCH]
        sample = np.asarray(matrix[rows, start:stop], dtype=np.float64)
        total += sample.sum(axis=0)
        cross += sample.T @ sample
    mean = total / len(selected)
    covariance = cross / len(selected) - np.outer(mean, mean)
    active = np.flatnonzero(np.diag(covariance) > 1e-10)
    if not len(active):
        return mean.astype(np.float32), np.zeros((width, rank), np.float32), 0.0, 0
    values, vectors = np.linalg.eigh(covariance[np.ix_(active, active)])
    count = min(rank, len(values))
    top = vectors[:, -count:][:, ::-1].copy()
    pivot = np.abs(top).argmax(axis=0)
    top *= np.where(top[pivot, np.arange(count)] < 0, -1., 1.)
    projection = np.zeros((width, rank), np.float32)
    projection[active, :count] = top
    energy = np.maximum(values, 0)
    return (mean.astype(np.float32), projection, float(energy[-count:].sum() / energy.sum()),
            len(active))


def compress(profile_directory, activity_directory, product_directory, output_directory,
             sample_users=65536, seed=42):
    started = perf_counter()
    ids, names, profile = read_profile(profile_directory)
    activity_ids, activity_names, activity = read_profile(activity_directory)
    product_ids, product_names, products = read_profile(product_directory)
    if not np.array_equal(ids, activity_ids) or not np.array_equal(ids, product_ids):
        raise ValueError("Client order differs across input profiles")
    if sample_users < 2 or len(ids) < 2:
        raise ValueError("PCA needs at least two user profiles")
    schema = json.loads((profile_directory / "standardization.json").read_text(encoding="utf-8-sig"))
    blocks = {b["name"]: (b["start"], b["stop"]) for b in schema["blocks"]}
    expected = {"pretrained_shared", "pretrained_churn", "pretrained_immediate_scores",
                "pretrained_delayed_scores", "statistics", "velocity", "purchase_days",
                "buy_history", "cart_history", "cart_balance_7_days", "url_history", "name", "query", "graph"}
    if set(blocks) != expected or activity_names[-2:] != ["repurchase_days_1_14", "repurchase_days_15_28"]:
        raise ValueError("Expected the multiwindow block-standardized profile and activity window scores")
    output_directory.mkdir(parents=True, exist_ok=True)
    destination = output_directory / "embeddings.npy"
    if destination.exists():
        raise FileExistsError(destination)
    selected = np.sort(np.random.default_rng(seed).choice(len(ids), min(len(ids), sample_users), replace=False))
    # Each plan specifies the matrix branch, contiguous slice, optional second horizon and basis.
    plans, output_names, summary, bases = [], [], [], {}
    def add(label, matrix, source_names, start, stop, weight=1.0, rank=None, paired=None):
        mean = basis = None
        entry = {"name": label, "input_width": stop-start, "weight": weight}
        labels = source_names[start:stop]
        if rank is not None:
            mean, basis, ratio, varying = fit_projection(matrix, start, stop, selected, rank)
            labels = [f"{label}.component_{i}" for i in range(rank)]
            entry.update(variance_fraction=ratio, varying_columns=varying)
            bases[label + "_mean"] = mean
            bases[label + "_basis"] = basis
        if paired is not None:
            labels = [name.replace("_immediate_", "_average_") for name in labels]
        output_names.extend(labels)
        entry["output_width"] = len(labels)
        summary.append(entry)
        plans.append((matrix, start, stop, weight, paired, mean, basis))

    with threadpool_limits(limits=2):
        for b in ("pretrained_shared", "pretrained_churn"):
            add(b, profile, names, *blocks[b])
        a, b = blocks["pretrained_immediate_scores"]
        c, d = blocks["pretrained_delayed_scores"]
        near = [name.replace("_immediate_", "_") for name in names[a:b]]
        far = [name.replace("_delayed_", "_") for name in names[c:d]]
        if near != far or "churn" not in near[0]:
            raise ValueError("Near and delayed auxiliary classes must agree in order")
        add("near_churn", profile, names, a, a + 1)
        add("delayed_churn", profile, names, c, c + 1)
        add("averaged_propensity", profile, names, a + 1, b, paired=(c + 1, d))
        for group in schema["blocks"]:
            label = group["name"]
            if label.startswith("pretrained_"):
                continue
            factor = 0.8 if group["kind"] in {"statistics", "counts", "signed_count"} else 1.0
            add(label, profile, names, *blocks[label], weight=factor, rank=HISTORY_RANKS.get(label))
        add("activity", activity, activity_names, 0, len(activity_names)-2, weight=0.8)
        add("historical_repeat_scores", activity, activity_names, len(activity_names)-2,
            len(activity_names), weight=1.3)
        for prefix in ("product_buy_cooccurrence_", "add_to_cart_cooccurrence_"):
            columns = [i for i, name in enumerate(product_names) if name.startswith(prefix)]
            if len(columns) != 64 or columns != list(range(columns[0], columns[0]+64)):
                raise ValueError(f"Expected 64 consecutive product factors for {prefix}")
            add(prefix.rstrip("_"), products, product_names, columns[0], columns[-1]+1, rank=32)
        temp = output_directory / "compact.partial.npy"
        result = np.lib.format.open_memmap(temp, mode="w+", dtype=np.float16,
                                           shape=(len(ids), len(output_names)))
        for offset in range(0, len(ids), BATCH):
            rows = slice(offset, min(offset+BATCH, len(ids)))
            cursor = 0
            for matrix, start, stop, weight, paired, mean, basis in plans:
                values = np.asarray(matrix[rows, start:stop], dtype=np.float32)
                if paired is not None:
                    values += matrix[rows, paired[0]:paired[1]]
                    values *= 0.5
                if basis is not None:
                    values = (values-mean) @ basis
                values *= weight
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite projected features")
                result[rows, cursor:cursor+values.shape[1]] = values
                cursor += values.shape[1]
        result.flush()
        del result
    np.savez(output_directory / "projections.npz", **bases)
    np.save(output_directory / "client_ids.npy", ids)
    (output_directory / "features.json").write_text(json.dumps(output_names), encoding="utf-8")
    record = {"clients": len(ids), "dimensions": len(output_names), "pca_sample": len(selected),
              "seed": seed, "official_targets_used": False, "blocks": summary,
              "seconds": perf_counter() - started, "dtype": "float16",
              "profile_input": profile_directory.name, "activity_input": activity_directory.name,
              "product_input": product_directory.name,
              "supervised_scaling": "input pretrained blocks retain the existing factor 1.3"}
    (output_directory / "manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    temp.replace(destination)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("profile", "activity", "product", "output"):
        parser.add_argument("--" + arg + "-directory", type=Path, required=True)
    parser.add_argument("--sample-users", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=42)
    compress(**vars(parser.parse_args()))
