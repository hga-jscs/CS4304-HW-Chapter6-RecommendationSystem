"""Daily activity windows and historical repeat-purchase scores for Synerise."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from .build_embeddings import EVENTS, DAY_NS, input_cutoff, timestamps_ns, validate_inputs

WINDOWS = (7, 14, 28, 56, 112)
ANCHORS = (98, 84, 70, 56, 42, 28)
WIDTH = 2 * len(WINDOWS) + 5
NAMES = [name for days in WINDOWS for name in (f"events_{days}d", f"active_days_{days}d")] + [
    "recency", "first_activity_age", "last_gap", "previous_gap", "mean_gap"]


def daily_summary(counts):
    """Rows are users and columns are recent to old, beginning at the input cutoff."""
    features = np.empty((len(counts), WIDTH), np.float32)
    present = counts > 0
    for i, days in enumerate(WINDOWS):
        features[:, 2*i] = counts[:, :days].sum(axis=1)
        features[:, 2*i+1] = present[:, :days].sum(axis=1)
    # Missing observed ages/gaps have a distinct sentinel beyond the 112-day window.
    horizon = counts.shape[1]
    active = present.sum(axis=1)
    recent = np.where(active, present.argmax(axis=1), horizon)
    oldest = np.where(active, horizon - 1 - present[:, ::-1].argmax(axis=1), horizon)
    remaining = present.copy()
    rows = np.arange(len(counts))
    last = recent
    features[:, -5] = recent
    features[:, -4] = oldest
    for k in (2, 3):
        remaining[rows, np.minimum(last, horizon - 1)] = False
        previous = remaining.argmax(axis=1)
        features[:, k-5] = np.where(active >= k, previous - last, horizon)
        last = previous
    features[:, -1] = np.where(active >= 2, (oldest - recent)/np.maximum(active - 1, 1), horizon)
    return np.log1p(features)


def event_days(path, clients, last_day):
    import scipy.sparse as sp
    # A sparse client-day table keeps individual URL visits out of the in-memory history.
    days = max(ANCHORS) + max(WINDOWS)
    counts = sp.csr_matrix((len(clients), days), dtype=np.int32)
    chunks = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=300000, columns=["client_id", "timestamp"]):
        frame = batch.to_pandas()
        users = clients.get_indexer(frame.client_id)
        age = last_day - timestamps_ns(frame.timestamp) // DAY_NS
        keep = (users >= 0) & (age >= 0) & (age < days)
        chunks.append((users[keep].astype(np.int32), age[keep].astype(np.int32)))
        if len(chunks) >= 10:
            counts += pack_days(chunks, counts.shape)
            chunks.clear()
    if chunks:
        counts += pack_days(chunks, counts.shape)
    return counts


def pack_days(chunks, shape):
    from scipy.sparse import csr_matrix
    rows = np.concatenate([entry[0] for entry in chunks])
    days = np.concatenate([entry[1] for entry in chunks])
    return csr_matrix((np.ones(len(rows), np.int32), (rows, days)), shape=shape)


def outcomes(days, anchor):
    # age == anchor still belongs to the prefix. Future days 1..14 and 15..28 are disjoint.
    return [np.asarray(days[:, anchor-end:anchor-start].sum(axis=1) > 0, dtype=np.uint8).ravel()
            for start, end in ((0, 14), (14, 28))]


def build(data_directory, output_directory, sample_users=280000, seed=42):
    started = perf_counter()
    data, output = Path(data_directory), Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"embeddings.npy").exists():
        raise FileExistsError(output/"embeddings.npy")
    ids, paths = validate_inputs(data)
    clients = pd.Index(ids)
    cutoff = input_cutoff(paths, 100000) // DAY_NS
    rng = np.random.default_rng(seed)
    sampled = np.sort(rng.choice(len(ids), min(len(ids), sample_users), replace=False))
    width = len(EVENTS)*WIDTH
    train = np.empty((len(ANCHORS), len(sampled), width), np.float32)
    full = np.empty((len(ids), width+2), np.float32)
    for index, path in enumerate(paths):
        counts = event_days(path, clients, cutoff)
        target_columns = slice(index*WIDTH, (index+1)*WIDTH)
        for begin in range(0, len(ids), 8192):
            stop = min(begin+8192, len(ids))
            days = counts[begin:stop].toarray()
            full[begin:stop, target_columns] = daily_summary(days[:, :max(WINDOWS)])
        for begin in range(0, len(sampled), 8192):
            rows = sampled[begin:begin+8192]
            days = counts[rows].toarray()
            for i, anchor in enumerate(ANCHORS):
                train[i, begin:begin+len(rows), target_columns] = daily_summary(days[:, anchor:anchor+max(WINDOWS)])
        if index == 0:
            targets = np.array([outcomes(counts[sampled], anchor) for anchor in ANCHORS])
        del counts
        print(f"Prepared {EVENTS[index]}", flush=True)
    # Only buyers observed in the preceding 112 days enter the repeat-purchase models.
    eligible = train[:, :, 2*len(WINDOWS)-2] > 0
    xtrain = train[eligible]
    xfull = full[:, :width]
    metrics = []
    with threadpool_limits(limits=4):
        for horizon in range(2):
            labels = targets[:, horizon, :]
            # Hold out the latest anchor. A 28-day embargo keeps fitting labels within its prefix.
            selection = np.asarray(ANCHORS) >= ANCHORS[-1]+28
            model = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=23, learning_rate=0.075,
                                                   min_samples_leaf=100, early_stopping=False, random_state=seed)
            model.fit(train[selection][eligible[selection]], labels[selection][eligible[selection]])
            auc = roc_auc_score(labels[-1][eligible[-1]], model.predict_proba(train[-1][eligible[-1]])[:, 1])
            model.fit(xtrain, labels[eligible])
            for start in range(0,len(ids),20000):
                block = xfull[start:start+20000]
                scores = model.decision_function(block)
                full[start:start+len(block), width+horizon] = np.where(
                    block[:, 2*len(WINDOWS)-2] > 0, scores, 0)
            metrics.append(float(auc))
    # Same normalization for every submitted user, using the full historical input only.
    center = full.mean(axis=0, dtype=np.float64)
    scale = full.std(axis=0, dtype=np.float64)
    scale[scale < 1e-6] = 1
    path = output/"activity.partial.npy"
    matrix = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=full.shape)
    for start in range(0,len(ids),20000):
        part = np.clip((full[start:start+20000] - center)/scale,-8,8)
        if not np.isfinite(part).all():
            raise ValueError("Nonfinite activity features")
        matrix[start:start+len(part)] = part
    del matrix
    np.save(output/"client_ids.npy", ids, allow_pickle=False)
    names = [f"{event}.{name}" for event in EVENTS for name in NAMES]+["repurchase_days_1_14", "repurchase_days_15_28"]
    (output/"features.json").write_text(json.dumps(names,indent=2))
    path.replace(output/"embeddings.npy")
    record = dict(clients=len(ids), dimensions=len(names), seed=seed, cutoff_day=int(cutoff),
                  anchors_days_before_end=ANCHORS, training_users_sampled=len(sampled),
                  training_buyer_windows=len(xtrain), latest_historical_window_auc=metrics,
                  official_targets_used=False, normalization="all historical profiles, column z-score clipped at 8",
                  seconds=round(perf_counter()-started,3))
    (output/"manifest.json").write_text(json.dumps(record,indent=2))
    print(record,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--sample-users", type=int, default=280000)
    parser.add_argument("--seed", type=int, default=42)
    build(**vars(parser.parse_args()))
