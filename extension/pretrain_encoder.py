"""Learn a 457-dimensional encoder using a held-out window inside input/ only.

The official target directory is never read. Preparation, training and extraction
are separate stages so the expensive historical caches can be reused.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as parquet

from .build_embeddings import (DAY_NS, EVENTS, PRODUCT_EVENTS, build_embeddings,
                               hash_buckets, input_cutoff, timestamps_ns, validate_inputs)
from .target_cache import create_target_cache


SEQUENCE_DTYPE = np.dtype([("timestamp", "<i8"), ("item", "<u2"),
                           ("category", "<u2"), ("event", "u1"), ("price", "u1")])
EMPTY_TIME = np.iinfo(np.int64).min


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def temporal_boundary(cutoff: int, days: int) -> int:
    """Start of the last `days` calendar dates represented by the input cutoff."""
    return int((pd.Timestamp(cutoff, tz="UTC").normalize()
                - pd.Timedelta(days=days - 1)).value)


def update_recent(cache: np.ndarray, rows: np.ndarray, events: np.ndarray) -> None:
    """Merge the latest L events per user without retaining the full event table."""
    if len(rows) == 0:
        return
    # Tie-breakers make results independent of Arrow batch boundaries and ordering.
    order = np.lexsort((events["price"], events["category"], events["item"],
                        events["event"], -events["timestamp"], rows))
    rows, events = rows[order], events[order]
    starts = np.r_[0, np.flatnonzero(rows[1:] != rows[:-1]) + 1]
    group = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(rows)]))
    rank = np.arange(len(rows)) - starts[group]
    unique = rows[starts]
    length = cache.shape[1]
    incoming = np.zeros((len(unique), length), dtype=SEQUENCE_DTYPE)
    incoming["timestamp"] = EMPTY_TIME
    keep = rank < length
    incoming[group[keep], rank[keep]] = events[keep]
    merged = np.concatenate((cache[unique], incoming), axis=1)
    # Ascending lexicographic tie-breaks, newest timestamp first, padding last.
    times = merged["timestamp"].copy()
    times[times == EMPTY_TIME] = 0
    selected = np.lexsort((merged["price"], merged["category"], merged["item"],
                           merged["event"], -times), axis=1)[:, :length]
    cache[unique] = np.take_along_axis(merged, selected, axis=1)


def stratified_users(eligible: np.ndarray, labels: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Split users, preserving historical/future purchase strata where possible."""
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must lie between zero and one")
    generator = np.random.default_rng(seed)
    validation = np.zeros(len(eligible), dtype=bool)
    strata = eligible.astype(np.uint8) * 2 + (labels[:, 0] == 0)
    for group in np.unique(strata):
        indices = np.flatnonzero(strata == group)
        generator.shuffle(indices)
        count = min(len(indices) - 1, max(1, round(len(indices) * fraction))) if len(indices) > 1 else 0
        validation[indices[:count]] = True
    if not validation.any() or validation.all():
        raise ValueError("At least two users in a purchase stratum are needed for a validation split")
    return validation


def prepare_training_data(data_directory: str | Path, cache_directory: str | Path,
                          pseudo_days: int = 14, sequence_length: int = 32,
                          batch_size: int = 50_000, validation_fraction: float = 0.1,
                          seed: int = 42, overwrite: bool = False) -> dict:
    started = time.perf_counter()
    data_directory, cache_directory = Path(data_directory), Path(cache_directory)
    cache_directory.mkdir(parents=True, exist_ok=True)
    if (cache_directory / "manifest.json").exists() and not overwrite:
        raise FileExistsError("Prepared cache exists; reuse it with --stage train or use --overwrite")
    if pseudo_days < 1 or sequence_length < 1:
        raise ValueError("pseudo_days and sequence_length must be positive")
    ids, paths = validate_inputs(data_directory)
    client_index = pd.Index(ids)
    cutoff = input_cutoff(paths, batch_size)
    boundary = temporal_boundary(cutoff, pseudo_days)
    properties = pd.read_parquet(data_directory / "product_properties.parquet",
                                 columns=["sku", "category", "price"]).set_index("sku")
    counts = {"sku": Counter(), "category": Counter()}
    eligible = np.zeros(len(ids), dtype=bool)
    for batch in parquet.ParquetFile(paths[0]).iter_batches(batch_size=batch_size):
        frame = batch.to_pandas()
        frame = frame.loc[timestamps_ns(frame["timestamp"]) < boundary]
        counts["sku"].update(frame["sku"].value_counts().to_dict())
        counts["category"].update(frame["sku"].map(properties["category"]).value_counts().to_dict())
        rows = client_index.get_indexer(frame["client_id"])
        eligible[rows[rows >= 0]] = True
    classes = {name: [int(value) for value, _ in sorted(values.items(),
                        key=lambda pair: (-pair[1], pair[0]))[:100]] for name, values in counts.items()}
    if not classes["sku"] or not classes["category"]:
        raise ValueError("The pseudo-input window needs purchases to define propensity classes")
    prefix = build_embeddings(data_directory, cache_directory / "prefix", batch_size=batch_size,
                              mode="enhanced", cutoff_utc=pd.Timestamp(boundary - 1, tz="UTC").isoformat(),
                              overwrite=overwrite)
    full = build_embeddings(data_directory, cache_directory / "full", batch_size=batch_size,
                            mode="enhanced", popular_identifiers=prefix["popular_history_identifiers"],
                            overwrite=overwrite)
    sequences = {}
    for name in ("prefix", "full"):
        sequences[name] = np.lib.format.open_memmap(cache_directory / f"{name}_sequence.npy", mode="w+",
                            dtype=SEQUENCE_DTYPE, shape=(len(ids), sequence_length))
        sequences[name][:] = np.zeros((), dtype=SEQUENCE_DTYPE)
        sequences[name]["timestamp"] = EMPTY_TIME
    labels = np.lib.format.open_memmap(cache_directory / "pseudo_targets.npy", mode="w+",
                                      dtype=np.uint8, shape=(len(ids), 201))
    labels[:] = 0
    labels[:, 0] = 1
    for event_number, (event, path) in enumerate(zip(EVENTS, paths), 1):
        columns = ["client_id", "timestamp"] + (["sku"] if event in PRODUCT_EVENTS else
                                                  ["url"] if event == "page_visit" else [])
        for batch in parquet.ParquetFile(path).iter_batches(batch_size=batch_size, columns=columns):
            frame = batch.to_pandas()
            rows = client_index.get_indexer(frame["client_id"])
            selected = rows >= 0
            frame, rows = frame.loc[selected].copy(), rows[selected]
            if len(rows) == 0:
                continue
            times = timestamps_ns(frame["timestamp"])
            records = np.zeros(len(rows), dtype=SEQUENCE_DTYPE)
            records["timestamp"], records["event"] = times, event_number
            if event in PRODUCT_EVENTS:
                records["item"] = hash_buckets(frame["sku"].to_numpy(), 8192) + 1
                categories = frame["sku"].map(properties["category"]).fillna(0).to_numpy()
                records["category"] = hash_buckets(categories, 2048) + 1
                records["price"] = frame["sku"].map(properties["price"]).fillna(0).clip(0, 100).to_numpy()
            elif event == "page_visit":
                records["item"] = hash_buckets(frame["url"].fillna(0).to_numpy(), 8192) + 1
            update_recent(sequences["full"], rows, records)
            past = times < boundary
            update_recent(sequences["prefix"], rows[past], records[past])
            if event == "product_buy":
                future = frame.loc[(times >= boundary) & (times <= cutoff)].copy()
                future["category"] = future["sku"].map(properties["category"])
                # The same binary membership calculator as the official cache, on
                # each batch's present users only; no official target data involved.
                users = future["client_id"].unique()
                destination = client_index.get_indexer(users)
                labels[destination, 0] = 0
                for column, offset in (("category", 1), ("sku", 101)):
                    block = create_target_cache(users, future, column, np.asarray(classes[column]))
                    labels[destination, offset:offset + block.shape[1]] = np.maximum(
                        labels[destination, offset:offset + block.shape[1]], block)
        for cache in sequences.values():
            cache.flush()
        print(f"Prepared recent sequences: {event}", flush=True)
    labels.flush()
    validation = stratified_users(eligible, labels, validation_fraction, seed)
    np.save(cache_directory / "churn_eligible.npy", eligible, allow_pickle=False)
    np.save(cache_directory / "validation_users.npy", validation, allow_pickle=False)
    np.save(cache_directory / "client_ids.npy", ids, allow_pickle=False)
    write_json(cache_directory / "classes.json", classes)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "clients": len(ids),
                "aggregate_dimensions": prefix["dimensions"], "sequence_length": sequence_length,
                "pseudo_days": pseudo_days, "pseudo_input_inclusive_cutoff_utc": prefix["input_cutoff_utc"],
                "pseudo_target_start_inclusive_utc": pd.Timestamp(boundary, tz="UTC").isoformat(),
                "full_input_inclusive_cutoff_utc": full["input_cutoff_utc"],
                "pseudo_target_end_inclusive_utc": full["input_cutoff_utc"],
                "official_target_directory_read": False, "class_selection": "Pseudo-input purchases only",
                "class_counts": {name: len(values) for name, values in classes.items()},
                "churn_eligible_users": int(eligible.sum()), "validation_users": int(validation.sum()),
                "seed": seed, "sequence_order": "Latest events, deterministic chronological/tie ordering",
                "seconds": round(time.perf_counter() - started, 3)}
    write_json(cache_directory / "manifest.json", manifest)
    return manifest


def make_encoder(aggregate_dimensions: int):
    """Import torch only for the optional pretraining path."""
    import torch
    from torch import nn

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.aggregate = nn.Sequential(nn.Linear(aggregate_dimensions, 512), nn.LayerNorm(512), nn.GELU())
            self.residual = nn.Sequential(nn.Linear(512, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 512))
            self.norm = nn.LayerNorm(512)
            self.item = nn.Embedding(8193, 16, padding_idx=0)
            self.category = nn.Embedding(2049, 8, padding_idx=0)
            self.event = nn.Embedding(6, 8, padding_idx=0)
            self.sequence = nn.GRU(35, 96, batch_first=True)
            self.bottleneck = nn.Sequential(nn.Linear(608, 256), nn.LayerNorm(256), nn.GELU())
            self.heads = nn.ModuleList([nn.Linear(256, 1), nn.Linear(256, 100), nn.Linear(256, 100)])

        def forward(self, aggregates, item, category, event, numeric, lengths):
            sequence = torch.cat((self.item(item), self.category(category), self.event(event), numeric), dim=-1)
            packed = nn.utils.rnn.pack_padded_sequence(sequence, lengths.clamp_min(1).cpu(),
                                                      batch_first=True, enforce_sorted=False)
            _, hidden = self.sequence(packed)
            hidden = hidden[0] * (lengths > 0).unsqueeze(1)
            aggregate = self.aggregate(aggregates)
            aggregate = self.norm(aggregate + self.residual(aggregate))
            bottleneck = self.bottleneck(torch.cat((aggregate, hidden), dim=1))
            logits = torch.cat([head(bottleneck) for head in self.heads], dim=1)
            return bottleneck, logits

    return Encoder()


def feature_statistics(matrix: np.ndarray, indices: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(matrix.shape[1], dtype=np.float64)
    squared = total.copy()
    for start in range(0, len(indices), batch_size):
        block = matrix[indices[start:start + batch_size]].astype(np.float64)
        total += block.sum(axis=0)
        squared += np.square(block).sum(axis=0)
    mean = total / len(indices)
    scale = np.sqrt(np.maximum(squared / len(indices) - mean ** 2, 0))
    scale[scale < 1e-6] = 1
    return mean.astype(np.float32), scale.astype(np.float32)


def model_inputs(matrix, sequence, indices, cutoff, mean, scale, device):
    import torch
    records = sequence[indices]
    lengths = (records["timestamp"] != EMPTY_TIME).sum(axis=1)
    positions = np.maximum(lengths[:, None] - 1 - np.arange(records.shape[1]), 0)
    records = np.take_along_axis(records, positions, axis=1)
    present = np.arange(records.shape[1])[None, :] < lengths[:, None]
    times = np.where(present, records["timestamp"], cutoff)
    ages = np.maximum((cutoff - times) / DAY_NS, 0)
    gaps = np.zeros_like(ages)
    gaps[:, 1:] = np.maximum((times[:, 1:] - times[:, :-1]) / DAY_NS, 0)
    numeric = np.stack((np.log1p(ages) / 6, np.log1p(gaps) / 6, records["price"] / 100), axis=2)
    numeric *= present[:, :, None]
    aggregates = np.clip((matrix[indices].astype(np.float32) - mean) / scale, -10, 10)
    arrays = [aggregates] + [(records[name] * present).astype(np.int64) for name in ("item", "category", "event")]
    arrays += [numeric.astype(np.float32), lengths.astype(np.int64)]
    return [torch.from_numpy(np.ascontiguousarray(value)).to(device) for value in arrays]


def multitask_loss(logits, labels, eligible, positive_weights, class_counts):
    import torch
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, pos_weight=positive_weights, reduction="none")
    churn = (losses[:, 0] * eligible).sum() / eligible.sum().clamp_min(1)
    category = losses[:, 1:1 + class_counts["category"]].mean()
    sku = losses[:, 101:101 + class_counts["sku"]].mean()
    return churn + category + sku


def training_positive_weights(labels, eligible, indices, batch_size):
    """Estimate negative/positive ratios from training users, with a churn mask."""
    positives = np.zeros(201, dtype=np.float64)
    for start in range(0, len(indices), batch_size):
        rows = indices[start:start + batch_size]
        block = labels[rows]
        positives += block.sum(axis=0)
        positives[0] -= block[~eligible[rows], 0].sum()
    denominators = np.full(201, len(indices), dtype=np.float64)
    denominators[0] = eligible[indices].sum()
    ratios = (denominators - positives) / np.maximum(positives, 1)
    weights = np.clip(ratios, 1, 50).astype(np.float32)
    # Churn is the majority positive class in this pseudo window. Allow weights
    # below one so its negative/positive ratio is retained instead of flooring it.
    weights[0] = np.clip(ratios[0], 0.02, 50)
    return weights


def train_encoder(cache_directory: str | Path, output_directory: str | Path, epochs: int = 10,
                  batch_size: int = 1024, learning_rate: float = 0.001, patience: int = 2,
                  seed: int = 42, device: str = "cuda", overwrite: bool = False) -> dict:
    import torch
    if min(epochs, batch_size, patience) < 1:
        raise ValueError("epochs, batch_size and patience must be positive")
    started = time.perf_counter()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cache_directory, output_directory = Path(cache_directory), Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_directory / "encoder.pt"
    if checkpoint_path.exists() and not overwrite:
        raise FileExistsError("Encoder exists; use --overwrite to train it again")
    metadata = json.loads((cache_directory / "manifest.json").read_text(encoding="utf-8"))
    matrix = np.load(cache_directory / "prefix" / "embeddings.npy", mmap_mode="r")
    sequence = np.load(cache_directory / "prefix_sequence.npy", mmap_mode="r")
    labels = np.load(cache_directory / "pseudo_targets.npy", mmap_mode="r")
    eligible = np.load(cache_directory / "churn_eligible.npy")
    validation = np.load(cache_directory / "validation_users.npy")
    training_indices, validation_indices = np.flatnonzero(~validation), np.flatnonzero(validation)
    mean, scale = feature_statistics(matrix, training_indices, batch_size)
    weights = training_positive_weights(labels, eligible, training_indices, batch_size)
    positive_weights = torch.from_numpy(weights).to(device)
    model = make_encoder(matrix.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    cutoff = pd.Timestamp(metadata["pseudo_input_inclusive_cutoff_utc"]).value
    best, stale, history = float("inf"), 0, []
    generator = np.random.default_rng(seed)
    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        model.train()
        shuffled = generator.permutation(training_indices)
        train_total = 0.0
        for start in range(0, len(shuffled), batch_size):
            indices = shuffled[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            _, logits = model(*model_inputs(matrix, sequence, indices, cutoff, mean, scale, device))
            loss = multitask_loss(logits, torch.from_numpy(labels[indices].astype(np.float32)).to(device),
                                  torch.from_numpy(eligible[indices].astype(np.float32)).to(device),
                                  positive_weights, metadata["class_counts"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            train_total += float(loss.detach()) * len(indices)
        model.eval()
        validation_sums = np.zeros(3, dtype=np.float64)
        with torch.no_grad():
            for start in range(0, len(validation_indices), batch_size):
                indices = validation_indices[start:start + batch_size]
                _, logits = model(*model_inputs(matrix, sequence, indices, cutoff, mean, scale, device))
                losses = torch.nn.functional.binary_cross_entropy_with_logits(logits,
                    torch.from_numpy(labels[indices].astype(np.float32)).to(device),
                    pos_weight=positive_weights, reduction="none")
                active = torch.from_numpy(eligible[indices]).to(device)
                validation_sums += [float(losses[active, 0].sum()),
                    float(losses[:, 1:1 + metadata["class_counts"]["category"]].sum()),
                    float(losses[:, 101:101 + metadata["class_counts"]["sku"]].sum())]
        denominators_validation = [max(int(eligible[validation_indices].sum()), 1),
            len(validation_indices) * metadata["class_counts"]["category"],
            len(validation_indices) * metadata["class_counts"]["sku"]]
        value = float((validation_sums / denominators_validation).sum())
        history.append({"epoch_zero_based": epoch, "training_weighted_bce": train_total / len(training_indices),
                        "validation_weighted_bce": value, "seconds": round(time.perf_counter() - epoch_start, 3)})
        print(json.dumps(history[-1]), flush=True)
        if value < best:
            best, stale = value, 0
            torch.save({"model": model.state_dict(), "aggregate_dimensions": matrix.shape[1],
                        "mean": torch.from_numpy(mean), "scale": torch.from_numpy(scale),
                        "positive_weights": torch.from_numpy(weights), "epoch": epoch,
                        "validation_weighted_bce": value, "seed": seed}, checkpoint_path)
        else:
            stale += 1
        if stale >= patience:
            break
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "seed": seed,
              "selection": "Minimum held-out-user weighted three-head BCE; weights estimated from training users only",
              "training_users": len(training_indices), "validation_users": len(validation_indices),
              "official_target_directory_read": False, "class_counts": metadata["class_counts"],
              "best_validation_weighted_bce": best, "epochs": history, "batch_size": batch_size,
              "positive_weight_clipping": {"churn": [0.02, 50], "category": [1, 50], "sku": [1, 50]},
              "churn_positive_weight": float(weights[0]),
              "learning_rate": learning_rate, "device": device, "torch_version": torch.__version__,
              "seconds": round(time.perf_counter() - started, 3)}
    write_json(output_directory / "training.json", result)
    return result


def extract_embeddings(cache_directory: str | Path, output_directory: str | Path,
                       batch_size: int = 1024, device: str = "cuda", overwrite: bool = False) -> dict:
    import torch
    started = time.perf_counter()
    cache_directory, output_directory = Path(cache_directory), Path(output_directory)
    path = output_directory / "embeddings.npy"
    if path.exists() and not overwrite:
        raise FileExistsError("Embeddings exist; use --overwrite to replace them")
    metadata = json.loads((cache_directory / "manifest.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(output_directory / "encoder.pt", map_location="cpu", weights_only=True)
    matrix = np.load(cache_directory / "full" / "embeddings.npy", mmap_mode="r")
    sequence = np.load(cache_directory / "full_sequence.npy", mmap_mode="r")
    model = make_encoder(checkpoint["aggregate_dimensions"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    output = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(matrix), 457))
    cutoff = pd.Timestamp(metadata["full_input_inclusive_cutoff_utc"]).value
    classes = json.loads((cache_directory / "classes.json").read_text(encoding="utf-8"))
    with torch.no_grad():
        for start in range(0, len(matrix), batch_size):
            indices = np.arange(start, min(start + batch_size, len(matrix)))
            bottleneck, logits = model(*model_inputs(matrix, sequence, indices, cutoff,
                         checkpoint["mean"].numpy(), checkpoint["scale"].numpy(), device))
            logits[:, 1 + len(classes["category"]):101] = 0
            logits[:, 101 + len(classes["sku"]):201] = 0
            output[indices] = torch.cat((bottleneck, logits), dim=1).cpu().numpy().astype(np.float16)
    output.flush()
    np.save(output_directory / "client_ids.npy", np.load(cache_directory / "client_ids.npy"), allow_pickle=False)
    names = [f"bottleneck_{i}" for i in range(256)] + ["pseudo_churn_logit"]
    names += [f"pseudo_{kind}_logit_{classes[kind][i] if i < len(classes[kind]) else 'unused_' + str(i)}"
              for kind in ("category", "sku") for i in range(100)]
    write_json(output_directory / "features.json", names)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "method": "input_only_pseudo_future_multitask_residual_mlp_gru",
                "clients": len(matrix), "dimensions": 457, "dtype": "float16", "standardized": False,
                "bottleneck_dimensions": 256, "prediction_logit_dimensions": 201,
                "official_target_directory_read": False, "pseudo_window": metadata,
                "checkpoint_epoch_zero_based": checkpoint["epoch"], "seed": checkpoint["seed"],
                "checkpoint_validation_weighted_bce": checkpoint["validation_weighted_bce"],
                "full_input_inclusive_cutoff_utc": metadata["full_input_inclusive_cutoff_utc"],
                "classes": classes, "seconds": round(time.perf_counter() - started, 3),
                "torch_version": torch.__version__}
    write_json(output_directory / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, required=True)
    parser.add_argument("--cache-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "train", "extract", "all"), default="all")
    parser.add_argument("--pseudo-days", type=int, default=14)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--prepare-batch-size", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    if arguments.stage in ("prepare", "all"):
        prepare_training_data(arguments.data_directory, arguments.cache_directory, arguments.pseudo_days,
                              arguments.sequence_length, arguments.prepare_batch_size,
                              arguments.validation_fraction, arguments.seed, arguments.overwrite)
    if arguments.stage in ("train", "all"):
        train_encoder(arguments.cache_directory, arguments.output_directory, arguments.epochs,
                      arguments.batch_size, arguments.learning_rate, arguments.patience,
                      arguments.seed, arguments.device, arguments.overwrite)
    if arguments.stage in ("extract", "all"):
        print(json.dumps(extract_embeddings(arguments.cache_directory, arguments.output_directory,
                         arguments.batch_size, arguments.device, arguments.overwrite), indent=2))


if __name__ == "__main__":
    main()
