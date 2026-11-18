"""Combine historical near and delayed purchase heads in the same frozen profile coordinates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from .purchase_targets import (
    fit_heads,
    normalized_latent,
    profile_batches,
    purchase_labels,
    sample_rows,
)


def blend_heads(near_weight, near_bias, late_weight, late_bias, support):
    """Average log odds, shrinking the delayed contribution for poorly supported classes."""
    fraction = (0.5 * support / (support + 15.0)).astype(np.float32)
    weight = near_weight + fraction * (late_weight - near_weight)
    bias = near_bias + fraction * (late_bias - near_bias)
    return weight, bias


def macro_comparison(x, near_weight, near_bias, weight, bias, matrices):
    """Compare both models on identical held-out users, on each fortnight separately."""
    totals = {h: [] for h in matrices}
    for start in range(0, weight.shape[1], 16):
        cols = slice(start, start + 16)
        near = x @ near_weight[:, cols] + near_bias[cols]
        both = x @ weight[:, cols] + bias[cols]
        for horizon, labels in matrices.items():
            truth = labels[:, cols].toarray()
            keep = np.flatnonzero((truth.sum(axis=0) > 0) & (truth.sum(axis=0) < len(x)))
            for j in keep:
                totals[horizon].append(
                    [roc_auc_score(truth[:, j], near[:, j]), roc_auc_score(truth[:, j], both[:, j])]
                )
    result = {}
    for horizon, values in totals.items():
        result[horizon] = {"classes_with_both_outcomes": len(values)}
        if values:
            old, new = np.asarray(values).mean(axis=0)
            result[horizon].update(near_only=float(old), joint=float(new), difference=float(new - old))
    return result


def build(data, cache, baseline, output, batch_size=4096, seed=42):
    started = perf_counter()
    if (output / "embeddings.npy").exists():
        raise FileExistsError(output / "embeddings.npy")
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    source = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
    windows = sorted(
        (w for w in metadata["windows"] if w["forecast_gap_days"] == 0),
        key=lambda w: w["target_start_inclusive_utc"],
    )
    (late,) = [w for w in metadata["windows"] if w["forecast_gap_days"] == 14]
    if (
        len(windows) != 2
        or late["feature_window"] != windows[0]["feature_window"]
        or late["input_cutoff_utc"] != windows[0]["input_cutoff_utc"]
        or late["target_window"] != windows[1]["target_window"]
        or any(
            late[k] != windows[1][k] for k in ("target_start_inclusive_utc", "target_end_inclusive_utc")
        )
        or pd.Timestamp(windows[0]["target_end_inclusive_utc"])
        >= pd.Timestamp(late["target_start_inclusive_utc"])
    ):
        raise ValueError(
            "Expected two distinct target fortnights and the earlier profile paired with the later fortnight"
        )
    ids = np.load(data / "input/relevant_clients.npy")
    early = cache / late["feature_window"]
    for directory in (cache, early, cache / "full", baseline):
        if not np.array_equal(np.load(directory / "client_ids.npy"), ids):
            raise ValueError(f"User order differs in {directory}")
    heldout = np.load(cache / "validation_users.npy")
    if heldout.dtype != bool or heldout.shape != ids.shape:
        raise ValueError("Invalid held-out user mask")
    vocab, labels, lines = purchase_labels(
        data, ids, heldout, windows, source["requested_caps"], source["min_purchase_lines"]
    )
    with np.load(baseline / "heads.npz") as archive:
        params = {k: archive[k] for k in archive.files}
    if any(not np.array_equal(vocab[f], params[f]) for f in vocab):
        raise ValueError("Historical vocabulary differs from the fitted near head")
    held_rows = np.flatnonzero(heldout)
    rng = np.random.default_rng(seed)
    background = sample_rows(np.flatnonzero(~heldout), 40000, rng)
    positive = np.logical_or.reduce([np.diff(m[1].indptr) > 0 for m in labels.values()])
    rows = np.union1d(background, np.flatnonzero(positive & ~heldout))
    background_in_pool = np.searchsorted(rows, background)
    diagnostics = {}
    for field in vocab:
        prefix = field + "_"

        def latent(raw):
            return normalized_latent(
                raw,
                params["mean"],
                params["scale"],
                params[prefix + "basis"],
                params[prefix + "center"],
                params[prefix + "deviation"],
            )

        design = np.concatenate(
            [latent(raw) for _, raw in profile_batches(early / "embeddings.npy", batch_size, rows)]
        )
        train_y = labels[field][1][rows].tocsc()
        support = np.diff(train_y.indptr)
        supported = support > 0
        nw, nb = params[prefix + "weight"], params[prefix + "bias"]
        lw, lb = nw.copy(), nb.copy()
        if supported.any():
            lw[:, supported], lb[supported] = fit_heads(
                design, train_y[:, supported], background_in_pool, np.random.default_rng(seed)
            )
        weight, bias = blend_heads(nw, nb, lw, lb, support)
        del design
        validation_x = np.concatenate(
            [latent(raw) for _, raw in profile_batches(early / "embeddings.npy", batch_size, held_rows)]
        )
        diagnostic = macro_comparison(
            validation_x,
            nw,
            nb,
            weight,
            bias,
            {
                name: matrix[held_rows].tocsc()
                for name, matrix in zip(("days_1_14", "days_15_28"), labels[field])
            },
        )
        diagnostic.update(
            delayed_classes_with_training_positives=int(supported.sum()), heldout_users=len(held_rows)
        )
        diagnostics[field] = diagnostic
        params[prefix + "weight"], params[prefix + "bias"] = weight, bias
        params[prefix + "delayed_support"] = support
        print(field, diagnostic, flush=True)

    columns = json.loads((baseline / "features.json").read_text(encoding="utf-8"))
    partial = output / "embeddings.partial.npy"
    vectors = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.float16, shape=(len(ids), len(columns))
    )
    for rr, raw in profile_batches(cache / "full/embeddings.npy", batch_size):
        pieces = []
        for field in vocab:
            b = {
                k: params[field + "_" + k]
                for k in (
                    "basis",
                    "center",
                    "deviation",
                    "weight",
                    "bias",
                    "probability_mean",
                    "axes",
                    "projected_scale",
                )
            }
            z = normalized_latent(
                raw, params["mean"], params["scale"], b["basis"], b["center"], b["deviation"]
            )
            p = expit(z @ b["weight"] + b["bias"])
            pieces.append(
                np.clip(((p - b["probability_mean"]) @ b["axes"]) / b["projected_scale"], -6, 6)
            )
        values = np.concatenate(pieces, axis=1)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite purchase scores")
        vectors[rr[0] : rr[-1] + 1] = values
    del vectors
    np.save(output / "client_ids.npy", ids)
    np.savez(output / "heads.npz", **params)
    (output / "features.json").write_text(json.dumps(columns), encoding="utf-8")
    record = dict(
        method="Historical near and delayed balanced logistic heads blended in log odds; frozen near vocabulary, basis, probability PCA, centering, clipping and scales.",
        delayed_weight="0.5 * training_positive_users / (training_positive_users + 15)",
        near_source_sha256=hashlib.sha256((baseline / "heads.npz").read_bytes()).hexdigest(),
        delayed_window={
            k: late[k]
            for k in ("input_cutoff_utc", "target_start_inclusive_utc", "target_end_inclusive_utc")
        },
        vocabulary_counted_purchase_lines_once=lines,
        historical_classes={k: len(v) for k, v in vocab.items()},
        clients=len(ids),
        dimensions=len(columns),
        diagnostics=diagnostics,
        validation="Held-out users at one shared historical anchor, not an independent time split; the near heads use other users from both target periods.",
        official_target_directory_read=False,
        seconds=round(perf_counter() - started, 3),
        seed=seed,
    )
    (output / "manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    partial.replace(output / "embeddings.npy")
    print(record, flush=True)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "cache", "baseline", "output"):
        parser.add_argument("--" + name + "-directory", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    with threadpool_limits(limits=args.threads):
        build(args.data_directory, args.cache_directory, args.baseline_directory, args.output_directory)
