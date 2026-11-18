"""Test whether event-day activity adds repeat-purchase signal to purchase priors.

The paired comparison changes only the 75 activity columns. The official target
directory is never read. Four horizon-specific scores are fitted on all eligible
historical training snapshots for use in downstream evaluation.
"""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.parquet as parquet
from scipy import sparse
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from .activity_windows import WINDOWS, WIDTH, daily_summary, pack_days
from .build_embeddings import DAY_NS, EVENTS, input_cutoff, validate_inputs
from .repeat_purchases import (
    BASE_NAMES,
    POST_NAMES,
    PRIOR_NAMES,
    SCORE_NAMES,
    HORIZONS,
    Snapshot,
    audience_folds,
    ensemble_trees,
    event_batches,
    future_labels,
    odds_scores,
)

AGES = (70, 56, 42, 28)
PURCHASE_WIDTH = len(BASE_NAMES + POST_NAMES + PRIOR_NAMES)
TOTAL_WIDTH = PURCHASE_WIDTH + len(EVENTS) * WIDTH


def purchases_and_catalog(data, buy_path):
    catalog = (
        parquet.read_table(
            data / "product_properties.parquet", columns=["sku", "category"], use_threads=False
        )
        .to_pandas()
        .set_index("sku")
        .category
    )
    buys = pd.concat(
        [
            pd.DataFrame({"user": user, "time": time, "sku": sku})
            for user, time, sku in event_batches(buy_path, product=True)
        ],
        ignore_index=True,
    )
    buys["day"] = (buys.time // DAY_NS).astype(np.int32)
    buys["category"] = buys.sku.map(catalog).fillna(-1).astype(np.int32)
    return buys, catalog


def activity_at_anchor(counts, rows, age):
    """Read a bounded CSR slice; exclude every calendar day later than the anchor."""
    return daily_summary(counts[rows, age : age + max(WINDOWS)].toarray())


def snapshots(paths, ids, cutoff, ages, buys, catalog, stage):
    first_day = int(buys.day.min())
    latest_dates = buys.loc[buys.day <= cutoff - min(ages), "user"].unique()
    clients = pd.Index(ids[pd.Index(ids).isin(latest_dates)])
    result = []
    accumulators = []
    index_maps = []
    for age in ages:
        purchased = buys.loc[buys.day <= cutoff - age, "user"].unique()
        rows = np.flatnonzero(clients.isin(purchased))
        selected_ids = clients.to_numpy()[rows]
        snap = Snapshot(selected_ids, cutoff - age, buys, first_day)
        features = np.lib.format.open_memmap(
            stage / f"features_{age}.npy", mode="w+", dtype=np.float32, shape=(len(rows), TOTAL_WIDTH)
        )
        result.append(
            dict(
                age=age,
                ids=selected_ids,
                rows=rows,
                days=snap.purchase_days,
                features=features,
                labels=future_labels(buys, snap) if age else None,
            )
        )
        positions = np.full(len(clients), -1, dtype=np.int32)
        positions[rows] = np.arange(len(rows))
        index_maps.append(positions)
        accumulators.append(snap)
    day_width = max(ages) + max(WINDOWS)
    for event_number, (event, path) in enumerate(zip(EVENTS, paths)):
        counts = sparse.csr_matrix((len(clients), day_width), dtype=np.int32)
        chunks = []
        item_event = event in ("product_buy", "add_to_cart")
        for batch in event_batches(path, product=item_event):
            users, times = batch[:2]
            all_rows = clients.get_indexer(users)
            eligible = all_rows >= 0
            rows = all_rows[eligible]
            times, users = times[eligible], users[eligible]
            if not len(rows):
                continue
            age_days = cutoff - times // DAY_NS
            in_range = (age_days >= min(ages)) & (age_days < day_width)
            chunks.append((rows[in_range].astype(np.int32), age_days[in_range].astype(np.int32)))
            if len(chunks) >= 12:
                counts += pack_days(chunks, counts.shape)
                chunks.clear()
            sku = batch[2][eligible] if item_event else None
            category = catalog.reindex(sku, fill_value=-1).to_numpy() if item_event else None
            earliest = times.min() // DAY_NS
            for snap, positions in zip(accumulators, index_maps):
                if earliest <= snap.cutoff:
                    snap.add_events(event, users, times, sku, category, positions[rows])
        if chunks:
            counts += pack_days(chunks, counts.shape)
        start = PURCHASE_WIDTH + event_number * WIDTH
        for profile in result:
            for offset in range(0, len(profile["rows"]), 4096):
                selected = profile["rows"][offset : offset + 4096]
                profile["features"][offset : offset + len(selected), start : start + WIDTH] = (
                    activity_at_anchor(counts, selected, profile["age"])
                )
        del counts
        print(f"Prepared {event} for anchors {ages}", flush=True)
    for snap, profile in zip(accumulators, result):
        profile["features"][:, :PURCHASE_WIDTH] = snap.finish()
        profile["features"].flush()
    return result


def compare(features, labels, validation, baseline_directory, seed):
    days = validation["days"]
    y = validation["labels"][:, -1]
    kept = audience_folds(validation["ids"]) == 0
    x = np.asarray(validation["features"][kept])
    y, one_day = y[kept], days[kept] == 1
    predictions, scores = {}, {}
    for name, width in (("purchase_priors", PURCHASE_WIDTH), ("with_activity", TOTAL_WIDTH)):
        models = ensemble_trees(seed)
        for model in models:
            model.fit(features[:, :width], labels)
        preds = odds_scores(models, x[:, :width])[2]
        predictions[name] = preds
        scores[name] = dict(
            all=float(roc_auc_score(y, preds)), one_day=float(roc_auc_score(y[one_day], preds[one_day]))
        )
    if baseline_directory is not None:
        report = json.loads((baseline_directory / "manifest.json").read_text(encoding="utf-8-sig"))
        old = next(
            v
            for v in report["historical_validation"]
            if v["model"] == "mean log-odds" and v["days"] == [14, 28]
        )
        if (
            len(y) != old["buyers"]
            or int(one_day.sum()) != old["single_day_buyers"]
            or int(y.sum()) != old["positives"]
            or int(y[one_day].sum()) != old["single_day_positives"]
            or abs(scores["purchase_priors"]["all"] - old["auc"]) > 1e-8
            or abs(scores["purchase_priors"]["one_day"] - old["single_day_auc"]) > 1e-8
        ):
            raise ValueError("Paired historical baseline did not reproduce the supplied run")
    differences = {}
    rng = np.random.default_rng(seed)
    for name, group in (("all", np.ones(len(y), dtype=bool)), ("one_day", one_day)):
        first, second = predictions["purchase_priors"][group], predictions["with_activity"][group]
        truth = y[group]
        pos, neg = np.flatnonzero(truth), np.flatnonzero(~truth)
        samples = []
        # Resample users within each class, using identical draws for both models.
        for _ in range(400):
            rows = np.concatenate((rng.choice(pos, len(pos)), rng.choice(neg, len(neg))))
            samples.append(
                roc_auc_score(truth[rows], second[rows]) - roc_auc_score(truth[rows], first[rows])
            )
        differences[name] = dict(
            gain=scores["with_activity"][name] - scores["purchase_priors"][name],
            paired_bootstrap_95pct=np.quantile(samples, [0.025, 0.975]).tolist(),
        )
    return dict(
        horizon="days 15-28",
        train_rows=len(labels),
        validation_buyers=len(y),
        validation_repeat_buyers=int(y.sum()),
        one_day_buyers=int(one_day.sum()),
        one_day_repeat_buyers=int(y[one_day].sum()),
        scores=scores,
        differences=differences,
        bootstrap_replicates=400,
    )


def save_scores(historical, current, ids, output, seed):
    x = np.concatenate([p["features"] for p in historical])
    y = np.concatenate([p["labels"] for p in historical])
    rows = pd.Index(ids).get_indexer(current["ids"])
    logits = np.zeros((len(ids), len(HORIZONS)), np.float32)
    for k in range(len(HORIZONS)):
        models = ensemble_trees(seed)
        for model in models:
            model.fit(x, y[:, k])
        for start in range(0, len(rows), 8192):
            take = slice(start, start + 8192)
            logits[rows[take], k] = odds_scores(models, current["features"][take])[2]
    for k in range(len(HORIZONS)):
        vector = logits[:, k]
        mean, std = vector.mean(dtype=np.float64), vector.std(dtype=np.float64)
        logits[:, k] = np.clip((vector - mean) / max(std, 1e-6), -8, 8)
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite scores")
    np.save(output / "client_ids.npy", ids)
    (output / "features.json").write_text(json.dumps(SCORE_NAMES), encoding="utf-8")
    np.save(output / "embeddings.partial.npy", logits.astype(np.float16))
    return len(x)


def build(data_directory, output_directory, baseline_directory=None, sample_users=200_000, seed=42):
    data, output = Path(data_directory), Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "embeddings.npy").exists() or (output / "manifest.json").exists():
        raise FileExistsError(output)
    started = perf_counter()
    ids, paths = validate_inputs(data)
    cutoff = input_cutoff(paths, 160000) // DAY_NS
    buys, catalog = purchases_and_catalog(data, paths[EVENTS.index("product_buy")])
    rows = np.sort(np.random.default_rng(seed).choice(len(ids), min(sample_users, len(ids)), replace=False))
    with TemporaryDirectory(dir=output) as temp, threadpool_limits(limits=2):
        stage = Path(temp)
        historical = snapshots(paths, ids[rows], cutoff, AGES, buys, catalog, stage)
        train = [p for p in historical if p["age"] >= AGES[-1] + 28]
        masks = [audience_folds(p["ids"]) != 0 for p in train]
        train_x = np.concatenate([p["features"][mask] for p, mask in zip(train, masks)])
        train_y = np.concatenate([p["labels"][mask, -1] for p, mask in zip(train, masks)])
        check = compare(train_x, train_y, historical[-1], baseline_directory, seed)
        del train_x, train_y
        record = dict(
            method="Identical historical LightGBM and CatBoost ensemble; purchase prior and post-purchase features with five event streams of activity summaries",
            original_purchase_columns=PURCHASE_WIDTH,
            added_activity_columns=TOTAL_WIDTH - PURCHASE_WIDTH,
            sample_users=len(rows),
            seed=seed,
            historical_anchor_ages=list(AGES),
            cutoff_day=int(cutoff),
            validation="Unseen fold-zero buyers at the final historical anchor; training labels end by that anchor",
            historical_ablation=check,
            official_target_directory_read=False,
        )
        print(json.dumps(check), flush=True)
        (output / "historical_comparison.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        current = snapshots(paths, ids, cutoff, (0,), buys, catalog, stage)[0]
        record["final_training_buyer_windows"] = save_scores(historical, current, ids, output, seed)
        record.update(
            dimensions=len(HORIZONS),
            users=len(ids),
            dtype="float16",
            score_names=SCORE_NAMES,
            normalization="Full-history population z-score, clipped to +/-8; nonbuyers have raw zero before scoring normalization",
        )
        del current["features"]
        for p in historical:
            del p["features"]
    record["seconds"] = round(perf_counter() - started, 3)
    (output / "manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    (output / "embeddings.partial.npy").replace(output / "embeddings.npy")
    return record


def replace_scores(baseline_directory, scores_directory, output_directory):
    """Copy the previously evaluated 31-column block, substituting four activity-informed repeat scores."""
    baseline_directory = Path(baseline_directory)
    scores_directory = Path(scores_directory)
    output_directory = Path(output_directory)
    ids = np.load(baseline_directory / "client_ids.npy", allow_pickle=False)
    score_ids = np.load(scores_directory / "client_ids.npy", allow_pickle=False)
    original_names = json.loads((baseline_directory / "features.json").read_text(encoding="utf-8-sig"))
    score_names = json.loads((scores_directory / "features.json").read_text(encoding="utf-8-sig"))
    if not np.array_equal(ids, score_ids):
        raise ValueError("Client IDs must have identical order for in-place column substitution")
    if original_names != BASE_NAMES + POST_NAMES + PRIOR_NAMES + SCORE_NAMES or score_names != SCORE_NAMES:
        raise ValueError("Expected purchase history columns followed by the four named repeat scores")
    original = np.load(baseline_directory / "embeddings.npy", mmap_mode="r")
    scores = np.load(scores_directory / "embeddings.npy", mmap_mode="r")
    if (
        original.shape != (len(ids), len(original_names))
        or scores.shape != (len(ids), len(score_names))
        or original.dtype != np.float16
        or scores.dtype != np.float16
    ):
        raise ValueError("Unexpected data types or shapes in score replacement")
    output_directory.mkdir(parents=True, exist_ok=True)
    destination = output_directory / "embeddings.npy"
    if destination.exists():
        raise FileExistsError(destination)
    with TemporaryDirectory(dir=output_directory) as scratch:
        stage = Path(scratch)
        result = np.lib.format.open_memmap(
            stage / "embeddings.npy", mode="w+", dtype=np.float16, shape=original.shape
        )
        for i in range(0, len(ids), 8192):
            batch = slice(i, i + 8192)
            result[batch, :PURCHASE_WIDTH] = original[batch, :PURCHASE_WIDTH]
            result[batch, PURCHASE_WIDTH:] = scores[batch]
        result.flush()
        del result
        saved = np.load(stage / "embeddings.npy", mmap_mode="r")
        for i in range(0, len(ids), 8192):
            batch = slice(i, i + 8192)
            if not np.array_equal(
                saved[batch, :PURCHASE_WIDTH], original[batch, :PURCHASE_WIDTH]
            ) or not np.array_equal(saved[batch, PURCHASE_WIDTH:], scores[batch]):
                raise ValueError("Full-file check of preserved and replaced columns failed")
        del saved
        np.save(stage / "client_ids.npy", ids, allow_pickle=False)
        (stage / "features.json").write_text(json.dumps(original_names, indent=2), encoding="utf-8")
        record = dict(
            clients=len(ids),
            dimensions=len(original_names),
            dtype="float16",
            unchanged_columns=PURCHASE_WIDTH,
            replaced_columns=SCORE_NAMES,
            previous_directory=baseline_directory.name,
            new_scores_directory=scores_directory.name,
            identical_user_order=True,
            all_columns_checked_against_sources=True,
        )
        (stage / "manifest.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        for filename in ("client_ids.npy", "features.json", "embeddings.npy", "manifest.json"):
            (stage / filename).replace(output_directory / filename)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    train = modes.add_parser("build")
    train.add_argument("--data-directory", type=Path, required=True)
    train.add_argument("--output-directory", type=Path, required=True)
    train.add_argument("--baseline-directory", type=Path)
    train.add_argument("--sample-users", type=int, default=200000)
    train.add_argument("--seed", type=int, default=42)
    substitution = modes.add_parser("replace")
    substitution.add_argument("--baseline-directory", type=Path, required=True)
    substitution.add_argument("--scores-directory", type=Path, required=True)
    substitution.add_argument("--output-directory", type=Path, required=True)
    args = vars(parser.parse_args())
    mode = args.pop("mode")
    action = build if mode == "build" else replace_scores
    print(json.dumps(action(**args), indent=2), flush=True)
