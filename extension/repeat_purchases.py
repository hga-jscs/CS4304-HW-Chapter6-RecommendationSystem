"""Describe purchase continuation and transfer historical repeat rates across customers."""

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

from .build_embeddings import DAY_NS, EVENTS, input_cutoff, validate_inputs

FOLDS = 5
SMOOTHING = 20
POST_EVENTS = ("page_visit", "add_to_cart", "search_query")
HORIZONS = ((0, 7), (0, 14), (0, 28), (14, 28))
BASE_NAMES = ["has_purchase", "log_purchase_days", "log_days_since_purchase"]
POST_NAMES = [f"log_{event}_after_last_buy" for event in POST_EVENTS] + [
    "log_active_days_after_last_buy",
    "active_day_share_after_last_buy",
    "latest_event_buy_only",
    "latest_event_buy_tied_with_other",
    "log_days_to_last_other_event",
]
PRIOR_NAMES = [
    f"{source}_{entity}_{outcome}_{h}d"
    for source in ("buy", "cart")
    for entity in ("sku", "category")
    for h in (14, 28)
    for outcome in ("same_entity_repeat", "any_purchase")
]
SCORE_NAMES = [f"historical_repeat_log_odds_days_{a+1}_{b}" for a, b in HORIZONS]


def event_batches(path, product=False):
    columns = ["client_id", "timestamp"] + (["sku"] if product else [])
    for batch in pq.ParquetFile(path).iter_batches(batch_size=160000, columns=columns, use_threads=False):
        users = batch.column(0).to_numpy()
        times = batch.column(1).to_numpy().astype("datetime64[ns]").view(np.int64)
        if product:
            yield users, times, batch.column(2).to_numpy()
        else:
            yield users, times


def audience_folds(users):
    # With randomly assigned customer IDs this gives stable folds across all historical snapshots.
    return np.remainder(users, FOLDS).astype(np.int8)


def prior_tables(buys, cutoff, first_day):
    """Each eligible buyer-entity pair has one vote per anchor, regardless of basket size."""
    tables = {}
    for entity in ("sku", "category"):
        for horizon in (14, 28):
            samples = []
            for anchor in (cutoff - horizon, cutoff - 2 * horizon):
                if anchor - 27 < first_day:
                    continue
                history = buys.loc[buys.day.between(anchor - 27, anchor), ["user", entity]].drop_duplicates()
                future = buys.loc[buys.day.between(anchor + 1, anchor + horizon), ["user", entity]]
                history = history.loc[history[entity] >= 0].copy()
                if history.empty:
                    continue
                future_entities = future.drop_duplicates().assign(same=1)
                joined = history.merge(future_entities, on=["user", entity], how="left", sort=False)
                joined["same"] = joined.same.fillna(0).astype(np.uint8)
                joined["any"] = joined.user.isin(future.user.unique()).astype(np.uint8)
                samples.append(joined[[entity, "user", "same", "any"]])
            if not samples:
                # No completely observed opportunity at this anchor; no known item-level rate.
                tables[(entity, horizon)] = (
                    pd.Index([], dtype=np.int64),
                    np.zeros((FOLDS, 1, 2), np.float32),
                )
                continue
            samples = pd.concat(samples, ignore_index=True)
            codes, levels = pd.factorize(samples[entity], sort=False)
            folds = audience_folds(samples.user.to_numpy())
            width = len(levels)
            cells = folds.astype(np.int64) * width + codes
            counts = np.bincount(cells, minlength=FOLDS * width).reshape(FOLDS, width)
            others = counts.sum(axis=0) - counts
            values = np.empty((FOLDS, width + 1, 2), np.float32)
            for output, outcome in enumerate(("same", "any")):
                positives = np.bincount(cells, weights=samples[outcome].to_numpy(), minlength=FOLDS * width)
                positives = positives.reshape(FOLDS, width)
                positives = positives.sum(axis=0) - positives
                total = others.sum(axis=1)
                prior = np.divide(positives.sum(axis=1), total, out=np.zeros(FOLDS), where=total > 0)
                values[:, -1, output] = prior
                values[:, :-1, output] = (positives + SMOOTHING * prior[:, None]) / (others + SMOOTHING)
            tables[(entity, horizon)] = (pd.Index(levels), values)
    return tables


class Snapshot:
    def __init__(self, ids, cutoff, buys, first_day):
        self.clients = pd.Index(ids)
        self.cutoff = cutoff
        self.first_day = first_day
        cutoff_rows = buys.loc[buys.day <= cutoff, ["user", "time", "day"]]
        self.last_buy = cutoff_rows.groupby("user").time.max().reindex(ids, fill_value=-1).to_numpy()
        self.purchase_days = (
            cutoff_rows.groupby("user").day.nunique().reindex(ids, fill_value=0).to_numpy(np.int32)
        )
        self.latest_other = np.full(len(ids), -1, np.int64)
        self.counts = np.zeros((len(ids), len(POST_EVENTS)), np.int32)
        self.day_bits = np.zeros((len(ids), max(1, (cutoff - first_day + 64) // 64)), np.uint64)
        self.priors = prior_tables(buys, cutoff, first_day)
        self.sums = np.zeros((2, len(ids), 8), np.float32)
        self.weights = np.zeros((2, len(ids)), np.float32)

    def add_events(self, kind, users, times, skus=None, categories=None, index=None):
        if index is None:
            index = self.clients.get_indexer(users)
        keep = (index >= 0) & (times < (self.cutoff + 1) * DAY_NS)
        rows = index[keep]
        times = times[keep]
        if not len(rows):
            return
        if kind != "product_buy":
            np.maximum.at(self.latest_other, rows, times)
        if kind in POST_EVENTS:
            after = (self.last_buy[rows] >= 0) & (times > self.last_buy[rows])
            after_rows = rows[after]
            days = times[after] // DAY_NS - self.first_day
            np.add.at(self.counts[:, POST_EVENTS.index(kind)], after_rows, 1)
            np.bitwise_or.at(
                self.day_bits,
                (after_rows, days // 64),
                np.left_shift(np.uint64(1), (days % 64).astype(np.uint64)),
            )
        if skus is None:
            return
        source = int(kind == "add_to_cart")
        weight = np.exp2((times // DAY_NS - self.cutoff) / 30).astype(np.float32)
        folds = audience_folds(users[keep])
        np.add.at(self.weights[source], rows, weight)
        for e, (name, keys) in enumerate((("sku", skus), ("category", categories))):
            for h, horizon in enumerate((14, 28)):
                levels, values = self.priors[(name, horizon)]
                idx = levels.get_indexer(keys[keep])
                probs = values[folds, idx, :]
                for k in (0, 1):
                    np.add.at(self.sums[source, :, 4 * e + 2 * h + k], rows, weight * probs[:, k])

    def finish(self):
        valid = self.last_buy >= 0
        last_day = self.last_buy // DAY_NS
        age = np.where(valid, self.cutoff - last_day, 0)
        active_days = np.bitwise_count(self.day_bits).sum(axis=1)
        delay = np.where(
            valid & (self.latest_other > self.last_buy), self.latest_other // DAY_NS - last_day, 0
        )
        features = np.zeros((len(self.clients), len(BASE_NAMES + POST_NAMES + PRIOR_NAMES)), np.float32)
        features[:, :3] = np.column_stack((valid, np.log1p(self.purchase_days), np.log1p(age)))
        features[:, 3:6] = np.log1p(self.counts)
        features[:, 6:11] = np.column_stack(
            (
                np.log1p(active_days),
                active_days / (age + 1),
                valid & (self.last_buy > self.latest_other),
                valid & (self.last_buy == self.latest_other),
                np.log1p(delay),
            )
        )
        for i in range(2):
            features[:, 11 + i * 8 : 19 + i * 8] = np.divide(
                self.sums[i], self.weights[i, :, None], out=self.sums[i], where=self.weights[i, :, None] > 0
            )
        del self.priors, self.sums, self.day_bits, self.weights, self.counts, self.latest_other
        return features


def future_labels(buys, snapshot):
    labels = []
    for start, end in HORIZONS:
        users = buys.loc[
            buys.day.between(snapshot.cutoff + start + 1, snapshot.cutoff + end), "user"
        ].unique()
        labels.append(snapshot.clients.isin(users))
    return np.column_stack(labels)


def evaluation_scores(name, indices, scores, target, single, horizon):
    def auc(mask):
        if len(np.unique(target[mask])) < 2:
            return None
        return float(roc_auc_score(target[mask], scores[mask]))

    return dict(
        days=list(horizon),
        model=name,
        columns=len(indices),
        positives=int(target.sum()),
        buyers=len(scores),
        single_day_buyers=int(single.sum()),
        single_day_positives=int(target[single].sum()),
        auc=auc(np.ones(len(scores), bool)),
        single_day_auc=auc(single),
    )


def ensemble_trees(seed):
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier

    return (
        LGBMClassifier(
            n_estimators=130,
            learning_rate=0.06,
            num_leaves=15,
            min_child_samples=90,
            verbosity=-1,
            n_jobs=2,
            random_state=seed,
        ),
        CatBoostClassifier(
            iterations=180,
            depth=5,
            learning_rate=0.07,
            verbose=False,
            thread_count=2,
            allow_writing_files=False,
            random_seed=seed,
        ),
    )


def odds_scores(models, features):
    # Test calibration and downstream inference both use the arithmetic mean of the two raw logits.
    leaf = models[0].booster_.predict(features, raw_score=True, num_threads=2)
    symmetric = models[1].predict(features, prediction_type="RawFormulaVal", thread_count=2)
    return leaf, symmetric, 0.5 * (leaf + symmetric)


def historical_models(features, labels, purchase_days, snapshots, full, seed, compare_models):
    """Validate on unseen users at the most recent historical anchor with a 28-day embargo."""
    test = len(snapshots) - 1
    train = [i for i, snap in enumerate(snapshots) if snap.cutoff + 28 <= snapshots[test].cutoff]
    diagnostic = []
    logits = np.zeros((len(full), len(HORIZONS)), np.float32)
    train_user = audience_folds(snapshots[test].clients.to_numpy()) != 0
    test_buyers = (~train_user) & (purchase_days[test] > 0)
    single = purchase_days[test][test_buyers] == 1
    xtrain = np.concatenate([features[i][train_user & (purchase_days[i] > 0)] for i in train])
    xvalid = features[test][test_buyers]
    xfit = np.concatenate([x[days > 0] for x, days in zip(features, purchase_days)])
    settings = dict(
        max_iter=80,
        max_leaf_nodes=15,
        max_bins=63,
        min_samples_leaf=90,
        learning_rate=0.08,
        early_stopping=False,
        random_state=seed,
    )
    base = list(range(len(BASE_NAMES)))
    post = list(range(len(BASE_NAMES), len(BASE_NAMES + POST_NAMES)))
    priors = list(range(len(BASE_NAMES + POST_NAMES), full.shape[1]))
    ablations = [
        ("purchase baseline", base),
        ("purchase history, continuation and priors", base + post + priors),
    ]
    if compare_models:
        ablations.extend(
            [
                ("purchase history and continuation", base + post),
                ("purchase history and priors", base + priors),
            ]
        )
    for column, horizon in enumerate(HORIZONS):
        ytrain = np.concatenate([labels[i][train_user & (purchase_days[i] > 0), column] for i in train])
        yvalid = labels[test][test_buyers, column]
        yfit = np.concatenate([y[days > 0, column] for y, days in zip(labels, purchase_days)])
        if len(np.unique(ytrain)) < 2 or not len(yvalid):
            raise ValueError(
                "Historical comparison needs buyers with both outcomes; increase sample size or history"
            )
        for name, indices in ablations:
            model = HistGradientBoostingClassifier(**settings).fit(xtrain[:, indices], ytrain)
            scores = model.decision_function(xvalid[:, indices])
            diagnostic.append(
                evaluation_scores(f"histogram boosting: {name}", indices, scores, yvalid, single, horizon)
            )
        if compare_models:
            models = ensemble_trees(seed)
            for model in models:
                model.fit(xtrain, ytrain)
            for name, scores in zip(("LightGBM", "CatBoost", "mean log-odds"), odds_scores(models, xvalid)):
                diagnostic.append(
                    evaluation_scores(name, base + post + priors, scores, yvalid, single, horizon)
                )
            for model in models:
                model.fit(xfit, yfit)
        else:
            model = HistGradientBoostingClassifier(**settings).fit(xfit, yfit)
        for start in range(0, len(full), 16384):
            x = full[start : start + 16384]
            scores = odds_scores(models, x)[2] if compare_models else model.decision_function(x)
            logits[start : start + len(x), column] = np.where(x[:, 0] > 0, scores, 0)
    return logits, diagnostic


def build(data_directory, output_directory, sample_users=200000, seed=42, compare_models=False):
    start = perf_counter()
    data, output = Path(data_directory), Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "embeddings.npy").exists():
        raise FileExistsError(output / "embeddings.npy")
    ids, paths = validate_inputs(data)
    catalog = pq.read_table(
        data / "product_properties.parquet", columns=["sku", "category"], use_threads=False
    ).to_pandas()
    category_map = catalog.set_index("sku").category
    del catalog
    buy_path = paths[EVENTS.index("product_buy")]
    parts = [
        pd.DataFrame({"user": user, "time": time, "sku": sku})
        for user, time, sku in event_batches(buy_path, product=True)
    ]
    buys = pd.concat(parts, ignore_index=True)
    del parts
    buys["day"] = (buys.time // DAY_NS).astype(np.int32)
    buys["category"] = buys.sku.map(category_map).fillna(-1).astype(np.int32)
    first_day = int(buys.day.min())
    cutoff = input_cutoff(paths, 160000) // DAY_NS
    rng = np.random.default_rng(seed)
    sampled = ids[np.sort(rng.choice(len(ids), min(sample_users, len(ids)), replace=False))]
    historical = [Snapshot(sampled, cutoff - age, buys, first_day) for age in (70, 56, 42, 28)]
    current = Snapshot(ids, cutoff, buys, first_day)
    labels = [future_labels(buys, snap) for snap in historical]
    for event, path in zip(EVENTS, paths):
        product = event in ("product_buy", "add_to_cart")
        for batch in event_batches(path, product):
            users, times = batch[:2]
            sku = batch[2] if product else None
            category = category_map.reindex(sku, fill_value=-1).to_numpy() if product else None
            earliest = int(times.min()) // DAY_NS
            # The historical snapshots share one sampled audience. Project the customer column once.
            if earliest <= historical[-1].cutoff:
                positions = historical[0].clients.get_indexer(users)
                for snapshot in historical:
                    if earliest <= snapshot.cutoff:
                        snapshot.add_events(event, users, times, sku, category, positions)
            current.add_events(event, users, times, sku, category)
        print(f"{event} complete in {perf_counter()-start:.1f}s", flush=True)
    del buys, category_map
    features = [snap.finish() for snap in historical]
    full = current.finish()
    with threadpool_limits(limits=2):
        logits, diagnostics = historical_models(
            features, labels, [s.purchase_days for s in historical], historical, full, seed, compare_models
        )
    names = BASE_NAMES + POST_NAMES + PRIOR_NAMES + SCORE_NAMES
    width = len(names)
    dest = np.lib.format.open_memmap(
        output / "repeat.partial.npy", mode="w+", dtype=np.float16, shape=(len(ids), width)
    )
    for values, begin in ((full, 0), (logits, full.shape[1])):
        for i in range(values.shape[1]):
            col = values[:, i]
            mean, std = col.mean(dtype=np.float64), col.std(dtype=np.float64)
            dest[:, begin + i] = np.clip((col - mean) / max(std, 1e-6), -8, 8)
    del dest
    np.save(output / "client_ids.npy", ids)
    (output / "features.json").write_text(json.dumps(names, indent=2), encoding="utf-8")
    record = dict(
        users=len(ids),
        dimensions=width,
        seed=seed,
        sample_users=len(sampled),
        cutoff=int(cutoff),
        historical_days_before_cutoff=[70, 56, 42, 28],
        buyers=int((current.purchase_days > 0).sum()),
        buyers_one_purchase_day=int((current.purchase_days == 1).sum()),
        empirical_prior_smoothing=SMOOTHING,
        source="input event Parquets, client IDs and product categories; no official target files",
        priors="Counts of distinct historical buyer/item pairs bought within 28 days ending at each anchor; "
        "outcomes are same entity or any purchase during next 14/28 days. Two disjoint outcome windows "
        "per horizon when completely observed. Exclude the recipient's entire client-ID fold.",
        pooled_by="Individual purchase/cart events weighted by 30-day half-life. Unseen keys use cohort prior.",
        post_activity="Strictly later visit/cart-add/search; mixed last timestamp encoded separately; "
        "last-event flags also compare cart removals. All five event streams scanned.",
        validation="Public input history only. Final held-out anchor equals cutoff minus 28 days. "
        "Fold zero held out; training windows end before validation outcomes start.",
        historical_validation=diagnostics,
        normalization="historical population z-score clipped to +/-8",
        chosen_tree_scores=(
            "LightGBM and CatBoost arithmetic mean of raw logits" if compare_models else "histogram boosting"
        ),
        seconds=round(perf_counter() - start, 3),
    )
    (output / "manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    (output / "repeat.partial.npy").replace(output / "embeddings.npy")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--sample-users", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Compare separate blocks and use LightGBM/CatBoost scores",
    )
    build(**vars(parser.parse_args()))
