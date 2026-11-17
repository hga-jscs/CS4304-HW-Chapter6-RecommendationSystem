"""Train and validate KKBox LightGBM models, then refit on selected rows."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import pickle
import platform
import subprocess
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from .features import FeatureBuilder, file_digest, load_training


def _json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _auc(labels, predictions):
    return float(roc_auc_score(labels, predictions)) if len(np.unique(labels)) == 2 else None


def train(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    data_dir, output_dir = Path(args.data_dir), Path(args.output_dir)
    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if min(args.num_boost_round, args.early_stopping_rounds, args.threads, args.num_leaves, args.min_child_samples) <= 0:
        raise ValueError("Training rounds, threads, leaves, and leaf samples must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty; choose a new experiment directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, sampling = load_training(data_dir / "train.csv", max_rows=args.max_train_rows, seed=args.seed, chunksize=args.chunk_size)
    class_counts = frame["target"].value_counts()
    if len(class_counts) != 2 or class_counts.min() < 2:
        raise ValueError("Both target classes need at least two examples")
    if frame.duplicated(["msno", "song_id"]).any():
        raise ValueError("Duplicate user/song pairs found; a pair-grouped split is needed before evaluation")
    train_indices, validation_indices = train_test_split(
        np.arange(len(frame)), test_size=args.validation_fraction,
        random_state=args.seed, stratify=frame["target"].to_numpy(),
    )
    training = frame.iloc[train_indices]
    validation = frame.iloc[validation_indices]
    builder = FeatureBuilder(data_dir, args.feature_mode, disable_frequency_features=args.disable_frequency_features,
                             user_context_features=args.user_context_features).fit(training)
    training_features = builder.transform(training, training_context=True)
    validation_features = builder.transform(validation)
    labels = training["target"].to_numpy()
    validation_labels = validation["target"].to_numpy()
    params = {
        "objective": "binary", "metric": "auc", "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves, "min_data_in_leaf": args.min_child_samples,
        "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 1,
        "lambda_l2": 1.0, "max_bin": 255, "max_cat_threshold": 32,
        "cat_smooth": 10.0, "num_threads": args.threads, "verbosity": -1,
        "seed": args.seed, "feature_fraction_seed": args.seed,
        "bagging_seed": args.seed, "data_random_seed": args.seed,
        "deterministic": True, "force_col_wise": True,
    }
    print(f"Loaded {len(frame):,} of {sampling['available_rows']:,} rows; feature mode={args.feature_mode}", flush=True)
    training_set = lgb.Dataset(training_features, label=labels, categorical_feature=builder.categorical_columns)
    validation_set = lgb.Dataset(validation_features, label=validation_labels, reference=training_set, categorical_feature=builder.categorical_columns)
    history = {}
    validation_started = time.perf_counter()
    model = lgb.train(params, training_set, num_boost_round=args.num_boost_round,
        valid_sets=[validation_set], valid_names=["validation"], callbacks=[
            lgb.early_stopping(args.early_stopping_rounds), lgb.log_evaluation(25), lgb.record_evaluation(history)
        ])
    predictions = model.predict(validation_features, num_iteration=model.best_iteration, num_threads=args.threads)
    best_iteration = int(model.best_iteration or args.num_boost_round)
    mean_prediction = float(labels.mean())
    metrics = {
        **sampling, "split": "stratified_random_holdout_not_chronological",
        "split_seed": args.seed, "training_rows": len(training), "validation_rows": len(validation),
        "feature_mode": args.feature_mode, "feature_count": len(builder.feature_names),
        "best_iteration": best_iteration, "validation_auc": float(roc_auc_score(validation_labels, predictions)),
        "validation_log_loss": float(log_loss(validation_labels, predictions, labels=[0, 1])),
        "constant_baseline_auc": 0.5,
        "constant_baseline_log_loss": float(log_loss(validation_labels, np.full(len(validation), mean_prediction), labels=[0, 1])),
        "validation_training_seconds": time.perf_counter() - validation_started,
        "segments": {}, "platform_score": None,
    }
    known_user = builder.categories["msno"].get_indexer(validation["msno"]) >= 0
    known_song = builder.categories["song_id"].get_indexer(validation["song_id"]) >= 0
    for segment, mask in {
        "known_user_and_song": known_user & known_song,
        "new_user": ~known_user, "new_song": ~known_song,
    }.items():
        metrics["segments"][segment] = {"rows": int(mask.sum()), "auc": _auc(validation_labels[mask], predictions[mask])}
    pd.DataFrame({"row_id": validation["row_id"].to_numpy(), "target": validation_labels, "prediction": predictions}).to_csv(output_dir / "validation_predictions.csv", index=False)
    # Python handles Unicode Windows paths; the native LightGBM file API does not.
    (output_dir / "validation_model.txt").write_text(model.model_to_string(num_iteration=best_iteration), encoding="utf-8")
    _json(output_dir / "validation_history.json", history)
    importance = pd.DataFrame({"feature": builder.feature_names, "gain": model.feature_importance("gain", iteration=best_iteration)})
    importance.sort_values("gain", ascending=False).to_csv(output_dir / "feature_importance.csv", index=False)
    del training_set, validation_set, training_features, validation_features, training, validation, model
    gc.collect()
    print(f"Validation ROC-AUC={metrics['validation_auc']:.6f}; refit {best_iteration} iterations on {len(frame):,} rows", flush=True)
    final_started = time.perf_counter()
    builder.fit(frame)
    final_features = builder.transform(frame, training_context=True)
    final_set = lgb.Dataset(final_features, label=frame["target"].to_numpy(), categorical_feature=builder.categorical_columns)
    final_model = lgb.train(params, final_set, num_boost_round=best_iteration, callbacks=[lgb.log_evaluation(25)])
    (output_dir / "model.txt").write_text(final_model.model_to_string(), encoding="utf-8")
    with (output_dir / "features.pkl").open("wb") as stream:
        pickle.dump(builder, stream, protocol=pickle.HIGHEST_PROTOCOL)
    metrics["final_training_rows"] = len(frame)
    metrics["final_training_seconds"] = time.perf_counter() - final_started
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parent.parent, text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    config = vars(args).copy()
    config.update({"lightgbm_parameters": params, "git_revision": revision, "working_tree_dirty": dirty,
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {package: importlib.metadata.version(package) for package in ["numpy", "pandas", "pyarrow", "scikit-learn", "lightgbm"]}})
    manifest_files = ["train.csv", "members.csv", "songs.csv"] + (["song_extra_info.csv"] if args.feature_mode == "enhanced" else [])
    manifest = {name: {"bytes": (data_dir/name).stat().st_size, "sha256": file_digest(data_dir/name)} for name in manifest_files}
    _json(output_dir / "data_manifest.json", manifest)
    _json(output_dir / "configuration.json", config)
    metrics["total_seconds"] = time.perf_counter() - started
    _json(output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return metrics


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-dir", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--feature-mode", choices=["baseline", "enhanced"], default="baseline")
    result.add_argument("--disable-frequency-features", action="store_true", help="Ablation: omit the five enhanced-mode frequency features; keep other features unchanged")
    result.add_argument("--user-context-features", action="store_true", help="Add training-only smoothed user/context probabilities with leave-one-out training counts")
    result.add_argument("--max-train-rows", type=int, default=0, help="Uniform sample size over the whole CSV; 0 uses all records")
    result.add_argument("--validation-fraction", type=float, default=0.2)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--num-boost-round", type=int, default=400)
    result.add_argument("--early-stopping-rounds", type=int, default=30)
    result.add_argument("--learning-rate", type=float, default=0.08)
    result.add_argument("--num-leaves", type=int, default=63)
    result.add_argument("--min-child-samples", type=int, default=50)
    result.add_argument("--threads", type=int, default=8)
    result.add_argument("--chunk-size", type=int, default=200_000)
    return result


if __name__ == "__main__":
    train(parser().parse_args())
