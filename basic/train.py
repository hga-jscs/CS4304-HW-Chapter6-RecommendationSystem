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

from .features import FeatureBuilder, file_fingerprint, load_training, load_unlabeled


TRAINING_OUTPUTS = (
    "model.txt", "features.pkl", "validation_model.txt", "metrics.json",
    "configuration.json", "data_manifest.json", "validation_predictions.csv",
    "validation_history.json", "feature_importance.csv",
)


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
    output_dir.mkdir(parents=True, exist_ok=True)
    previous = [output_dir / name for name in TRAINING_OUTPUTS if (output_dir / name).exists()]
    if previous and not args.overwrite:
        raise ValueError(f"Training outputs already exist; use --overwrite to replace this experiment: {output_dir}")
    for path in previous:
        path.unlink()
    frame, sampling = load_training(data_dir / "train.csv", max_rows=args.max_train_rows, seed=args.seed,
                                    chunksize=args.chunk_size, sampling_mode=args.sampling_mode)
    if args.transductive:
        frame["global_row_id"] = frame["row_id"]
    class_counts = frame["target"].value_counts()
    if len(class_counts) != 2 or class_counts.min() < 2:
        raise ValueError("Both target classes need at least two examples")
    if args.split_mode == "row_order":
        boundary = int(len(frame) * (1 - args.validation_fraction))
        training, validation = frame.iloc[:boundary], frame.iloc[boundary:]
    else:
        train_indices, validation_indices = train_test_split(
            np.arange(len(frame)), test_size=args.validation_fraction,
            random_state=args.seed, stratify=frame["target"].to_numpy(),
        )
        training, validation = frame.iloc[train_indices], frame.iloc[validation_indices]
    if training["target"].nunique() != 2 or validation["target"].nunique() != 2:
        raise ValueError("Training and validation partitions both need the two target classes")
    builder = FeatureBuilder(data_dir, args.feature_mode, disable_frequency_features=args.disable_frequency_features,
                             user_context_features=args.user_context_features,
                             numeric_features=args.numeric_features, frequency_mode=args.frequency_mode,
                             context_counting=args.context_counting,
                             user_preference_features=args.user_preference_features,
                             collaborative_features=args.collaborative_features,
                             transductive=args.transductive, row_order_features=args.row_order_features,
                             target_encoding=args.target_encoding, song_components=args.song_components,
                             artist_components=args.artist_components, genre_components=args.genre_components)
    unsupervised_started = time.perf_counter()
    if args.transductive:
        unlabeled, official_train_rows = load_unlabeled(data_dir)
        builder.fit(training, unlabeled_frames=unlabeled, train_rows=official_train_rows)
        del unlabeled
    else:
        builder.fit(training)
    unsupervised_seconds = time.perf_counter() - unsupervised_started
    training_target_values = (builder.fit_target_encoding(training, split_mode=args.split_mode, seed=args.seed)
                              if args.target_encoding else None)
    training_features = builder.transform(training, training_context=True, target_values=training_target_values)
    del training_target_values
    validation_features = builder.transform(validation)
    labels = training["target"].to_numpy()
    validation_labels = validation["target"].to_numpy()
    params = {
        "objective": "binary", "metric": "auc", "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves, "min_data_in_leaf": args.min_child_samples,
        "feature_fraction": args.feature_fraction, "bagging_fraction": 0.9, "bagging_freq": 1,
        "lambda_l2": 1.0, "max_bin": 255, "max_cat_threshold": args.max_cat_threshold,
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
        **sampling, "split": "row_order_holdout_not_verified_time" if args.split_mode == "row_order" else "stratified_random_holdout_not_chronological",
        "split_seed": args.seed, "training_rows": len(training), "validation_rows": len(validation),
        "feature_mode": args.feature_mode, "feature_count": len(builder.feature_names),
        "best_iteration": best_iteration, "validation_auc": float(roc_auc_score(validation_labels, predictions)),
        "validation_log_loss": float(log_loss(validation_labels, predictions, labels=[0, 1])),
        "constant_baseline_auc": 0.5,
        "constant_baseline_log_loss": float(log_loss(validation_labels, np.full(len(validation), mean_prediction), labels=[0, 1])),
        "validation_training_seconds": time.perf_counter() - validation_started,
        "unsupervised_statistics": {
            "scope": "full_train_and_test_without_labels" if args.transductive else "current_supervised_training_rows_without_labels",
            "validation_fit_rows": builder.unlabeled_rows if args.transductive else len(training),
            "frequency_mode": "relative" if args.transductive else args.frequency_mode,
            "context_counting": "full" if args.transductive else args.context_counting,
            "initial_fit_seconds": unsupervised_seconds,
            "reused_for_refit": bool(args.transductive and not args.no_refit),
        },
        "segments": {}, "platform_score": None,
    }
    if args.collaborative_features:
        metrics["validation_collaborative_fit"] = builder.collaborative.metadata
    if args.target_encoding:
        metrics["validation_target_encoding"] = builder.target_encoder.metadata
    # Diagnose availability of labeled training history, even with global vocabularies.
    known_user = validation["msno"].isin(training["msno"].unique()).to_numpy()
    known_song = validation["song_id"].isin(training["song_id"].unique()).to_numpy()
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
    del training_set, validation_set, training_features, validation_features, training, validation
    gc.collect()
    final_started = time.perf_counter()
    if args.no_refit:
        print(f"Validation ROC-AUC={metrics['validation_auc']:.6f}; keep validated model without refitting", flush=True)
        final_model = model
        metrics["final_training_rows"] = metrics["training_rows"]
    else:
        del model
        print(f"Validation ROC-AUC={metrics['validation_auc']:.6f}; refit {best_iteration} iterations on {len(frame):,} rows", flush=True)
        builder.fit(frame)
        final_target_values = (builder.fit_target_encoding(frame, split_mode=args.split_mode, seed=args.seed)
                               if args.target_encoding else None)
        final_features = builder.transform(frame, training_context=True, target_values=final_target_values)
        del final_target_values
        final_set = lgb.Dataset(final_features, label=frame["target"].to_numpy(), categorical_feature=builder.categorical_columns)
        final_model = lgb.train(params, final_set, num_boost_round=best_iteration, callbacks=[lgb.log_evaluation(25)])
        metrics["final_training_rows"] = len(frame)
    (output_dir / "model.txt").write_text(final_model.model_to_string(num_iteration=best_iteration), encoding="utf-8")
    with (output_dir / "features.pkl").open("wb") as stream:
        pickle.dump(builder, stream, protocol=pickle.HIGHEST_PROTOCOL)
    metrics["refit_performed"] = not args.no_refit
    metrics["unsupervised_statistics"]["final_fit_rows"] = (
        builder.unlabeled_rows if args.transductive else metrics["final_training_rows"])
    if args.collaborative_features:
        metrics["final_collaborative_fit"] = builder.collaborative.metadata
    if args.target_encoding:
        metrics["final_target_encoding"] = builder.target_encoder.metadata
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
    config["effective_statistics"] = {key: metrics["unsupervised_statistics"][key]
                                       for key in ("scope", "frequency_mode", "context_counting")}
    config["effective_statistics"]["rows"] = metrics["unsupervised_statistics"]["final_fit_rows"]
    manifest_files = ["train.csv", "members.csv", "songs.csv"] + (["song_extra_info.csv"] if args.feature_mode == "enhanced" else [])
    if args.transductive:
        manifest_files.append("test.csv")
    manifest = {name: file_fingerprint(data_dir/name) for name in manifest_files}
    _json(output_dir / "data_manifest.json", manifest)
    _json(output_dir / "configuration.json", config)
    metrics["total_seconds"] = time.perf_counter() - started
    if platform.system() == "Linux":
        import resource
        metrics["peak_rss_megabytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
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
    result.add_argument("--numeric-features", choices=["legacy", "calendar"], default="calendar", help="Calendar uses continuous days and an ISRC reference-year proxy; legacy reproduces earlier numeric columns")
    result.add_argument("--frequency-mode", choices=["count", "relative"], default="count")
    result.add_argument("--context-counting", choices=["leave_one_out", "full"], default="leave_one_out", help="Ablate self-counting for context and preference probabilities")
    result.add_argument("--user-preference-features", action="store_true", help="Add training-only smoothed user/artist and user/genre-combination probabilities")
    result.add_argument("--collaborative-features", action="store_true", help="Add sparse SVD similarities from the selected interaction scope")
    result.add_argument("--transductive", action="store_true", help="Fit unsupervised vocabularies, relative frequencies, full context counts and optional SVD once on unlabeled full train+test")
    result.add_argument("--row-order-features", action="store_true", help="Add six global file-order proxies; requires --transductive; gaps are not seconds or verified sessions")
    result.add_argument("--target-encoding", action="store_true", help="Add five training-only OOF target means; row_order uses strictly preceding blocks")
    result.add_argument("--song-components", type=int, default=None, help="SVD rank cap; default 48 transductive, 32 training-only")
    result.add_argument("--artist-components", type=int, default=None, help="SVD rank cap; default 24 transductive, 16 training-only")
    result.add_argument("--genre-components", type=int, default=None, help="SVD rank cap; default 16 transductive, 0 training-only; genre strings denote combinations")
    result.add_argument("--overwrite", action="store_true", help="Replace this experiment's training outputs; leave unrelated files untouched")
    result.add_argument("--no-refit", action="store_true", help="Save the validated model and its fitted feature state without training again on the holdout")
    result.add_argument("--max-train-rows", type=int, default=0, help="Uniform sample size over the whole CSV; 0 uses all records")
    result.add_argument("--sampling-mode", choices=["uniform", "tail"], default="uniform")
    result.add_argument("--split-mode", choices=["random", "row_order"], default="random", help="Row order is a diagnostic split, not a verified timestamp")
    result.add_argument("--validation-fraction", type=float, default=0.2)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--num-boost-round", type=int, default=400)
    result.add_argument("--early-stopping-rounds", type=int, default=30)
    result.add_argument("--learning-rate", type=float, default=0.08)
    result.add_argument("--num-leaves", type=int, default=63)
    result.add_argument("--min-child-samples", type=int, default=50)
    result.add_argument("--max-cat-threshold", type=int, default=32)
    result.add_argument("--feature-fraction", type=float, default=0.9)
    result.add_argument("--threads", type=int, default=8)
    result.add_argument("--chunk-size", type=int, default=200_000)
    return result


if __name__ == "__main__":
    train(parser().parse_args())
