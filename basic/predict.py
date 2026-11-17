"""Produce complete test probabilities from a fitted KKBox model in chunks."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import file_digest, read_interactions


def predict(data_dir: Path, model_dir: Path, output: Path, *, chunk_size: int = 200_000, threads: int = 8) -> dict:
    started = time.perf_counter()
    data_dir, model_dir, output = Path(data_dir), Path(model_dir), Path(output)
    if chunk_size <= 0 or threads <= 0:
        raise ValueError("chunk_size and threads must be positive")
    if output.exists():
        raise ValueError(f"Refusing to replace an existing prediction file: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    model = lgb.Booster(model_str=(model_dir / "model.txt").read_text(encoding="utf-8"))
    # features.pkl must come from this project's own trusted training output.
    with (model_dir / "features.pkl").open("rb") as stream:
        builder = pickle.load(stream)
    if model.feature_name() != builder.feature_names:
        raise ValueError("Model feature names do not match serialized feature state")
    temporary = output.with_name(output.name + ".partial")
    if temporary.exists():
        raise ValueError(f"A partial prediction already exists: {temporary}")
    rows = 0
    try:
        for frame in read_interactions(data_dir / "test.csv", training=False, chunksize=chunk_size):
            features = builder.transform(frame)
            probabilities = model.predict(features, num_threads=threads)
            if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
                raise ValueError("Model produced non-finite or out-of-range probabilities")
            pd.DataFrame({"id": frame["id"].to_numpy(), "target": probabilities}).to_csv(
                temporary, mode="w" if rows == 0 else "a", header=rows == 0,
                index=False, float_format="%.10g",
            )
            rows += len(frame)
            print(f"Predicted {rows:,} rows", flush=True)
        if rows == 0:
            raise ValueError("The test CSV is empty")
        from .submission import validate_submission
        check = validate_submission(temporary, data_dir / "sample_submission.csv", data_dir / "test.csv")
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    result = {"rows": rows, "output": str(output.resolve()), "validation": check,
        "seconds": time.perf_counter()-started, "model_sha256": file_digest(model_dir/"model.txt"),
        "test_sha256": file_digest(data_dir/"test.csv"), "submission_sha256": file_digest(output)}
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(predict(**vars(args)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
