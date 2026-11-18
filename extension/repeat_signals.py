"""Encode activity-informed repeat scores for appending to the compact purchase profile.

Run `python -m extension.repeat_signals --help` for the four input/output paths.
The purchase profile itself keeps its four original scores. This additional block
already includes the gain, so append it with combine_profiles weight 1.
"""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pyarrow.parquet as parquet
from scipy.special import expit
from threadpoolctl import threadpool_limits

from .build_embeddings import DAY_NS, timestamps_ns
from .repeat_purchases import SCORE_NAMES


def buyer_groups(ids, purchases, cutoff_day):
    """First and last observed purchase day suffice to identify single-day buyers."""
    index = pd.Index(ids)
    first = np.full(len(ids), np.iinfo(np.int32).max, np.int32)
    last = np.full(len(ids), np.iinfo(np.int32).min, np.int32)
    for batch in parquet.ParquetFile(purchases).iter_batches(
        columns=["client_id", "timestamp"], batch_size=131072, use_threads=False
    ):
        rows = index.get_indexer(batch.column(0).to_numpy())
        times = timestamps_ns(batch.to_pandas()["timestamp"])
        days = times // DAY_NS
        keep = (rows >= 0) & (times != np.iinfo(np.int64).min) & (days <= cutoff_day)
        np.minimum.at(first, rows[keep], days[keep])
        np.maximum.at(last, rows[keep], days[keep])
    return last >= first, last == first


def standardized(values):
    means = values.mean(axis=0, dtype=np.float64)
    scales = values.std(axis=0, dtype=np.float64)
    adjusted = (values - means) / np.where(scales > 1e-8, scales, 1)
    return np.clip(adjusted, -8, 8).astype(np.float32)


def encode(purchase, activity, single_day, gain):
    """Fit only on pre-cutoff scores of historical buyers; no outcomes are needed."""
    design = np.column_stack((np.ones(len(purchase)), purchase))
    coefficients = np.linalg.lstsq(design, activity, rcond=None)[0]
    residual = activity - design @ coefficients
    scores = standardized(activity)
    gated = np.zeros_like(scores)
    if single_day.any():
        gated[single_day] = standardized(activity[single_day])

    # The final column is the 15-28 day score. Smooth thresholds preserve its order.
    delayed = scores[:, -1]
    centers = np.quantile(delayed, np.arange(1, 17) / 17)
    activations = expit((delayed[:, None] - centers) / 0.35)
    values = np.column_stack((scores, standardized(residual), gated, standardized(activations)))
    return (gain * values).astype(np.float16), coefficients, centers


def build(data_directory, purchase_directory, activity_directory, output_directory, gain=2.0):
    if not np.isfinite(gain) or not 0 < gain <= np.finfo(np.float16).max / 8:
        raise ValueError("Gain must be positive and fit scaled scores in float16")
    paths = [Path(purchase_directory), Path(activity_directory)]
    data = Path(data_directory) / "input"
    ids = np.load(data / "relevant_clients.npy")
    inputs, cutoffs = [], []
    for path in paths:
        if not np.array_equal(np.load(path / "client_ids.npy"), ids):
            raise ValueError(f"Client order differs in {path}")
        names = json.loads((path / "features.json").read_text(encoding="utf-8-sig"))
        matrix = np.load(path / "embeddings.npy", mmap_mode="r")
        if matrix.shape != (len(ids), len(names)) or names[-4:] != SCORE_NAMES:
            raise ValueError(f"Missing four aligned repeat scores in {path}")
        inputs.append(matrix[:, -4:])
        metadata = json.loads((path / "manifest.json").read_text(encoding="utf-8-sig"))
        cutoffs.append(metadata["cutoff_day"] if "cutoff_day" in metadata else metadata["cutoff"])
    if cutoffs[0] != cutoffs[1]:
        raise ValueError("Purchase and activity score cutoffs differ")
    buyers, single_day = buyer_groups(ids, data / "product_buy.parquet", cutoffs[0])
    if not buyers.any():
        raise ValueError("No history buyers in the supplied profile population")
    with threadpool_limits(limits=1):
        encoded, coefficients, centers = encode(
            inputs[0][buyers].astype(np.float32), inputs[1][buyers].astype(np.float32),
            single_day[buyers], gain,
        )
    if not np.isfinite(encoded).all():
        raise ValueError("Nonfinite repeat signal")
    names = [f"{kind}_{name}" for kind in ("activity", "activity_residual", "single_day_activity")
             for name in SCORE_NAMES]
    names.extend(f"activity_repeat_days_15_28_threshold_{i}" for i in range(1, 17))
    record = dict(clients=len(ids), dimensions=len(names), dtype="float16", gain=gain,
                  buyers=int(buyers.sum()), single_day_buyers=int(single_day.sum()), cutoff_day=cutoffs[0],
                  residual_design=["intercept", *SCORE_NAMES], coefficients=coefficients.tolist(),
                  threshold_centers=centers.tolist(), threshold_temperature=0.35,
                  normalization="History buyers only; gated columns use single-day buyers; ineligible rows are zero.",
                  official_target_directory_read=False)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "embeddings.npy").exists():
        raise FileExistsError(output)
    with TemporaryDirectory(dir=output) as temporary:
        stage = Path(temporary)
        result = np.lib.format.open_memmap(
            stage / "embeddings.npy", mode="w+", dtype=np.float16, shape=(len(ids), len(names))
        )
        result[:] = 0
        result[buyers] = encoded
        del result
        np.save(stage / "client_ids.npy", ids)
        (stage / "features.json").write_text(json.dumps(names, indent=2), encoding="utf-8")
        (stage / "manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        for name in ("features.json", "manifest.json", "client_ids.npy", "embeddings.npy"):
            (stage / name).replace(output / name)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "purchase", "activity", "output"):
        parser.add_argument(f"--{name}-directory", required=True, type=Path)
    parser.add_argument("--gain", type=float, default=2.)
    print(json.dumps(build(**vars(parser.parse_args())), indent=2))
