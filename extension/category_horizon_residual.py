"""Keep near category scores and encode the extra joint-horizon signal in a small residual block."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

from .purchase_targets import normalized_latent, profile_batches, sample_rows


def category_coordinates(raw, params):
    z = normalized_latent(
        raw,
        params["mean"],
        params["scale"],
        params["category_basis"],
        params["category_center"],
        params["category_deviation"],
    )
    probability = expit(z @ params["category_weight"] + params["category_bias"])
    coordinates = (probability - params["category_probability_mean"]) @ params["category_axes"]
    # Same serialization as the frozen blocks which are subtracted during extraction.
    return np.clip(coordinates / params["category_projected_scale"], -6, 6).astype(np.float16)


def build(cache, near, joint, output, rank=16, residual_gain=0.5, seed=42):
    start = perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "embeddings.npy").exists():
        raise FileExistsError(output / "embeddings.npy")
    names = json.loads((near / "features.json").read_text(encoding="utf-8"))
    if names != json.loads((joint / "features.json").read_text(encoding="utf-8")):
        raise ValueError("Near and joint output columns differ")
    category = [
        i
        for i, name in enumerate(names)
        if name.startswith("historical_category_probability_component_")
    ]
    if not category or not 1 <= rank <= len(category):
        raise ValueError("Requested rank does not fit the category probability block")
    ids = np.load(cache / "client_ids.npy")
    heldout = np.load(cache / "validation_users.npy")
    if heldout.dtype != bool or heldout.shape != ids.shape:
        raise ValueError("Invalid historical validation mask")
    rng = np.random.default_rng(seed)
    users = sample_rows(np.flatnonzero(~heldout), 8192, rng)
    arrays = []
    for directory in (near, joint):
        with np.load(directory / "heads.npz") as archive:
            arrays.append(
                {
                    k: archive[k]
                    for k in archive.files
                    if k.startswith("category_") or k in ("mean", "scale")
                }
            )
    differences = []
    for window in ("earlier", "later"):
        anchor = cache / window
        if not np.array_equal(ids, np.load(anchor / "client_ids.npy")):
            raise ValueError(f"User order mismatch at {anchor}")
        for _, raw in profile_batches(anchor / "embeddings.npy", 4096, users):
            old, new = [category_coordinates(raw, a).astype(np.float32) for a in arrays]
            differences.append(new - old)
    delta = np.concatenate(differences)
    center = delta.mean(axis=0)
    delta -= center
    _, singular, axes = np.linalg.svd(delta, full_matrices=False)
    if len(singular) < rank:
        raise ValueError("Too few historical rows for requested residual dimensions")
    explained = float(np.square(singular[:rank]).sum() / max(np.square(singular).sum(), 1e-12))
    scale = np.maximum(singular[:rank] / np.sqrt(len(delta) - 1), 0.05).astype(np.float32)
    projection = axes[:rank].T.copy()
    partial = output / "embeddings.partial.npy"
    for directory in (near, joint):
        if not np.array_equal(ids, np.load(directory / "client_ids.npy")):
            raise ValueError(f"User order mismatch at {directory}")
    source = np.load(near / "embeddings.npy", mmap_mode="r")
    both = np.load(joint / "embeddings.npy", mmap_mode="r")
    if (
        source.shape != both.shape
        or source.shape != (len(ids), len(names))
        or source.dtype != both.dtype
        or both.dtype != np.float16
    ):
        raise ValueError("Expected two aligned float16 embedding matrices")
    result = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.float16, shape=(len(ids), len(names) + rank)
    )
    for begin in range(0, len(ids), 4096):
        rows = slice(begin, begin + 4096)
        old_cat = source[rows][:, category]
        change = both[rows][:, category].astype(np.float32) - old_cat
        reduced = np.clip(((change - center) @ projection) / scale, -4, 4) * residual_gain
        if not np.isfinite(reduced).all():
            raise ValueError("Nonfinite category residuals")
        block = result[rows]
        block[:, : len(names)] = both[rows]
        block[:, category] = old_cat
        block[:, len(names) :] = reduced
    del block, result
    names.extend([f"category_horizon_difference_{i}" for i in range(rank)])
    np.save(output / "client_ids.npy", ids)
    np.savez(output / "residual_projection.npz", center=center, axes=projection, scale=scale)
    (output / "features.json").write_text(json.dumps(names), encoding="utf-8")
    record = dict(
        clients=len(ids),
        dimensions=len(names),
        historical_training_users=len(users),
        fit_rows=len(delta),
        historical_snapshots=["earlier", "later"],
        residual_gain=residual_gain,
        residual_clip_before_gain=4,
        residual_scale_floor=0.05,
        residual_rank=rank,
        residual_pca_explained_fraction=explained,
        seed=seed,
        official_target_directory_read=False,
        seconds=round(perf_counter() - start, 3),
        method="Joint SKU coordinates; original near category coordinates; compressed joint-minus-near category difference.",
    )
    (output / "manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    partial.replace(output / "embeddings.npy")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "near", "joint", "output"):
        parser.add_argument("--" + name + "-directory", required=True, type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        print(
            build(
                args.cache_directory, args.near_directory, args.joint_directory, args.output_directory
            ),
            flush=True,
        )
