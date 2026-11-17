"""Standardize an existing user representation without rebuilding its features."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np


def _values(matrix: np.ndarray, start: int, stop: int, log_transform: bool) -> np.ndarray:
    values = np.asarray(matrix[start:stop], dtype=np.float32)
    return np.log1p(values) if log_transform else values


def write_embeddings(matrix: np.ndarray, destination: Path, *, batch_size: int = 20_000,
                     standardize: bool = False, log_transform: bool = False,
                     clip: float = 10.0) -> dict:
    """Write float16 in bounded chunks; fit column moments without target labels."""
    if batch_size < 1 or clip <= 0 or not np.isfinite(clip):
        raise ValueError("batch_size and clip must be positive")
    rows, dimensions = matrix.shape
    mean, variance = np.zeros(dimensions), np.zeros(dimensions)
    count = 0
    if standardize:
        for start in range(0, rows, batch_size):
            values = _values(matrix, start, start + batch_size, log_transform).astype(np.float64)
            if not np.isfinite(values).all():
                raise ValueError("Non-finite features cannot be standardized")
            length = len(values)
            block_mean = values.mean(axis=0)
            difference = block_mean - mean
            total = count + length
            variance += values.var(axis=0) * length + difference ** 2 * count * length / total
            mean += difference * length / total
            count = total
        scale = np.sqrt(variance / rows)
        scale[scale < 1e-6] = 1.0
    else:
        scale = np.ones(dimensions)
    temporary = destination.with_name(destination.stem + ".partial.npy")
    output = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16, shape=matrix.shape)
    try:
        for start in range(0, rows, batch_size):
            values = _values(matrix, start, start + batch_size, log_transform)
            if standardize:
                values = np.clip((values - mean) / scale, -clip, clip)
            if not np.isfinite(values).all():
                raise ValueError("Non-finite features were produced")
            output[start:start + batch_size] = values.astype(np.float16)
        output.flush()
    finally:
        del output
    temporary.replace(destination)
    return {"enabled": standardize, "clip": clip if standardize else None,
            "mean": mean.tolist() if standardize else None,
            "scale": scale.tolist() if standardize else None,
            "statistics_population": rows if standardize else None}


def standardize_embeddings(embeddings_directory: str | Path, output_directory: str | Path,
                           batch_size: int = 20_000, clip: float = 10.0,
                           overwrite: bool = False) -> dict:
    started = time.perf_counter()
    source, output = Path(embeddings_directory).resolve(), Path(output_directory).resolve()
    if source == output:
        raise ValueError("Use a different output directory to retain the original representation")
    output.mkdir(parents=True, exist_ok=True)
    reserved = ("embeddings.npy", "client_ids.npy", "features.json", "manifest.json", "standardization.json")
    if not overwrite and any((output / name).exists() for name in reserved):
        raise FileExistsError("Output exists; use --overwrite to replace this experiment")
    if overwrite:
        (output / "manifest.json").unlink(missing_ok=True)
    matrix = np.load(source / "embeddings.npy", mmap_mode="r", allow_pickle=False)
    ids = np.load(source / "client_ids.npy", mmap_mode="r", allow_pickle=False)
    if matrix.ndim != 2 or ids.ndim != 1 or len(matrix) != len(ids) or len(ids) == 0:
        raise ValueError("Client IDs and embedding rows must align and be non-empty")
    normalization = write_embeddings(matrix, output / "embeddings.npy", batch_size=batch_size,
                                     standardize=True, clip=clip)
    np.save(output / "client_ids.npy", ids, allow_pickle=False)
    if (source / "features.json").exists():
        shutil.copyfile(source / "features.json", output / "features.json")
    else:
        (output / "features.json").unlink(missing_ok=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "column_standardization_of_existing_embeddings",
        "source_directory": str(source), "clients": len(ids), "dimensions": matrix.shape[1],
        "source_files": {name: {"bytes": (source / name).stat().st_size,
                                "mtime_ns": (source / name).stat().st_mtime_ns}
                         for name in ("embeddings.npy", "client_ids.npy")},
        "dtype": "float16", "target_data_used_for_embeddings": False,
        "seconds": round(time.perf_counter() - started, 3), "numpy_version": np.__version__,
    }
    (output / "standardization.json").write_text(json.dumps(normalization, indent=2) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=20_000)
    parser.add_argument("--clip", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    print(json.dumps(standardize_embeddings(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
