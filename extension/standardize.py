"""Standardize an existing user representation without rebuilding its features."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
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


def feature_blocks(source: Path, dimensions: int, *, pretrained: bool = False) -> list[dict]:
    """Read explicit block boundaries, including the previous experiment layout."""
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if "blocks" in manifest:
        return manifest["blocks"]
    if pretrained and dimensions == 457:
        return [{"name": "bottleneck", "start": 0, "stop": 256, "kind": "bottleneck"},
                {"name": "logits", "start": 256, "stop": 457, "kind": "logits"}]
    if dimensions == 544:
        return [{"name": "statistics_counts", "start": 0, "stop": 352, "kind": "statistics"},
                {"name": "name", "start": 352, "stop": 384, "kind": "embedding"},
                {"name": "query", "start": 384, "stop": 416, "kind": "embedding"},
                {"name": "graph", "start": 416, "stop": 544, "kind": "embedding"}]
    raise ValueError("Block scaling requires explicit manifest.blocks or a documented legacy layout")


def write_block_embeddings(matrix: np.ndarray, destination: Path, blocks: list[dict], *,
                           batch_size: int = 20_000, clip: float = 10.0) -> dict:
    """Scale counts by column and embedding blocks uniformly, without whitening."""
    rows, dimensions = matrix.shape
    cursor = 0
    rules = []
    for block in blocks:
        start, stop, kind = block["start"], block["stop"], block["kind"]
        if start != cursor or stop <= start or stop > dimensions:
            raise ValueError("Blocks must partition the feature columns in order")
        cursor = stop
        if kind in {"statistics", "counts", "count", "signed_count"}:
            method, weight = "column", 1.0
        elif kind == "bottleneck" or (block["name"].startswith("pretrained_") and kind == "embedding"):
            method, weight = "layer", 1.3
        elif kind in {"logits", "probability"}:
            method, weight = kind, 1.3
        elif kind == "embedding":
            method, weight = "row_l2", 2.0
        else:
            raise ValueError(f"Unknown block kind: {kind}")
        rules.append({**block, "method": method, "weight": weight})
    if cursor != dimensions:
        raise ValueError("Blocks do not cover every feature column")
    column_indices = np.asarray([column for b in rules if b["method"] == "column"
                                 for column in range(b["start"], b["stop"])], dtype=np.int64)
    mean, sum_squares = np.zeros(len(column_indices)), np.zeros(len(column_indices))
    count = 0
    for start in range(0, rows, batch_size):
        values = np.asarray(matrix[start:start + batch_size, column_indices], dtype=np.float64)
        length = len(values)
        block_mean = values.mean(axis=0)
        difference = block_mean - mean
        total = count + length
        sum_squares += values.var(axis=0) * length + difference ** 2 * count * length / total
        mean += difference * length / total
        count = total
    scale = np.sqrt(sum_squares / rows)
    scale[scale < 1e-6] = 1.0
    temporary = destination.with_name(destination.stem + ".partial.npy")
    output = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16, shape=matrix.shape)
    for start in range(0, rows, batch_size):
        values = np.array(matrix[start:start + batch_size], dtype=np.float32, copy=True)
        values[:, column_indices] = np.clip((values[:, column_indices] - mean) / scale, -clip, clip)
        for block in rules:
            part = values[:, block["start"]:block["stop"]]
            if block["method"] == "row_l2":
                part /= np.maximum(np.linalg.norm(part, axis=1, keepdims=True), 1e-12)
            elif block["method"] == "layer":
                part -= part.mean(axis=1, keepdims=True)
                part /= np.maximum(part.std(axis=1, keepdims=True), 1e-6)
            elif block["method"] == "logits":
                part[:] = 1 / (1 + np.exp(-np.clip(part, -80, 80)))
            part *= block["weight"]
        if not np.isfinite(values).all():
            raise ValueError("Block scaling produced non-finite features")
        output[start:start + batch_size] = values.astype(np.float16)
    output.flush()
    del output
    temporary.replace(destination)
    return {"method": "block", "blocks": rules, "statistics_population": rows,
            "column_indices": column_indices.tolist(), "mean": mean.tolist(),
            "scale": scale.tolist(), "clip": clip,
            "probability_note": "Sigmoid scores are not claimed to be calibrated probabilities"}


def standardize_embeddings(embeddings_directory: str | Path, output_directory: str | Path,
                           batch_size: int = 20_000, clip: float = 10.0,
                           overwrite: bool = False, method: str = "column",
                           pretrained_directory: str | Path | None = None) -> dict:
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
    blocks = feature_blocks(source, matrix.shape[1]) if method == "block" else []
    names = json.loads((source / "features.json").read_text(encoding="utf-8")) if (source / "features.json").exists() else None
    combined_path = None
    if pretrained_directory is not None:
        pretrained = Path(pretrained_directory)
        if not np.array_equal(ids, np.load(pretrained / "client_ids.npy", allow_pickle=False)):
            raise ValueError("Pretrained and historical client IDs must have the same order")
        core = np.load(pretrained / "embeddings.npy", mmap_mode="r", allow_pickle=False)
        width = core.shape[1]
        if width + matrix.shape[1] > 2048:
            raise ValueError("Combined representation exceeds the official 2048-column limit")
        combined_path = output / "combined.partial.npy"
        combined = np.lib.format.open_memmap(combined_path, mode="w+", dtype=np.float32,
                                            shape=(len(ids), width + matrix.shape[1]))
        for start in range(0, len(ids), batch_size):
            combined[start:start + batch_size, :width] = core[start:start + batch_size]
            combined[start:start + batch_size, width:] = matrix[start:start + batch_size]
        if method == "block":
            blocks = [{**b, "name": "pretrained_" + b["name"]}
                      for b in feature_blocks(pretrained, width, pretrained=True)] + [
                          {**b, "start": b["start"] + width, "stop": b["stop"] + width} for b in blocks]
        if names is not None:
            core_names = json.loads((pretrained / "features.json").read_text(encoding="utf-8"))
            names = ["pretrained_" + name for name in core_names] + names
        matrix = combined
    normalization = (write_block_embeddings(matrix, output / "embeddings.npy", blocks,
                                           batch_size=batch_size, clip=clip) if method == "block" else
                     write_embeddings(matrix, output / "embeddings.npy", batch_size=batch_size,
                                      standardize=True, clip=clip))
    np.save(output / "client_ids.npy", ids, allow_pickle=False)
    if names is not None:
        (output / "features.json").write_text(json.dumps(names, indent=2) + "\n", encoding="utf-8")
    else:
        (output / "features.json").unlink(missing_ok=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "column_standardization_of_existing_embeddings" if method == "column" else "block_scaling_of_existing_embeddings",
        "source_directory": str(source), "clients": len(ids), "dimensions": matrix.shape[1],
        "source_files": {name: {"bytes": (source / name).stat().st_size,
                                "mtime_ns": (source / name).stat().st_mtime_ns}
                         for name in ("embeddings.npy", "client_ids.npy")},
        "dtype": "float16", "target_data_used_for_embeddings": False,
        "seconds": round(time.perf_counter() - started, 3), "numpy_version": np.__version__,
    }
    if pretrained_directory is not None:
        manifest["pretrained_directory"] = str(Path(pretrained_directory).resolve())
    if combined_path is not None:
        del matrix, combined
        combined_path.unlink()
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
    parser.add_argument("--method", choices=("column", "block"), default="column")
    parser.add_argument("--pretrained-directory", type=Path)
    print(json.dumps(standardize_embeddings(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
