"""Append ready-to-evaluate float16 feature blocks, preserving their values."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np


def combine(directories, output, weights=None):
    started = perf_counter()
    directories = [Path(path).resolve() for path in directories]
    if weights is None:
        weights = [1.0] * len(directories)
    if not directories or len(weights) != len(directories) or not np.isfinite(weights).all():
        raise ValueError("Supply one finite weight per input directory")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/"embeddings.npy").exists() or output in directories:
        raise FileExistsError(output)
    ids = np.load(directories[0]/"client_ids.npy")
    matrices = []
    names = []
    blocks = []
    width = 0
    for directory, weight in zip(directories, weights):
        matrix = np.load(directory/"embeddings.npy", mmap_mode="r")
        if matrix.ndim != 2 or matrix.dtype != np.float16 or len(matrix) != len(ids):
            raise ValueError(f"Expected aligned float16 vectors in {directory}")
        if not np.array_equal(np.load(directory/"client_ids.npy"), ids):
            raise ValueError(f"Client order differs in {directory}")
        columns = json.loads((directory/"features.json").read_text(encoding="utf-8-sig"))
        if len(columns) != matrix.shape[1]:
            raise ValueError(f"Feature list differs from matrix width in {directory}")
        names.extend(columns)
        matrices.append(matrix)
        blocks.append({"source": directory.name, "start": width, "stop": width + matrix.shape[1],
                       "weight": weight})
        width += matrix.shape[1]
    if width > 2048:
        raise ValueError(f"Official dimension limit exceeded: {width}")
    partial = output/"combined.partial.npy"
    result = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float16, shape=(len(ids),width))
    for start in range(0, len(ids), 8192):
        rows = slice(start, start+8192)
        for block, matrix in zip(blocks, matrices):
            values = matrix[rows]
            if block["weight"] != 1.0:
                adjusted = values.astype(np.float32) * block["weight"]
                if not np.isfinite(adjusted).all() or np.max(np.abs(adjusted)) > np.finfo(np.float16).max:
                    raise ValueError(f"Scaled features exceed float16 range in {block['source']}")
                values = adjusted
            result[rows, block["start"]:block["stop"]] = values
    del result
    np.save(output/"client_ids.npy", ids)
    (output/"features.json").write_text(json.dumps(names),encoding="utf-8")
    partial.replace(output/"embeddings.npy")
    record = dict(clients=len(ids), dimensions=width, dtype="float16", concatenated_blocks=blocks,
                  transformation=("column concatenation without numeric conversion" if all(w == 1 for w in weights)
                                  else "weighted blocks, float32 multiply followed by float16 storage"),
                  seconds=round(perf_counter()-started,3))
    (output/"manifest.json").write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding="utf-8")
    print(record,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directories", type=Path, nargs="+", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--weights", type=float, nargs="+", help="Multiplier for each input block, in input order")
    args = parser.parse_args()
    combine(args.input_directories,args.output_directory,args.weights)
