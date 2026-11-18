"""Append ready-to-evaluate float16 feature blocks, preserving their values."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np


def combine(directories, output):
    started = perf_counter()
    directories = [Path(path).resolve() for path in directories]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/"embeddings.npy").exists() or output in directories:
        raise FileExistsError(output)
    ids = np.load(directories[0]/"client_ids.npy")
    matrices = []
    names = []
    blocks = []
    width = 0
    for directory in directories:
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
        blocks.append({"source": directory.name, "start": width, "stop": width + matrix.shape[1]})
        width += matrix.shape[1]
    if width > 2048:
        raise ValueError(f"Official dimension limit exceeded: {width}")
    partial = output/"combined.partial.npy"
    result = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float16, shape=(len(ids),width))
    for start in range(0, len(ids), 8192):
        rows = slice(start, start+8192)
        for block, matrix in zip(blocks, matrices):
            result[rows, block["start"]:block["stop"]] = matrix[rows]
    del result
    np.save(output/"client_ids.npy", ids)
    (output/"features.json").write_text(json.dumps(names),encoding="utf-8")
    partial.replace(output/"embeddings.npy")
    record = dict(clients=len(ids), dimensions=width, dtype="float16", concatenated_blocks=blocks,
                  transformation="column concatenation without numeric conversion", seconds=round(perf_counter()-started,3))
    (output/"manifest.json").write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding="utf-8")
    print(record,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directories", type=Path, nargs="+", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    combine(args.input_directories,args.output_directory)
