"""Validate KKBox predictions before sending them through the official Kaggle CLI."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

COMPETITION = "kkbox-music-recommendation-challenge"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_header(path: Path, exact: bool) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    if (exact and header != ["id", "target"]) or (not exact and header.count("id") != 1):
        raise ValueError(f"Invalid CSV header in {path.name}: {header}")


def _ids(path: Path) -> np.ndarray:
    ids = pd.read_csv(path, usecols=["id"], dtype={"id": "int64"})["id"].to_numpy()
    if len(ids) == 0 or len(np.unique(ids)) != len(ids):
        raise ValueError(f"Empty or duplicate IDs in {path.name}")
    return ids


def _prediction_chunks(path: Path):
    # index_col=False prevents pandas silently accepting an extra CSV field as
    # an implicit index. Treat its truncation warning as a validation failure.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.ParserWarning)
            yield from pd.read_csv(path, dtype={"id": "int64", "target": "float64"},
                                   chunksize=200_000, index_col=False)
    except pd.errors.ParserWarning as error:
        raise ValueError("Submission rows must contain exactly two CSV fields") from error


def validate_submission(submission_path, sample_path, test_path=None) -> dict:
    """Check complete coverage, exact sample order and finite probabilities.

    Reads probabilities in chunks. ID arrays use int64 so multi-million-row
    submissions do not require millions of Python strings or a Python set.
    """
    submission_path, sample_path = Path(submission_path), Path(sample_path)
    _check_header(submission_path, exact=True)
    _check_header(sample_path, exact=True)
    expected = _ids(sample_path)
    if test_path is not None:
        test_path = Path(test_path)
        _check_header(test_path, exact=False)
        test_ids = _ids(test_path)
        if len(test_ids) != len(expected) or not np.array_equal(np.sort(test_ids), np.sort(expected)):
            raise ValueError("Sample IDs do not exactly cover test IDs")
        del test_ids
    count, low, high = 0, 1.0, 0.0
    for chunk in _prediction_chunks(submission_path):
        ids = chunk["id"].to_numpy()
        values = chunk["target"].to_numpy()
        if not np.array_equal(ids, expected[count:count + len(chunk)]):
            raise ValueError(f"Submission IDs differ from sample order/coverage at row {count}")
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("Predictions must be finite probabilities between 0 and 1")
        if len(values):
            low, high = min(low, float(values.min())), max(high, float(values.max()))
        count += len(chunk)
    if count != len(expected):
        raise ValueError(f"Expected {len(expected)} rows, got {count}")
    return {"valid": True, "rows": count, "minimum_probability": low,
            "maximum_probability": high, "sha256": file_sha256(submission_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "upload"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--submission", type=Path, required=True)
        subparser.add_argument("--sample", type=Path, required=True)
        subparser.add_argument("--test", type=Path, required=True)
        if name == "upload":
            subparser.add_argument("--message", required=True)
            subparser.add_argument("--record", type=Path)
    subparsers.add_parser("list")
    args = parser.parse_args()
    if args.command == "list":
        subprocess.run([sys.executable, "-m", "kaggle", "competitions", "submissions",
                        "-c", COMPETITION, "--csv"], check=True)
        return
    report = validate_submission(args.submission, args.sample, args.test)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.command == "check":
        return
    record = args.record or args.submission.parent / "submission_history.jsonl"
    if record.exists():
        for line in record.read_text(encoding="utf-8").splitlines():
            previous = json.loads(line)
            if previous.get("sha256") == report["sha256"] and previous.get("upload_returncode") == 0:
                raise ValueError("This exact file was already uploaded; inspect Kaggle submission status first")
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run([sys.executable, "-m", "kaggle", "competitions", "submit",
                             "-c", COMPETITION, "-f", str(args.submission.resolve()),
                             "-m", args.message], encoding="utf-8", errors="replace",
                            capture_output=True, env=environment)
    record.parent.mkdir(parents=True, exist_ok=True)
    report.update({"created_at_utc": datetime.now(timezone.utc).isoformat(),
                   "file": str(args.submission.resolve()), "message": args.message,
                   "competition": COMPETITION, "upload_returncode": result.returncode,
                   "response": result.stdout.strip(), "error": result.stderr.strip()})
    with record.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False) + "\n")
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    if result.returncode:
        raise SystemExit(result.returncode)
    print("Upload returned successfully. Run the list command to verify scoring; no score is inferred here.")


if __name__ == "__main__":
    main()
