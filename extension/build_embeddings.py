"""Build bounded-memory Synerise user representations from historical inputs only."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as parquet

from .standardize import write_embeddings


EVENTS = ("product_buy", "add_to_cart", "remove_from_cart", "page_visit", "search_query")
PRODUCT_EVENTS = EVENTS[:3]
STATISTICS = ("count", "count_1_day", "count_7_days", "count_30_days",
              "decay_7_days", "decay_30_days", "recency_days", "span_days")
DAY_NS = 86_400_000_000_000
DOWNLOAD_URL = "https://data.recsys.synerise.com/dataset/challenge_dataset.tar.gz"


PRICE_STATISTICS = ("price_bucket_mean", "price_bucket_max", "price_bucket_std",
                    "price_bucket_mean_30_days", "price_observations",
                    "price_band_0", "price_band_1", "price_band_2", "price_band_3", "price_band_4")
CROSS_STATISTICS = ("buy_to_cart_count_ratio", "remove_to_cart_count_ratio", "buy_to_search_count_ratio",
                    "page_views_per_span_day", "mean_purchase_interval_days", "purchase_recency_to_interval_ratio")


def feature_layout(mode: str = "baseline", popular: dict | None = None) -> tuple[list[str], dict]:
    names = [f"{event}.{statistic}" for event in EVENTS for statistic in STATISTICS]
    blocks: dict[str, dict[str, tuple[int, int]]] = {}
    for event in EVENTS:
        sizes = ({"sku": 16, "category": 16, "price": 8} if mode == "baseline" else
                 {"sku": 16, "category": 16}) if event in PRODUCT_EVENTS else (
            {"url": 16} if event == "page_visit" else {"query": 32})
        blocks[event] = {}
        for column, width in sizes.items():
            blocks[event][column] = (len(names), width)
            names.extend(f"{event}.{column}_bucket_{i}" for i in range(width))
    if mode == "enhanced":
        for event in PRODUCT_EVENTS:
            for column in ("sku", "category"):
                values = popular[event][column]
                blocks[event]["popular_" + column] = (len(names), len(values))
                names.extend(f"{event}.popular_{column}_{value}" for value in values)
            blocks[event]["price_statistics"] = (len(names), len(PRICE_STATISTICS))
            names.extend(f"{event}.{name}" for name in PRICE_STATISTICS)
        names.extend(f"{event}.has_history" for event in EVENTS)
        names.extend(CROSS_STATISTICS)
    return names, blocks


def popular_history(properties: pd.DataFrame, paths: list[Path], batch_size: int, top_k: int) -> dict:
    """Choose identifiers using purchase/cart inputs only, including historical non-target users."""
    counts = {"sku": Counter(), "category": Counter()}
    for path in paths[:2]:
        for batch in parquet.ParquetFile(path).iter_batches(batch_size=batch_size, columns=["sku"]):
            skus = batch.to_pandas()["sku"]
            counts["sku"].update(skus.value_counts().to_dict())
            counts["category"].update(skus.map(properties["category"]).value_counts().to_dict())
    ranked = {column: [int(identifier) for identifier, _ in sorted(values.items(),
               key=lambda pair: (-pair[1], pair[0]))[:top_k]] for column, values in counts.items()}
    # Purchases get the largest exact block; cart/removal retain smaller, separate blocks.
    return {event: {column: values[:max(1, top_k // divisor)] for column, values in ranked.items()}
            for event, divisor in zip(PRODUCT_EVENTS, (1, 2, 4))}


def add_price_statistics(matrix: np.ndarray, rows: np.ndarray, prices: np.ndarray, ages: np.ndarray,
                         weights: np.ndarray, offset: int, recent_counts: np.ndarray) -> None:
    valid = np.isfinite(prices) & (prices >= 0)
    rows, prices, ages, weights = rows[valid], prices[valid], ages[valid], weights[valid]
    np.add.at(matrix[:, offset], rows, prices)
    np.maximum.at(matrix[:, offset + 1], rows, prices)
    np.add.at(matrix[:, offset + 2], rows, prices ** 2)
    recent = ages <= 30
    np.add.at(matrix[:, offset + 3], rows[recent], prices[recent])
    np.add.at(recent_counts, rows[recent], 1)
    np.add.at(matrix[:, offset + 4], rows, 1)
    # Price values are quantile-bucket indices, not currency amounts.
    bands = np.minimum((prices // 20).astype(np.int64), 4)
    np.add.at(matrix, (rows, offset + 5 + bands), weights)


def finish_price_statistics(matrix: np.ndarray, offset: int, recent_counts: np.ndarray,
                            batch_size: int) -> None:
    for start in range(0, len(matrix), batch_size):
        block = matrix[start:start + batch_size]
        count = np.maximum(block[:, offset + 4], 1)
        mean = block[:, offset] / count
        variance = np.maximum(block[:, offset + 2] / count - mean ** 2, 0)
        block[:, offset] = mean
        block[:, offset + 2] = np.sqrt(variance)
        block[:, offset + 3] /= np.maximum(recent_counts[start:start + batch_size], 1)


def add_cross_statistics(matrix: np.ndarray, names: list[str], batch_size: int) -> None:
    lookup = {name: index for index, name in enumerate(names)}
    for start in range(0, len(matrix), batch_size):
        block = matrix[start:start + batch_size]
        for event in EVENTS:
            block[:, lookup[f"{event}.has_history"]] = block[:, lookup[f"{event}.count"]] > 0
        buy = block[:, lookup["product_buy.count"]]
        cart = block[:, lookup["add_to_cart.count"]]
        remove = block[:, lookup["remove_from_cart.count"]]
        search = block[:, lookup["search_query.count"]]
        views = block[:, lookup["page_visit.count"]]
        page_span = block[:, lookup["page_visit.span_days"]]
        eligible = buy >= 2
        interval = np.zeros(len(block), dtype=np.float32)
        interval[eligible] = block[eligible, lookup["product_buy.span_days"]] / (buy[eligible] - 1)
        relative_recency = np.zeros(len(block), dtype=np.float32)
        relative_recency[eligible] = block[eligible, lookup["product_buy.recency_days"]] / (interval[eligible] + 1)
        values = (buy / (cart + 1), remove / (cart + 1), buy / (search + 1),
                  views / (page_span + 1), interval, relative_recency)
        for name, values in zip(CROSS_STATISTICS, values):
            block[:, lookup[name]] = values


def hash_buckets(values: np.ndarray, width: int, salt: int = 0) -> np.ndarray:
    """Stable integer hashing; independent of Python's randomized hash seed."""
    values = np.asarray(values, dtype=np.uint64)
    with np.errstate(over="ignore"):
        mixed = values + np.uint64(0x9E3779B97F4A7C15) + np.uint64(salt)
        mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        mixed = mixed ^ (mixed >> np.uint64(31))
    return (mixed % np.uint64(width)).astype(np.int64)


@lru_cache(maxsize=32768)
def parse_query(text: str) -> tuple[int, ...]:
    """Accept official space-separated PQ codes, and equivalent comma-separated lists."""
    text = text.strip()
    if not (text.startswith("[") and text.endswith("]")):
        raise ValueError("Search query must contain a bracketed integer array")
    inner = text[1:-1].strip()
    if not inner or re.fullmatch(r"[0-9,\s]+", inner) is None:
        raise ValueError("Search query has invalid or empty product-quantization codes")
    codes = tuple(int(value) for value in re.split(r"[,\s]+", inner) if value)
    if not codes or any(value > 255 for value in codes):
        raise ValueError("Search query product-quantization codes must lie in [0, 255]")
    return codes


def timestamps_ns(values: pd.Series) -> np.ndarray:
    parsed = pd.to_datetime(values, utc=True, errors="raise")
    if parsed.isna().any():
        raise ValueError("Input events contain missing timestamps")
    return parsed.to_numpy(dtype="datetime64[ns]").view(np.int64)


def input_cutoff(paths: list[Path], batch_size: int) -> int:
    """Read only historical Parquet metadata/columns, never a target window."""
    maxima = []
    for path in paths:
        file = parquet.ParquetFile(path)
        column_index = file.schema_arrow.names.index("timestamp")
        statistics = [file.metadata.row_group(i).column(column_index).statistics
                      for i in range(file.metadata.num_row_groups)]
        if statistics and all(value is not None and value.has_min_max for value in statistics):
            maxima.append(max(pd.Timestamp(value.max).value for value in statistics))
        else:
            for batch in file.iter_batches(batch_size=batch_size, columns=["timestamp"]):
                if batch.num_rows:
                    maxima.append(int(timestamps_ns(batch.to_pandas()["timestamp"]).max()))
    if not maxima:
        raise ValueError("No historical timestamps were found in input/")
    return max(maxima)


def validate_inputs(data_directory: Path) -> tuple[np.ndarray, list[Path]]:
    input_directory = data_directory / "input"
    ids = np.load(input_directory / "relevant_clients.npy", allow_pickle=False)
    if ids.ndim != 1 or ids.dtype != np.int64 or len(ids) == 0:
        raise ValueError("relevant_clients.npy must be a non-empty one-dimensional int64 array")
    if pd.Index(ids).has_duplicates:
        raise ValueError("relevant_clients.npy contains duplicate client IDs")
    paths = [input_directory / f"{event}.parquet" for event in EVENTS]
    for event, path in zip(EVENTS, paths):
        file = parquet.ParquetFile(path)
        payload = "sku" if event in PRODUCT_EVENTS else "url" if event == "page_visit" else "query"
        missing = {"client_id", "timestamp", payload} - set(file.schema_arrow.names)
        if missing:
            raise ValueError(f"{path.name} lacks required columns: {sorted(missing)}")
    return ids, paths


def build_embeddings(data_directory: str | Path, output_directory: str | Path,
                     batch_size: int = 100_000, decay_days: float = 30.0,
                     mode: str = "baseline", standardize: bool = False, top_k: int = 128,
                     overwrite: bool = False) -> dict:
    """Create one representation for every relevant client in their original order.

    Aggregates use bounded Arrow batches and a disk-backed float32 matrix.
    Output is float16 (~416 MB for baseline, ~1.35 GB for default enhanced).
    target/ is never opened.
    """
    if batch_size < 1 or decay_days < 0 or not np.isfinite(decay_days) or top_k < 1:
        raise ValueError("batch_size must be positive and decay_days finite and non-negative")
    if mode not in {"baseline", "enhanced"}:
        raise ValueError("mode must be baseline or enhanced")
    started = time.perf_counter()
    data_directory, output_directory = Path(data_directory), Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    reserved = ("client_ids.npy", "embeddings.npy", "features.json", "manifest.json", "standardization.json")
    if not overwrite and any((output_directory / name).exists() for name in reserved):
        raise FileExistsError("Output files already exist; use --overwrite to replace this experiment")
    if overwrite:
        (output_directory / "manifest.json").unlink(missing_ok=True)
        (output_directory / "standardization.json").unlink(missing_ok=True)
    ids, paths = validate_inputs(data_directory)
    properties_path = data_directory / "product_properties.parquet"
    properties = pd.read_parquet(properties_path, columns=["sku", "category", "price"])
    if properties["sku"].isna().any() or properties["sku"].duplicated().any():
        raise ValueError("Product properties must have unique, non-missing sku values")
    properties = properties.set_index("sku")
    client_index = pd.Index(ids)
    popular = popular_history(properties, paths, batch_size, top_k) if mode == "enhanced" else None
    names, blocks = feature_layout(mode, popular)
    if len(names) > 2048:
        raise ValueError("Representation exceeds the official 2048-dimension limit; reduce top_k")
    scratch = output_directory / "features.partial.npy"
    matrix = np.lib.format.open_memmap(scratch, mode="w+", dtype=np.float32,
                                      shape=(len(ids), len(names)), fortran_order=True)
    matrix[:] = 0
    cutoff = input_cutoff(paths, batch_size)
    summary = {}
    source_manifest = []
    query_width = None
    for event_number, (event, path) in enumerate(zip(EVENTS, paths)):
        file = parquet.ParquetFile(path)
        payload = "sku" if event in PRODUCT_EVENTS else "url" if event == "page_visit" else "query"
        first = np.full(len(ids), np.iinfo(np.int64).max, dtype=np.int64)
        last = np.full(len(ids), np.iinfo(np.int64).min, dtype=np.int64)
        selected_rows = 0
        observed_rows = 0
        recent_price_counts = np.zeros(len(ids), dtype=np.float32) if mode == "enhanced" and event in PRODUCT_EVENTS else None
        event_start = time.perf_counter()
        for batch in file.iter_batches(batch_size=batch_size, columns=["client_id", "timestamp", payload]):
            frame = batch.to_pandas()
            observed_rows += len(frame)
            rows = client_index.get_indexer(frame["client_id"])
            selected = rows >= 0
            if not selected.any():
                continue
            frame, rows = frame.loc[selected], rows[selected]
            selected_rows += len(frame)
            times = timestamps_ns(frame["timestamp"])
            ages = (cutoff - times) / DAY_NS
            if (ages < 0).any():
                raise ValueError("An input event is later than the recorded input cutoff")
            base = event_number * len(STATISTICS)
            for offset, values in enumerate((np.ones(len(rows), dtype=np.float32),
                                             ages <= 1, ages <= 7, ages <= 30,
                                             np.exp2(-ages / 7), np.exp2(-ages / 30))):
                np.add.at(matrix[:, base + offset], rows, values)
            np.minimum.at(first, rows, times)
            np.maximum.at(last, rows, times)
            weights = np.exp2(-ages / decay_days).astype(np.float32) if decay_days else np.ones(len(rows), np.float32)
            if event in PRODUCT_EVENTS:
                metadata = properties.reindex(frame["sku"].to_numpy())
                for column in (("sku", "category", "price") if mode == "baseline" else ("sku", "category")):
                    values = frame["sku"].to_numpy() if column == "sku" else metadata[column].to_numpy()
                    valid = ~pd.isna(values)
                    if mode == "enhanced":
                        exact_codes = pd.Index(popular[event][column]).get_indexer(values)
                        exact = valid & (exact_codes >= 0)
                        exact_offset, _ = blocks[event]["popular_" + column]
                        np.add.at(matrix, (rows[exact], exact_offset + exact_codes[exact]), weights[exact])
                        valid &= ~exact
                    offset, width = blocks[event][column]
                    buckets = hash_buckets(values[valid], width)
                    np.add.at(matrix, (rows[valid], offset + buckets), weights[valid])
                if mode == "enhanced":
                    price_offset, _ = blocks[event]["price_statistics"]
                    prices = pd.to_numeric(metadata["price"], errors="coerce").to_numpy(dtype=np.float64)
                    add_price_statistics(matrix, rows, prices, ages, weights, price_offset, recent_price_counts)
            elif event == "page_visit":
                valid = frame["url"].notna().to_numpy()
                offset, width = blocks[event]["url"]
                buckets = hash_buckets(frame["url"].to_numpy()[valid], width)
                np.add.at(matrix, (rows[valid], offset + buckets), weights[valid])
            else:
                offset, width = blocks[event]["query"]
                queries = frame["query"].tolist()
                parsed = [parse_query(value) if isinstance(value, str) else
                          tuple(int(code) for code in value) for value in queries]
                lengths = np.fromiter((len(value) for value in parsed), dtype=np.int64)
                if (lengths < 1).any() or len(set(lengths.tolist())) != 1:
                    raise ValueError("Search query arrays must be non-empty with a consistent length")
                if query_width is not None and int(lengths[0]) != query_width:
                    raise ValueError("Search query array length changes across input batches")
                query_width = int(lengths[0])
                codes = np.asarray(parsed, dtype=np.int64)
                if ((codes < 0) | (codes > 255)).any():
                    raise ValueError("Search query product-quantization codes must lie in [0, 255]")
                # PQ code coordinates are categorical; no ordinal mean of bucket IDs.
                keys = codes + np.arange(codes.shape[1], dtype=np.int64)[None, :] * 256
                buckets = hash_buckets(keys.reshape(-1), width)
                repeated_rows = np.repeat(rows, codes.shape[1])
                repeated_weights = np.repeat(weights / codes.shape[1], codes.shape[1])
                np.add.at(matrix, (repeated_rows, offset + buckets), repeated_weights)
        base = event_number * len(STATISTICS)
        present = matrix[:, base] > 0
        matrix[:, base + 6] = 366.0  # Distinguishes no history from a very recent event.
        matrix[present, base + 6] = (cutoff - last[present]) / DAY_NS
        matrix[present, base + 7] = (last[present] - first[present]) / DAY_NS
        if mode == "enhanced" and event in PRODUCT_EVENTS:
            finish_price_statistics(matrix, blocks[event]["price_statistics"][0], recent_price_counts,
                                    min(batch_size, 20_000))
        matrix.flush()
        summary[event] = {"input_rows": observed_rows, "relevant_client_rows": selected_rows,
                          "seconds": round(time.perf_counter() - event_start, 3)}
        source_manifest.append({"path": f"input/{path.name}", "bytes": path.stat().st_size,
                                "rows": file.metadata.num_rows})
        print(f"{event}: {selected_rows:,}/{observed_rows:,} rows, {summary[event]['seconds']:.1f}s", flush=True)
    if mode == "enhanced":
        add_cross_statistics(matrix, names, min(batch_size, 20_000))
    normalization = write_embeddings(matrix, output_directory / "embeddings.npy",
                                     batch_size=min(batch_size, 20_000), standardize=standardize,
                                     log_transform=True)
    del matrix
    scratch.unlink()
    np.save(output_directory / "client_ids.npy", ids, allow_pickle=False)
    (output_directory / "features.json").write_text(json.dumps(names, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": ("historical_aggregates_with_hashed_categories" if mode == "baseline" else
                   "historical_popular_items_ordinal_prices_and_behavior_ratios"),
        "mode": mode, "standardized": standardize, "popular_history_identifiers": popular,
        "clients": len(ids), "dimensions": len(names), "dtype": "float16",
        "batch_size": batch_size, "histogram_half_life_days": decay_days,
        "input_cutoff_utc": pd.Timestamp(cutoff, tz="UTC").isoformat(),
        "relevant_clients_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
        "sources": source_manifest + [{"path": properties_path.name, "bytes": properties_path.stat().st_size}],
        "target_data_used_for_embeddings": False, "events": summary,
        "seconds": round(time.perf_counter() - started, 3),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__},
    }
    if standardize:
        (output_directory / "standardization.json").write_text(json.dumps(normalization, indent=2) + "\n", encoding="utf-8")
    (output_directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--decay-days", type=float, default=30.0,
                        help="Half-life of hashed history counts; 0 disables decay")
    parser.add_argument("--mode", choices=("baseline", "enhanced"), default="baseline")
    parser.add_argument("--standardize", action="store_true", help="Standardize log features by column and clip to [-10, 10]")
    parser.add_argument("--top-k", type=int, default=128,
                        help="Maximum history-selected SKU/category identifiers; cart/removal use half/quarter")
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    result = build_embeddings(**vars(arguments))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
