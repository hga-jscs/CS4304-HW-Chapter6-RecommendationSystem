"""Post-hoc DuckDB query of purchase continuation; writes aggregate counts only."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

QUERY = """
WITH buyers AS MATERIALIZED (
    SELECT h.client_id, max(h.timestamp) AS last_buy,
           CAST(min(h.timestamp) AS DATE) = CAST(max(h.timestamp) AS DATE) AS one_day
    FROM purchases AS h SEMI JOIN relevant_clients USING (client_id)
    WHERE h.timestamp < TIMESTAMP '2022-11-10 00:00:00'
    GROUP BY h.client_id
), later_activity AS (
    SELECT DISTINCT e.client_id
    FROM (
        SELECT client_id, timestamp FROM visits
        UNION ALL SELECT client_id, timestamp FROM searches
        UNION ALL SELECT client_id, timestamp FROM additions
    ) AS e
    JOIN buyers AS b ON b.client_id = e.client_id AND e.timestamp > b.last_buy
    WHERE e.timestamp < TIMESTAMP '2022-11-10 00:00:00'
), returns AS (
    SELECT DISTINCT client_id FROM validation_purchases
    WHERE timestamp >= TIMESTAMP '2022-11-24 00:00:00'
      AND timestamp < TIMESTAMP '2022-12-08 00:00:00'
), user_groups AS (
    SELECT CASE WHEN b.one_day THEN 'single_day' ELSE 'multiple_days' END AS cohort,
           CASE WHEN date_diff('day', b.last_buy::DATE, DATE '2022-11-09') <= 14 THEN '0-14'
                WHEN date_diff('day', b.last_buy::DATE, DATE '2022-11-09') <= 28 THEN '15-28'
                WHEN date_diff('day', b.last_buy::DATE, DATE '2022-11-09') <= 56 THEN '29-56'
                ELSE '57+' END AS age,
           e.client_id IS NOT NULL AS continued,
           r.client_id IS NOT NULL AS returned
    FROM buyers AS b
    LEFT JOIN later_activity AS e ON e.client_id = b.client_id
    LEFT JOIN returns AS r ON r.client_id = b.client_id
)
SELECT cohort,
       CASE WHEN GROUPING(age) = 1 THEN 'all' ELSE age END AS recency_days,
       continued, count(*) AS buyers,
       count(*) FILTER (WHERE returned) AS repeat_buyers
FROM user_groups
GROUP BY GROUPING SETS ((cohort, continued), (cohort, age, continued))
HAVING GROUPING(age) = 1 OR cohort = 'single_day'
ORDER BY cohort, recency_days, continued
"""


def independent_counts(data, ids):
    """Derive one vs multiple purchase dates from per-client date extrema with Arrow and NumPy."""
    sorted_ids = np.sort(ids)
    if len(np.unique(sorted_ids)) != len(ids):
        raise ValueError("Expected unique relevant client IDs")
    latest = np.full(len(ids), -1, np.int32)
    earliest = np.full(len(ids), np.iinfo(np.int32).max, np.int32)
    returned = np.zeros(len(ids), dtype=bool)
    min_target = np.datetime64("2022-11-24", "D").astype(np.int32)
    max_target = np.datetime64("2022-12-07", "D").astype(np.int32)
    last_history = np.datetime64("2022-11-09", "D").astype(np.int32)
    for is_target, path in [
        (False, data / "input" / "product_buy.parquet"),
        (True, data / "target" / "validation_target.parquet"),
    ]:
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=150000, columns=["client_id", "timestamp"], use_threads=False
        ):
            users = batch.column(0).to_numpy()
            days = batch.column(1).to_numpy().astype("datetime64[D]").astype(np.int32)
            positions = sorted_ids.searchsorted(users)
            keep = (positions < len(ids)) & (sorted_ids[np.minimum(positions, len(ids) - 1)] == users)
            if is_target:
                if np.any((days < min_target) | (days > max_target)):
                    raise ValueError("Validation purchases outside November 24 to December 7")
                returned[positions[keep]] = True
            else:
                keep &= days <= last_history
                np.maximum.at(latest, positions[keep], days[keep])
                np.minimum.at(earliest, positions[keep], days[keep])
    buyers = latest >= 0
    single = buyers & (earliest == latest)
    counts = {}
    for name, mask in (("all", buyers), ("single_day", single), ("multiple_days", buyers & ~single)):
        counts[name] = {"buyers": int(mask.sum()), "repeat_buyers": int(np.sum(returned & mask))}
    return counts


def audit(data_directory, output):
    import duckdb

    data = Path(data_directory).resolve()
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    ids = np.load(data / "input" / "relevant_clients.npy", allow_pickle=False)
    sources = {
        "purchases": "input/product_buy.parquet",
        "visits": "input/page_visit.parquet",
        "additions": "input/add_to_cart.parquet",
        "searches": "input/search_query.parquet",
        "validation_purchases": "target/validation_target.parquet",
    }
    started = perf_counter()
    with TemporaryDirectory(prefix="purchase_query_", dir=output.parent) as temp:
        temp = Path(temp)
        with duckdb.connect(
            config={
                "threads": "2",
                "memory_limit": "512MiB",
                "temp_directory": str(temp),
                "max_temp_directory_size": "4GiB",
                "preserve_insertion_order": "false",
            }
        ) as db:
            for view, relpath in sources.items():
                db.read_parquet(str(data / relpath)).create_view(view)
            db.register("relevant_clients", pa.table({"client_id": ids}))
            profile = temp / "profile.json"
            db.execute("SET enable_profiling = 'json'")
            db.execute("SET profiling_output = ?", [str(profile)])
            begin = perf_counter()
            rows = db.execute(QUERY).fetchall()
            query_seconds = perf_counter() - begin
        profiler = json.loads(profile.read_text(encoding="utf-8"))
    strata = [
        dict(cohort=c, recency_days=a, continued=bool(v), buyers=n, repeat_buyers=r, repeat_rate=r / n)
        for c, a, v, n, r in rows
    ]
    counts = independent_counts(data, ids)
    for cohort in counts:
        selected = [
            row
            for row in strata
            if row["recency_days"] == "all" and (cohort == "all" or row["cohort"] == cohort)
        ]
        for key in ("buyers", "repeat_buyers"):
            if sum(row[key] for row in selected) != counts[cohort][key]:
                raise ValueError(f"DuckDB and independent buyer counts differ for {cohort} {key}")
    for post in (False, True):
        split = [row for row in strata if row["cohort"] == "single_day" and row["continued"] == post]
        for key in ("buyers", "repeat_buyers"):
            if sum(row[key] for row in split if row["recency_days"] != "all") != sum(
                row[key] for row in split if row["recency_days"] == "all"
            ):
                raise ValueError("Single-day recency strata failed reconciliation")
    record = dict(
        scope="Post-hoc aggregate description only. Public validation outcomes are accessed "
        "after representation design is frozen and never enter the training pipeline.",
        historical_cutoff="2022-11-09 inclusive",
        validation="2022-11-24 through 2022-12-07 inclusive",
        cohort="Relevant clients with at least one historical purchase. Distinct dates define single-day buyers.",
        continuation="At least one historical page visit, search or cart addition strictly after the last "
        "purchase timestamp. Later on the same day counts; simultaneous timestamps do not.",
        days_since_purchase="Calendar-day difference from 2022-11-09; bins are inclusive.",
        repurchase="At least one validation purchase of any product. Denominator is all buyers in the same stratum.",
        inputs=sources,
        independent_arrow_counts=counts,
        results=strata,
        execution=dict(
            duckdb=duckdb.__version__,
            threads=2,
            memory_limit_mib=512,
            max_temp_disk_gib=4,
            query_wall_seconds=round(query_seconds, 3),
            total_wall_seconds_with_reconciliation=round(perf_counter() - started, 3),
            peak_duckdb_buffer_bytes=profiler.get("system", {}).get(
                "peak_buffer_memory", profiler.get("system_peak_buffer_memory")
            ),
            peak_duckdb_temp_disk_bytes=profiler.get("system", {}).get(
                "peak_temp_dir_size", profiler.get("system_peak_temp_dir_size")
            ),
            remark="DuckDB's profiled buffer memory excludes Python and selected engine allocations.",
        ),
    )
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.data_directory, args.output), indent=2))
