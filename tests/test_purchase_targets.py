"""Regression test for chronological supervision, held-out users and embedding row alignment."""
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from extension.purchase_targets import build, purchase_labels


def test_historical_heads_use_training_users_and_binary_future_labels(tmp_path):
    data = tmp_path / "data"
    cache = tmp_path / "cache"
    (data / "input").mkdir(parents=True)
    cache.mkdir()
    generator = np.random.default_rng(8)
    ids = generator.permutation(np.arange(90)) + 500
    validation = np.arange(len(ids)) % 5 == 0
    windows = []
    for name, cutoff, start, end in [
        ("earlier", "2022-10-12", "2022-10-13", "2022-10-26"),
        ("later", "2022-10-26", "2022-10-27", "2022-11-09"),
    ]:
        windows.append({"feature_window": name, "target_window": name, "forecast_gap_days": 0,
                        "input_cutoff_utc": cutoff, "target_start_inclusive_utc": start,
                        "target_end_inclusive_utc": end})
    (cache / "manifest.json").write_text(json.dumps({"windows": windows}), encoding="utf-8")
    np.save(data / "input/relevant_clients.npy", ids)
    np.save(cache / "client_ids.npy", ids)
    np.save(cache / "validation_users.npy", validation)
    raw = generator.normal(size=(len(ids), 8)).astype(np.float32)
    for name in ("earlier", "later", "full"):
        (cache / name).mkdir()
        np.save(cache / name / "client_ids.npy", ids)
        np.save(cache / name / "embeddings.npy", raw)
    pd.DataFrame({"sku": [1, 2, 3, 18, 19], "category": [21, 22, 23, 38, 39]}).to_parquet(
        data / "product_properties.parquet", index=False)
    logs = []
    for day in ("2022-10-15", "2022-11-03"):
        for row, client in enumerate(ids):
            sku = 1 + row % 3
            logs += [(client, day, sku)] * 2
            if validation[row]:
                logs += [(client, day, 18)] * 4
    logs += [(ids[1], "2022-11-12", 19)] * 100
    frame = pd.DataFrame(logs, columns=["client_id", "timestamp", "sku"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame.to_parquet(data / "input/product_buy.parquet", index=False)
    classes, targets, lines = purchase_labels(data, ids, validation, windows, {"sku": 10, "category": 10}, 2)
    assert set(classes["sku"].tolist()) == {1, 2, 3}
    assert set(classes["category"].tolist()) == {21, 22, 23}
    assert lines == (~validation).sum() * 4
    assert np.array_equal(targets["sku"][0].sum(axis=1).A1, np.ones(len(ids)))
    with threadpool_limits(limits=1):
        result = build(data, cache, tmp_path / "vectors", 10, 10, 2, rank=2, batch_size=13)
    matrix = np.load(tmp_path / "vectors/embeddings.npy")
    assert not result["official_target_directory_read"]
    assert matrix.shape == (len(ids), 4) and matrix.dtype == np.float16
    assert np.isfinite(matrix).all()
    assert np.array_equal(np.load(tmp_path / "vectors/client_ids.npy"), ids)
    for field in ("sku", "category"):
        assert result["diagnostics"][field]["classes_with_heldout_positives"] == 3
