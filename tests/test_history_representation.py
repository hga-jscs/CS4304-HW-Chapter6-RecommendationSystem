"""Check compact block semantics and client/target isolation."""
import json

import numpy as np
import pandas as pd
import pytest

from extension.build_embeddings import EVENTS
from extension.history_representation import build_history, code_matrix, combine


def fixture_data(tmp_path):
    data = tmp_path / "data"
    (data / "input").mkdir(parents=True)
    np.save(data / "input/relevant_clients.npy", np.array([9, 3, 20], dtype=np.int64))
    pd.DataFrame({"sku": [100, 200], "category": [7, 8], "price": [10, 30],
                  "name": ["[1 2 3 4]", "[4 3 2 1]"]}).to_parquet(data / "product_properties.parquet")
    for event in EVENTS:
        frame = pd.DataFrame({"client_id": [9, 9, 3],
                              "timestamp": pd.to_datetime(["2022-10-01", "2022-10-11", "2022-10-31"])})
        if event in EVENTS[:3]:
            frame["sku"] = [100, 200, 100]
        elif event == "page_visit":
            frame["url"] = [4, 5, 4]
        else:
            frame["query"] = ["[1 2 3 4]", "[4 3 2 1]", "[1 2 3 4]"]
        frame.to_parquet(data / "input" / f"{event}.parquet", index=False)
    return data


def test_compact_history_counts_prices_intervals_and_no_target_access(tmp_path):
    data = fixture_data(tmp_path)
    # A malformed target file would fail if the representation tried reading it.
    (data / "target").mkdir()
    (data / "target/validation_target.parquet").write_text("must not be opened")
    output = tmp_path / "history"
    result = build_history(data, output, batch_size=2)
    x = np.load(output / "embeddings.npy")
    names = json.loads((output / "features.json").read_text())
    assert x.shape == (3, 544)
    np.testing.assert_array_equal(np.load(output / "client_ids.npy"), [9, 3, 20])
    # User9 has two purchases ten days apart, observed20 days after last purchase.
    assert np.expm1(x[0, names.index("mean_purchase_interval_days")]) == pytest.approx(10)
    assert np.expm1(x[0, names.index("purchase_recency_to_interval_ratio")]) == pytest.approx(20 / 11)
    assert np.expm1(x[0, names.index("product_buy.price_bucket_mean")]) == pytest.approx(20)
    assert np.expm1(x[0, names.index("product_buy.price_bucket_std")]) == pytest.approx(10)
    assert np.expm1(x[0, names.index("product_buy.price_bucket_sum_30_days")]) == pytest.approx(40)
    assert np.expm1(x[0, names.index("popular_sku_100")]) == pytest.approx(2 * 0.5)
    assert np.isfinite(x).all()
    np.testing.assert_allclose(np.linalg.norm(x[:2, 416:544], axis=1), 1, atol=1e-5)
    np.testing.assert_array_equal(x[2, 416:544], 0)
    assert result["target_data_used_for_embeddings"] is False


def test_position_codes_and_combination_row_identity(tmp_path):
    np.testing.assert_array_equal(code_matrix(["[4 4]"]), [[4, 260]])
    history, core, output = (tmp_path / name for name in ("history", "core", "output"))
    for path in (history, core):
        path.mkdir()
        np.save(path / "client_ids.npy", np.array([9, 3], np.int64))
        (path / "manifest.json").write_text('{}')
    np.save(history / "embeddings.npy", np.ones((2, 544), np.float32))
    np.save(core / "embeddings.npy", np.ones((2, 457), np.float32))
    (history / "features.json").write_text(json.dumps([f"feature_{i}" for i in range(544)]))
    result = combine(history, core, output)
    assert result["dimensions"] == 1001
    assert np.load(output / "embeddings.npy").dtype == np.float16
    np.save(core / "client_ids.npy", np.array([3, 9], np.int64))
    with pytest.raises(ValueError, match="different client order"):
        combine(history, core, output, overwrite=True)
