"""Check behavior semantics and fixed coordinates across historical windows."""
import json

import numpy as np
import pandas as pd
import pytest

from extension.behavior_features import build_behavior, feature_layout
from extension.build_embeddings import EVENTS


def behavior_data(tmp_path):
    data = tmp_path / "data"
    (data / "input").mkdir(parents=True)
    (data / "target").mkdir()
    (data / "target/validation_target.parquet").write_text("must not be read")
    np.save(data / "input/relevant_clients.npy", np.array([9, 3, 20], np.int64))
    pd.DataFrame({"sku": [100, 200, 300], "category": [7, 8, 9], "price": [10, 30, 40],
                  "name": ["[1 2 3 4]", "[4 3 2 1]", "[8 9 10 11]"]}).to_parquet(data / "product_properties.parquet")
    records = {
        "product_buy": [(9, "2022-01-01 01:00", 100), (9, "2022-01-01 01:00", 200),
                        (9, "2022-01-05 10:00", 100), (3, "2022-01-06 12:00", 100),
                        (9, "2022-01-09 12:00", 300)],
        "add_to_cart": [(9, "2022-01-06 12:00", 100)] + [(9, "2022-01-06 13:00", 300)] * 20,
        "remove_from_cart": [(9, "2022-01-06 15:00", 100)],
        "page_visit": [(9, "2022-01-02", 4), (3, "2022-01-06", 5), (9, "2022-01-09", 999)],
        "search_query": [(9, "2022-01-03", "[1 2 3 4]"), (3, "2022-01-06", "[4 3 2 1]"),
                         (9, "2022-01-09", "[10 11 12 13]")],
    }
    for event, entries in records.items():
        payload = "sku" if event in EVENTS[:3] else "url" if event == "page_visit" else "query"
        frame = pd.DataFrame(entries, columns=["client_id", "timestamp", payload])
        frame.timestamp = pd.to_datetime(frame.timestamp, format="mixed")
        frame.to_parquet(data / "input" / f"{event}.parquet", index=False)
    return data


def test_separate_counts_signed_balance_and_independent_purchase_days(tmp_path):
    data, output = behavior_data(tmp_path), tmp_path / "prefix"
    result = build_behavior(data, output, cutoff_utc="2022-01-07T23:59:59Z", batch_size=2)
    matrix = np.load(output / "embeddings.npy")
    names = json.loads((output / "features.json").read_text())
    values = dict(zip(names, matrix[0]))
    assert matrix.shape == (3, 1203)
    assert matrix.dtype == np.float32
    assert np.isfinite(matrix).all()
    np.testing.assert_array_equal(np.load(output / "client_ids.npy"), [9, 3, 20])
    with np.load(output / "feature_state.npz") as state:
        np.testing.assert_array_equal(state["top_skus"], [100, 200])
    assert "buy_history.sku_300" not in names  # Many cart events cannot select a purchase Top-K.
    assert np.expm1(values["buy_history.sku_100"]) > np.expm1(values["cart_history.sku_100"])
    assert values["cart_balance_7_days.sku_100"] == pytest.approx(-np.log1p(2))
    assert np.expm1(values["purchase_active_days"]) == pytest.approx(2)
    assert np.expm1(values["repeat_purchase_day_observed"]) == pytest.approx(1)
    assert np.expm1(values["mean_purchase_day_interval_days"]) == pytest.approx(4)
    assert np.expm1(values["purchase_day_recency_to_interval_ratio"]) == pytest.approx(2 / 5)
    assert matrix[1, names.index("repeat_purchase_day_observed")] == 0
    assert matrix[1, names.index("mean_purchase_day_interval_days")] == 0
    expected_velocity = (np.expm1(values["product_buy.decay_7_days"])
                         / (np.expm1(values["product_buy.decay_30_days"]) + 0.1))
    assert values["product_buy.decay_7_to_30_ratio"] == pytest.approx(expected_velocity, rel=1e-6)
    for block in result["blocks"]:
        if block["kind"] == "embedding":
            np.testing.assert_array_equal(matrix[2, block["start"]:block["stop"]], 0)
    assert result["target_data_used_for_embeddings"] is False
    assert result["blocks"][-1]["stop"] == 1203
    assert {block["kind"] for block in result["blocks"]} == {
        "statistics", "counts", "signed_count", "embedding"}


def test_reference_freezes_vocabulary_factors_and_graph_coordinates(tmp_path):
    data = behavior_data(tmp_path)
    first, same, later = [tmp_path / name for name in ("prefix", "same", "later")]
    fitted = build_behavior(data, first, cutoff_utc="2022-01-07T23:59:59Z", batch_size=2)
    reapplied = build_behavior(data, same, cutoff_utc="2022-01-07T23:59:59Z", reference_directory=first, batch_size=3)
    future = build_behavior(data, later, cutoff_utc="2022-01-10T23:59:59Z", reference_directory=first, batch_size=2)
    assert fitted["state_sha256"] == reapplied["state_sha256"] == future["state_sha256"]
    assert fitted["fit_cutoff_utc"] == future["fit_cutoff_utc"]
    assert future["input_cutoff_utc"] != future["fit_cutoff_utc"]
    assert future["reference_used"]
    assert (first / "features.json").read_bytes() == (later / "features.json").read_bytes()
    np.testing.assert_allclose(np.load(first / "embeddings.npy"), np.load(same / "embeddings.npy"), atol=2e-6)
    with np.load(later / "feature_state.npz") as state:
        assert 300 not in state["top_skus"]
        assert 999 not in state["top_urls"]
        assert 999 not in state["graph_urls"]
    with pytest.raises(ValueError, match="after the requested"):
        build_behavior(data, tmp_path / "invalid", cutoff_utc="2022-01-06", reference_directory=first)


def test_future_events_do_not_change_fitted_prefix_or_optional_graph_controls(tmp_path):
    data = behavior_data(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    fitted = build_behavior(data, first, cutoff_utc="2022-01-07T23:59:59Z", batch_size=2,
                            graph_idf=False, graph_event_weights=False)
    for event in EVENTS:
        path = data / "input" / f"{event}.parquet"
        frame = pd.read_parquet(path)
        future = frame.iloc[:2].copy()
        future.timestamp = pd.Timestamp("2022-01-20")
        pd.concat([frame, future], ignore_index=True).to_parquet(path, index=False)
    build_behavior(data, second, cutoff_utc="2022-01-07T23:59:59Z", batch_size=2,
                   graph_idf=False, graph_event_weights=False)
    np.testing.assert_array_equal(np.load(first / "embeddings.npy"), np.load(second / "embeddings.npy"))
    with np.load(first / "feature_state.npz") as a, np.load(second / "feature_state.npz") as b:
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_array_equal(a["graph_idf"], 1)
        names, blocks = feature_layout(dict(a))
    assert len(names) == blocks[-1]["stop"] == 1203
    assert not fitted["graph_idf"] and not fitted["graph_event_weights"]
