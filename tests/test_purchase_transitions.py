import json

import numpy as np
import pandas as pd

from extension.build_embeddings import DAY_NS
from extension.purchase_transitions import build, next_baskets, purchase_baskets


def test_forward_window_handles_simultaneous_items_and_never_crosses_users():
    users = np.array([1, 0, 0, 0, 0, 0])
    times = np.array([12, 20, 11, 11, 20, 12]) * DAY_NS
    items = np.array([55, 100, 23, 42, 100, 99])
    baskets = purchase_baskets(users, times, items, 3)
    rows, matched = next_baskets(np.array([0, 0, 0, 1, 2, 0, 0]),
                                 np.array([4, 11, 12, 12, 11, 13, 20]) * DAY_NS, baskets)
    np.testing.assert_array_equal(rows, [0, 1, 5])
    np.testing.assert_array_equal(matched, [0, 1, 2])
    np.testing.assert_allclose(baskets.products[0].data, [.5, .5])
    np.testing.assert_allclose(baskets.products[2].data, [1.0])


def test_streamed_graph_counts_distinct_baskets_and_aligns_recent_rows(tmp_path):
    source = tmp_path / "data" / "input"
    source.mkdir(parents=True)
    np.save(source / "relevant_clients.npy", np.array([20, 10, 30], dtype=np.int64))
    def write(name, field, users, days, items):
        frame = pd.DataFrame({"client_id": users, "timestamp": pd.Timestamp("2022-01-01")
                              + pd.to_timedelta(days, unit="D"), field: items})
        frame.to_parquet(source / (name + ".parquet"), index=False)

    write("product_buy", "sku", [10, 10, 10, 20, 10], [3, 3, 12, 12, 29], [55, 56, 56, 56, 70])
    write("add_to_cart", "sku", [10, 10, 10, 20, 20], [1, 1, 11, 11, 25], [9, 9, 9, 9, 9])
    write("page_visit", "url", [10, 10, 10, 10, 10], [1, 1, 11, 24, 28], [1, 1, 1, 1, 2])
    output = tmp_path / "result"
    report = build(source.parent, output, dimensions=3, batch_size=2,
                   cutoff_utc="2022-01-31", threads=1)
    assert report["relations"]["page_visit"]["retained_distinct_source_baskets"] == 2
    assert report["relations"]["add_to_cart"]["retained_distinct_source_baskets"] == 3
    assert report["relations"]["page_visit"]["raw_linked_sources"] == 1
    assert report["relations"]["page_visit"]["directed_source_product_edges"] == 2
    assert report["nonzero_user_vectors"] == 2
    values = np.load(output / "embeddings.npy")
    np.testing.assert_array_equal(np.load(output / "client_ids.npy"), [20, 10, 30])
    np.testing.assert_allclose(np.linalg.norm(values.astype("float32"), axis=1), [1, 1, 0], atol=.001)
    assert np.isfinite(values).all() and values.dtype == np.float16
    assert len(json.loads((output / "features.json").read_text())) == values.shape[1]


def test_purchase_window_includes_exact_days_and_skips_same_basket():
    times = np.array([0, 1, DAY_NS, 28 * DAY_NS, 58 * DAY_NS])
    baskets = purchase_baskets(np.zeros(5, int), times, np.arange(5), 1)
    rows, matched = next_baskets(np.zeros(4, int), np.array([0, 1, DAY_NS, 28 * DAY_NS]),
                                 baskets, minimum_days=1, maximum_days=28)
    np.testing.assert_array_equal(rows, [0, 1, 2])
    np.testing.assert_array_equal(matched, [2, 3, 3])
    rows, matched = next_baskets(np.zeros(2, int), np.array([30 * DAY_NS, 30 * DAY_NS - 1]),
                                 baskets, minimum_days=1, maximum_days=28)
    np.testing.assert_array_equal(rows, [0])
    np.testing.assert_array_equal(matched, [4])


def test_outside_buyer_edges_project_to_users_with_no_purchase(tmp_path):
    directory = tmp_path / "input"
    directory.mkdir()
    np.save(directory / "relevant_clients.npy", np.array([32, 10], np.int64))
    def write(name, field, users, days, nodes):
        pd.DataFrame({"client_id": users, "timestamp": pd.Timestamp("2022-01-01")
                      + pd.to_timedelta(days, unit="D"), field: nodes}).to_parquet(
                          directory / (name + ".parquet"), index=False)
    # Two outside buyers supply both supporting baskets. User 32 only visits late.
    # Day 34 has an observed following basket, but less than 28 days of follow-up,
    # so the graph cannot train on it at this cutoff.
    write("product_buy", "sku", [90, 90, 91, 91, 90, 90], [0, 1, 0, 4, 34, 35],
          [5, 8, 5, 9, 11, 12])
    write("page_visit", "url", [90, 91, 32], [0, 0, 39], [7, 7, 7])
    write("add_to_cart", "sku", [90, 91, 10], [0, 0, 39], [5, 5, 5])
    output = tmp_path / "expanded"
    report = build(tmp_path, output, dimensions=2, batch_size=2, threads=1,
                   cutoff_utc="2022-02-10", all_user_graph=True)
    assert report["additional_buyers_outside_output_cohort"] == 2
    for relation in report["relations"].values():
        assert relation["retained_distinct_source_baskets"] == 2
        assert relation["matches_from_outside_output_cohort"] == 2
    assert report["relations"]["product_buy"]["users_with_retained_recent_sources"] == 0
    np.testing.assert_array_equal(np.load(output / "client_ids.npy"), [32, 10])
    np.testing.assert_allclose(np.linalg.norm(np.load(output / "embeddings.npy"), axis=1),
                               [1, 1], atol=.001)
