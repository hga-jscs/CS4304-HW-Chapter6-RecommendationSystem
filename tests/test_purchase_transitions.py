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
