"""Input isolation, shared spectral coordinates and separate relation pooling."""
import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from extension.product_cooccurrence import build_product_cooccurrence, product_factors, weighted_history


def example(tmp_path):
    data = tmp_path / "data"
    (data / "input").mkdir(parents=True)
    (data / "target").mkdir()
    (data / "target" / "forbidden.parquet").write_text("Official targets must not be read")
    np.save(data / "input" / "relevant_clients.npy", np.array([9, 3, 20, 30], np.int64))
    buy = [(9, 100, "2022-01-01"), (9, 100, "2022-01-02"), (9, 200, "2022-01-03"),
           (30, 300, "2022-01-01"), (30, 300, "2022-01-02")]
    buy += [(99, 999, "2022-01-01")] * 6 + [(9, 400, "2022-02-01")] * 3
    cart = [(3, 100, "2022-01-03"), (3, 200, "2022-01-04"), (3, 200, "2022-01-05")]
    for event, rows in (("product_buy", buy), ("add_to_cart", cart)):
        frame = pd.DataFrame(rows, columns=["client_id", "sku", "timestamp"])
        frame.timestamp = pd.to_datetime(frame.timestamp)
        frame.to_parquet(data / "input" / f"{event}.parquet", index=False)
    return data


def test_shared_factors_match_exact_small_singular_subspace():
    matrix = sparse.csr_matrix(np.array([[2, 1, 0, 0], [1, 3, 1, 0],
                                        [0, 1, 4, 2], [0, 0, 2, 3]], np.float32))
    factors, values, rank = product_factors(matrix, dimensions=2, oversamples=2, power_iterations=1)
    _, singular, right = np.linalg.svd(matrix.toarray(), full_matrices=False)
    assert rank == 2
    np.testing.assert_allclose(factors @ factors.T, right[:2].T @ right[:2], atol=2e-5)
    np.testing.assert_allclose(factors.T @ factors, np.eye(2), atol=2e-5)
    np.testing.assert_allclose(values, singular[:2], rtol=2e-5)


def test_document_frequency_counts_users_after_duplicate_and_relation_merge():
    counts = sparse.coo_matrix((np.ones(4, np.float32), ([0, 0, 0, 1], [0, 0, 1, 0])), shape=(3, 2)).tocsr()
    matrix, idf, frequency = weighted_history(counts)
    np.testing.assert_array_equal(frequency, [2, 1])
    np.testing.assert_allclose(idf, [1, 1 + np.log(3 / 2)])
    expected = np.array([np.log1p(2), np.log1p(1) * idf[1]])
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(matrix.toarray()[0], expected, rtol=1e-6)
    np.testing.assert_array_equal(matrix.toarray()[2], 0)


def test_frequency_cutoff_relation_separation_decay_and_full_id_coverage(tmp_path, monkeypatch):
    import extension.product_cooccurrence as module
    original = module.parquet.ParquetFile

    def only_product_inputs(path, *args, **kwargs):
        assert path.parent.name == "input" and path.stem in ("product_buy", "add_to_cart")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(module.parquet, "ParquetFile", only_product_inputs)
    data, output = example(tmp_path), tmp_path / "result"
    manifest = build_product_cooccurrence(data, output, cutoff_utc="2022-01-10T00:00:00Z",
                                          batch_size=2, threads=1)
    values = np.load(output / "embeddings.npy")
    np.testing.assert_array_equal(np.load(output / "client_ids.npy"), [9, 3, 20, 30])
    assert values.shape == (4, 128) and values.dtype == np.float16 and np.isfinite(values).all()
    np.testing.assert_array_equal(values[0, 64:], 0)
    np.testing.assert_array_equal(values[1, :64], 0)
    np.testing.assert_array_equal(values[2:], 0)
    assert np.linalg.norm(values[0, :64].astype(np.float32)) == pytest.approx(1, abs=5e-4)
    assert np.linalg.norm(values[1, 64:].astype(np.float32)) == pytest.approx(1, abs=5e-4)
    with np.load(output / "product_factors.npz") as state:
        np.testing.assert_array_equal(state["product_ids"], [100, 200])
        np.testing.assert_array_equal(state["event_frequency"], [3, 3])
        np.testing.assert_array_equal(state["document_frequency"], [2, 2])
        pooled = np.log1p([2 ** (-9 / 30) + 2 ** (-8 / 30), 2 ** (-7 / 30)]) * state["idf"]
        expected = pooled @ state["item_factors"]
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(values[0, :64], expected, atol=5e-4)
        assert np.count_nonzero(state["singular_values"]) == manifest["effective_rank"]
    assert manifest["retained_products"] == manifest["qualified_products"] == 2
    assert manifest["maximum_products"] == 0
    assert manifest["coverage"]["product_buy"]["zero_projection_users"] == 3
    assert manifest["coverage"]["add_to_cart"]["zero_projection_users"] == 3
    assert manifest["official_target_directory_read"] is False
    assert manifest["blocks"] == [
        {"name": "product_buy_cooccurrence", "start": 0, "stop": 64, "kind": "embedding"},
        {"name": "add_to_cart_cooccurrence", "start": 64, "stop": 128, "kind": "embedding"}]
    assert len(json.loads((output / "features.json").read_text())) == 128
    assert not list(output.glob("*.partial"))


def test_future_events_and_batch_size_do_not_change_fitted_values(tmp_path):
    data = example(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    arguments = dict(cutoff_utc="2022-01-10T00:00:00Z", seed=17, threads=1)
    build_product_cooccurrence(data, first, batch_size=2, **arguments)
    path = data / "input" / "add_to_cart.parquet"
    frame = pd.read_parquet(path)
    future = frame.iloc[:2].copy()
    future.timestamp = pd.Timestamp("2022-02-01")
    future.sku = 444
    pd.concat([frame, future], ignore_index=True).to_parquet(path, index=False)
    build_product_cooccurrence(data, second, batch_size=7, **arguments)
    np.testing.assert_array_equal(np.load(first / "embeddings.npy"), np.load(second / "embeddings.npy"))
    with np.load(first / "product_factors.npz") as a, np.load(second / "product_factors.npz") as b:
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])


def test_empty_and_rank_one_graphs_pad_with_exact_zeros(tmp_path):
    one = sparse.csr_matrix(np.array([[1, 2], [2, 4], [0, 0]], np.float32))
    factors, values, rank = product_factors(one, dimensions=64, oversamples=2, power_iterations=1)
    assert rank == 1 and factors.shape == (2, 64)
    np.testing.assert_array_equal(factors[:, 1:], 0)
    np.testing.assert_array_equal(values[1:], 0)
    data, output = example(tmp_path), tmp_path / "empty"
    manifest = build_product_cooccurrence(data, output, minimum_events=100, threads=1,
                                          cutoff_utc="2022-01-10T00:00:00Z")
    assert manifest["retained_products"] == manifest["effective_rank"] == 0
    np.testing.assert_array_equal(np.load(output / "embeddings.npy"), np.zeros((4, 128), np.float16))
