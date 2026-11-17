"""Behavioral checks for the Synerise historical representation pipeline."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extension.build_embeddings import EVENTS, build_embeddings, parse_query


def example_data(directory: Path) -> Path:
    source = directory / "dataset"
    (source / "input").mkdir(parents=True)
    np.save(source / "input" / "relevant_clients.npy", np.array([9, 3, 20], dtype=np.int64))
    pd.DataFrame({"sku": [100, 200], "category": [7, 8], "price": [1, 2]}).to_parquet(
        source / "product_properties.parquet", index=False)
    for event in EVENTS:
        frame = pd.DataFrame({"client_id": [9, 9, 3, 99],
                              "timestamp": ["2025-01-01", "2025-01-24", "2025-01-30", "2025-01-31"]})
        if event in EVENTS[:3]:
            frame["sku"] = [100, 200, 100, 200]
        elif event == "page_visit":
            frame["url"] = [4, 5, 4, 6]
        else:
            frame["query"] = ["[1 2 3 4]", "[1 3 5 7]", "[1 2 3 4]", "[2 4 6 8]"]
        frame.to_parquet(source / "input" / f"{event}.parquet", index=False)
    return source


def test_historical_features_alignment_and_chunk_independence(tmp_path):
    source = example_data(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest = build_embeddings(source, first, batch_size=1)
    # Future target events cannot change history-derived features or the cutoff.
    (source / "target").mkdir()
    pd.DataFrame({"client_id": [20], "timestamp": ["2030-12-31"], "sku": [100]}).to_parquet(
        source / "target" / "validation_target.parquet", index=False)
    build_embeddings(source, second, batch_size=100)
    ids = np.load(first / "client_ids.npy")
    matrix = np.load(first / "embeddings.npy")
    np.testing.assert_array_equal(ids, [9, 3, 20])
    assert ids.dtype == np.int64
    assert matrix.shape == (3, 208)
    assert matrix.dtype == np.float16
    np.testing.assert_array_equal(matrix, np.load(second / "embeddings.npy"))
    # Independently calculated purchase counts and elapsed days for clients 9 and 3.
    np.testing.assert_allclose(matrix[:2, :4], np.log1p([[2, 0, 1, 2], [1, 1, 1, 1]]), atol=0.001)
    np.testing.assert_allclose(matrix[:2, 6:8], np.log1p([[7, 23], [1, 0]]), atol=0.002)
    assert matrix[2, 0] == 0  # No history still produces a row for relevant client 20.
    assert matrix[2, 6] > 0
    assert np.isfinite(matrix).all()
    assert manifest["target_data_used_for_embeddings"] is False
    assert manifest["events"]["product_buy"]["relevant_client_rows"] == 3
    assert len(json.loads((first / "features.json").read_text())) == matrix.shape[1]


def test_duplicate_relevant_clients_are_rejected(tmp_path):
    source = example_data(tmp_path)
    np.save(source / "input" / "relevant_clients.npy", np.array([9, 9], dtype=np.int64))
    with pytest.raises(ValueError, match="duplicate"):
        build_embeddings(source, tmp_path / "output")


def test_outputs_are_not_silently_replaced(tmp_path):
    source = example_data(tmp_path)
    output = tmp_path / "output"
    build_embeddings(source, output)
    before = (output / "embeddings.npy").read_bytes()
    with pytest.raises(FileExistsError):
        build_embeddings(source, output)
    assert (output / "embeddings.npy").read_bytes() == before


@pytest.mark.parametrize("query", ["1 2", "[-1 2]", "[256]", "[1 evil]", "[]"])
def test_malformed_query_codes_are_rejected(query):
    with pytest.raises(ValueError):
        parse_query(query)


def test_query_length_changes_fail_even_across_batches(tmp_path):
    source = example_data(tmp_path)
    path = source / "input" / "search_query.parquet"
    frame = pd.read_parquet(path)
    frame.loc[1, "query"] = "[1 2]"
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="length"):
        build_embeddings(source, tmp_path / "output", batch_size=1)


def test_relative_python_is_resolved_before_official_working_directory(tmp_path, monkeypatch):
    from extension import evaluate as module

    interpreter = tmp_path / "environment" / "python.exe"
    interpreter.parent.mkdir()
    interpreter.touch()
    calls = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "check_official_code", lambda directory: module.OFFICIAL_COMMIT)
    monkeypatch.setattr(module, "run_logged", lambda command, directory, log: calls.append(command))
    module.evaluate("official", "data", "embeddings", "scores", python="environment/python.exe", validate_only=True)
    assert calls[0][0] == str(interpreter.resolve())


def test_target_cache_preserves_order_and_binary_membership():
    from extension.target_cache import create_target_cache

    clients = np.array([7, 2, 9], dtype=np.int64)
    targets = pd.DataFrame({"client_id": [7, 7, 7, 2, 99], "sku": [10, 10, 30, 99, 30]})
    # Class order is deliberately reversed; duplicate events do not add counts.
    result = create_target_cache(clients, targets, "sku", np.array([30, 10]))
    np.testing.assert_array_equal(result, [[1, 1], [0, 0], [0, 0]])
    np.testing.assert_array_equal(create_target_cache(clients, targets), [[0], [0], [1]])
    assert result[0].astype(np.float32).dtype == np.float32


def test_target_cache_empty_window_and_invalid_orders():
    from extension.target_cache import create_target_cache

    empty = pd.DataFrame({"client_id": pd.Series(dtype="int64"), "sku": pd.Series(dtype="int64")})
    clients = np.array([8, 2])
    np.testing.assert_array_equal(create_target_cache(clients, empty), [[1], [1]])
    np.testing.assert_array_equal(create_target_cache(clients, empty, "sku", np.array([5])), [[0], [0]])
    with pytest.raises(ValueError, match="unique client"):
        create_target_cache(np.array([8, 8]), empty)
    with pytest.raises(ValueError, match="unique"):
        create_target_cache(clients, empty, "sku", np.array([5, 5]))
