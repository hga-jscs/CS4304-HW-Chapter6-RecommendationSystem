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


def test_enhanced_exact_preferences_prices_and_intervals_use_history_only(tmp_path):
    source = example_data(tmp_path)
    properties = pd.read_parquet(source / "product_properties.parquet")
    properties["price"] = [1, 95]
    properties.to_parquet(source / "product_properties.parquet", index=False)
    first, second = tmp_path / "enhanced", tmp_path / "enhanced_again"
    build_embeddings(source, first, mode="enhanced", batch_size=1)
    (source / "target").mkdir()
    np.save(source / "target" / "propensity_sku.npy", np.array([999]))
    pd.DataFrame({"client_id": [20] * 100, "timestamp": ["2030-01-01"] * 100,
                  "sku": [999] * 100}).to_parquet(source / "target" / "train_target.parquet", index=False)
    build_embeddings(source, second, mode="enhanced", batch_size=100)
    matrix = np.load(first / "embeddings.npy")
    np.testing.assert_array_equal(matrix, np.load(second / "embeddings.npy"))
    names = json.loads((first / "features.json").read_text())
    values = {name: matrix[:, i] for i, name in enumerate(names)}
    # Client 9 bought price-bucket indices 1 and 95, exactly 23 days apart.
    np.testing.assert_allclose(values["product_buy.price_bucket_mean"], np.log1p([48, 1, 0]), atol=0.002)
    np.testing.assert_allclose(values["product_buy.price_bucket_std"], np.log1p([47, 0, 0]), atol=0.002)
    np.testing.assert_allclose(values["mean_purchase_interval_days"], np.log1p([23, 0, 0]), atol=0.002)
    np.testing.assert_allclose(values["purchase_recency_to_interval_ratio"], np.log1p([7 / 24, 0, 0]), atol=0.001)
    np.testing.assert_allclose(values["product_buy.popular_sku_100"], np.log1p([0.5, 2 ** (-1 / 30), 0]), atol=0.001)
    np.testing.assert_allclose(values["product_buy.price_band_4"], np.log1p([2 ** (-7 / 30), 0, 0]), atol=0.001)
    np.testing.assert_allclose(values["product_buy.has_history"], np.log1p([1, 1, 0]), atol=0.001)
    assert "product_buy.price_bucket_0" not in names
    assert not (first / "features.partial.npy").exists()
    # A rerun replaces this builder's results and leaves unrelated user notes.
    (first / "notes.txt").write_text("keep")
    build_embeddings(source, first, mode="enhanced", standardize=True, overwrite=True)
    assert (first / "notes.txt").read_text() == "keep"
    assert json.loads((first / "manifest.json").read_text())["standardized"] is True


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
    monkeypatch.setattr(module, "check_official_code", lambda directory, allow_modified: {
        "commit": module.OFFICIAL_COMMIT, "tracked_changes": [], "matches_reference": True})
    monkeypatch.setattr(module, "run_logged", lambda command, directory, log: calls.append(command))
    module.evaluate("official", "data", "embeddings", "scores", python="environment/python.exe", validate_only=True)
    assert calls[0][0] == str(interpreter.resolve())


def test_modified_evaluator_is_allowed_only_when_labelled(monkeypatch, tmp_path):
    from extension import evaluate as module

    def output(command, **kwargs):
        return module.OFFICIAL_COMMIT if "rev-parse" in command else " M training_pipeline/model.py\n"
    monkeypatch.setattr(module.subprocess, "check_output", output)
    with pytest.raises(ValueError, match="allow-modified-official"):
        module.check_official_code(tmp_path)
    with pytest.warns(UserWarning, match="not marked"):
        state = module.check_official_code(tmp_path, allow_modified=True)
    assert state["matches_reference"] is False
    assert state["tracked_changes"] == [" M training_pipeline/model.py"]


def test_evaluation_overwrite_clears_stale_scores_but_keeps_notes(tmp_path, monkeypatch):
    from extension import evaluate as module

    output = tmp_path / "scores"
    output.mkdir()
    for name in ("scores.json", "epochs.json", "evaluation.json"):
        (output / name).write_text('{}')
    (output / "notes.txt").write_text("keep")
    monkeypatch.setattr(module, "check_official_code", lambda *args: {
        "commit": module.OFFICIAL_COMMIT, "tracked_changes": [], "matches_reference": True})
    monkeypatch.setattr(module, "run_logged", lambda *args: None)
    module.evaluate(tmp_path, tmp_path, tmp_path, output, validate_only=True, overwrite=True)
    assert not (output / "scores.json").exists()
    assert not (output / "epochs.json").exists()
    assert (output / "notes.txt").read_text() == "keep"
    assert json.loads((output / "evaluation.json").read_text())["official_protocol_verified"] is True


def test_target_cache_preserves_order_and_binary_membership():
    from extension.target_cache import create_target_cache

    clients = np.array([7, 2, 9], dtype=np.int64)
    targets = pd.DataFrame({"client_id": [7, 7, 7, 2, 99], "sku": [10, 10, 30, 99, 30]})
    # Class order is deliberately reversed; duplicate events do not add counts.
    result = create_target_cache(clients, targets, "sku", np.array([30, 10]))
    np.testing.assert_array_equal(result, [[1, 1], [0, 0], [0, 0]])
    np.testing.assert_array_equal(create_target_cache(clients, targets), [[0], [0], [1]])
    assert result[0].astype(np.float32).dtype == np.float32


def test_evaluator_records_a_completed_task_without_requiring_other_tasks(tmp_path, monkeypatch):
    from extension import evaluate as module

    output = tmp_path / "scores"
    commands = []
    monkeypatch.setattr(module, "check_official_code", lambda *args: {
        "commit": module.OFFICIAL_COMMIT, "tracked_changes": [], "matches_reference": True})

    def run(command, directory, log):
        commands.append(command)
        if "--tasks" in command:
            assert command[command.index("--tasks") + 1:command.index("--log-name")] == ["propensity_sku"]
            (output / "scores.json").write_text('{"propensity_sku": 0.8}')
            (output / "epochs.json").write_text('{"propensity_sku": 2}')

    monkeypatch.setattr(module, "run_logged", run)
    result = module.evaluate(tmp_path, tmp_path, tmp_path, output, tasks=("propensity_sku",))
    assert len(commands) == 2
    assert result["tasks"] == ["propensity_sku"] and result["training_completed"]


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
