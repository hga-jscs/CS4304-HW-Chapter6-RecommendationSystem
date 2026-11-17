"""Pseudo-future isolation and an end-to-end tiny encoder training check."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extension.build_embeddings import EVENTS, build_embeddings
from extension.pretrain_encoder import (EMPTY_TIME, SEQUENCE_DTYPE, extract_embeddings,
    multitask_loss, prepare_training_data, temporal_boundary, train_encoder, update_recent)
from extension.pretrain_encoder import training_positive_weights
from extension.pretrain_encoder import (binary_auc, load_windows, make_encoder, prepare_multiwindow_data,
                                       row_major_cache, train_multiwindow_encoder)


def historical_example(directory: Path) -> Path:
    source = directory / "data"
    (source / "input").mkdir(parents=True)
    (source / "target").mkdir()
    (source / "target" / "do_not_read.txt").write_text("Official labels are unavailable to the encoder")
    ids = np.arange(10, 18, dtype=np.int64)
    np.save(source / "input" / "relevant_clients.npy", ids)
    pd.DataFrame({"sku": [100, 200, 999], "category": [1, 2, 9], "price": [10, 60, 99]}).to_parquet(
        source / "product_properties.parquet", index=False)
    for event in EVENTS:
        past_users = ids[:6] if event == "product_buy" else ids
        rows = [(int(user), "2022-10-20T00:00:00Z", 100 if user % 2 else 200) for user in past_users]
        rows += [(int(user), "2022-10-27T00:00:00Z", 999) for user in ids[::2]]
        rows += [(99, "2022-11-09T23:59:59Z", 999)]
        frame = pd.DataFrame(rows, columns=["client_id", "timestamp", "sku"])
        if event == "page_visit":
            frame = frame.rename(columns={"sku": "url"})
        if event == "search_query":
            frame = frame.drop(columns="sku")
            frame["query"] = "[1 2 3 4]"
        frame.to_parquet(source / "input" / f"{event}.parquet", index=False)
    return source


def test_window_boundary_and_frozen_popularity_exclude_pseudo_future(tmp_path, monkeypatch):
    source = historical_example(tmp_path)
    cutoff = pd.Timestamp("2022-11-09T23:59:59Z").value
    assert temporal_boundary(cutoff, 14) == pd.Timestamp("2022-10-27T00:00:00Z").value
    import extension.pretrain_encoder as module
    original = module.parquet.ParquetFile

    def no_official_targets(path, *args, **kwargs):
        assert "target" not in Path(path).parts
        return original(path, *args, **kwargs)

    monkeypatch.setattr(module.parquet, "ParquetFile", no_official_targets)
    first, second = tmp_path / "first", tmp_path / "second"
    manifest = prepare_training_data(source, first, batch_size=2, sequence_length=3)
    classes = json.loads((first / "classes.json").read_text())
    assert classes == {"sku": [100, 200], "category": [1, 2]}
    targets = np.load(first / "pseudo_targets.npy")
    np.testing.assert_array_equal(targets[:, 0], [0, 1, 0, 1, 0, 1, 0, 1])
    assert not targets[:, 1:].any()  # Future-only product/class cannot enter the task vocabulary.
    np.testing.assert_array_equal(np.load(first / "churn_eligible.npy"), [True] * 6 + [False] * 2)
    prefix_sequence = np.load(first / "prefix_sequence.npy")
    valid = prefix_sequence["timestamp"] != EMPTY_TIME
    assert (prefix_sequence["timestamp"][valid] < temporal_boundary(cutoff, 14)).all()
    path = source / "input" / "product_buy.parquet"
    frame = pd.read_parquet(path)
    future = frame.loc[frame["timestamp"] == "2022-10-27T00:00:00Z"]
    pd.concat([frame] + [future] * 20, ignore_index=True).to_parquet(path, index=False)
    prepare_training_data(source, second, batch_size=100, sequence_length=3)
    np.testing.assert_array_equal(np.load(first / "prefix" / "embeddings.npy"),
                                  np.load(second / "prefix" / "embeddings.npy"))
    np.testing.assert_array_equal(prefix_sequence, np.load(second / "prefix_sequence.npy"))
    assert json.loads((second / "classes.json").read_text()) == classes
    full_manifest = json.loads((second / "full" / "manifest.json").read_text())
    assert 999 not in full_manifest["popular_history_identifiers"]["product_buy"]["sku"]
    assert full_manifest["popular_identifiers_frozen"]
    assert manifest["official_target_directory_read"] is False
    assert np.load(second / "full" / "embeddings.npy").shape == np.load(second / "prefix" / "embeddings.npy").shape


def test_recent_sequence_is_latest_and_independent_of_batch_order():
    rows = np.array([0, 1, 0, 0, 1, 0])
    records = np.zeros(len(rows), dtype=SEQUENCE_DTYPE)
    records["timestamp"] = [10, 8, 30, 20, 4, 30]
    records["item"] = [1, 2, 3, 4, 5, 6]
    records["event"] = 1
    first = np.zeros((2, 3), dtype=SEQUENCE_DTYPE)
    first["timestamp"] = EMPTY_TIME
    second = first.copy()
    update_recent(first, rows, records)
    for index in [5, 3, 1, 4, 2, 0]:
        update_recent(second, rows[index:index + 1], records[index:index + 1])
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[0]["timestamp"], [30, 30, 20])
    np.testing.assert_array_equal(first[0]["item"], [3, 6, 4])


def test_churn_head_has_no_gradient_for_users_without_past_purchase():
    torch = pytest.importorskip("torch")
    logits = torch.zeros((3, 201), requires_grad=True)
    labels = torch.ones((3, 201))
    loss = multitask_loss(logits, labels, torch.tensor([1., 0., 1.]),
                          torch.ones(201), {"category": 2, "sku": 2})
    loss.backward()
    assert logits.grad[1, 0] == 0
    assert logits.grad[0, 0] != 0
    assert logits.grad[1, 1] != 0
    assert not logits.grad[:, 3:101].any()


def test_churn_weight_balances_majority_positives_using_eligible_training_users_only():
    labels = np.zeros((7, 201), dtype=np.uint8)
    labels[:4, 0] = 1
    labels[5:, 0] = 1  # Row 5 is ineligible; row 6 is a validation user.
    labels[0, 1] = 1
    eligible = np.array([True] * 5 + [False, True])
    weights = training_positive_weights(labels, eligible, np.arange(6), batch_size=2)
    assert weights[0] == 0.25  # One negative / four positive eligible training users.
    assert weights[1] == 5
    assert (weights[1:] >= 1).all()


def test_tiny_training_and_full_history_extraction(tmp_path):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    source = historical_example(tmp_path)
    cache, output = tmp_path / "cache", tmp_path / "encoder"
    prepare_training_data(source, cache, sequence_length=4, batch_size=5)
    training = train_encoder(cache, output, epochs=2, patience=2, batch_size=3, device="cpu")
    manifest = extract_embeddings(cache, output, batch_size=3, device="cpu")
    embeddings = np.load(output / "embeddings.npy")
    assert embeddings.shape == (8, 457)
    assert embeddings.dtype == np.float16
    assert np.isfinite(embeddings).all()
    assert len(training["epochs"]) == 2
    assert np.isfinite(training["best_validation_weighted_bce"])
    assert manifest["official_target_directory_read"] is False
    np.testing.assert_array_equal(np.load(output / "client_ids.npy"), np.arange(10, 18))
    assert not embeddings[:, 256 + 3:256 + 101].any()  # Unused synthetic classes have no arbitrary logits.


def test_validation_labels_do_not_change_training_updates_or_fitted_statistics(tmp_path):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    source = historical_example(tmp_path)
    cache, first, second = tmp_path / "cache", tmp_path / "first", tmp_path / "second"
    prepare_training_data(source, cache, sequence_length=4, batch_size=5)
    train_encoder(cache, first, epochs=1, batch_size=3, device="cpu", seed=7)
    validation = np.load(cache / "validation_users.npy")
    labels = np.load(cache / "pseudo_targets.npy", mmap_mode="r+")
    for column in (0, 1, 2, 101, 102):
        labels[validation, column] = 1 - labels[validation, column]
    labels.flush()
    train_encoder(cache, second, epochs=1, batch_size=3, device="cpu", seed=7)
    left = torch.load(first / "encoder.pt", weights_only=True)
    right = torch.load(second / "encoder.pt", weights_only=True)
    for key in ("mean", "scale", "positive_weights"):
        assert torch.equal(left[key], right[key])
    for key in left["model"]:
        assert torch.equal(left["model"][key], right["model"][key])
    assert left["validation_weighted_bce"] != right["validation_weighted_bce"]


def multiwindow_example(directory: Path) -> Path:
    source = historical_example(directory)
    properties = pd.read_parquet(source / "product_properties.parquet")
    properties["name"] = [[1, 2, 3, 4], [2, 4, 6, 8], [3, 6, 9, 12]]
    properties.to_parquet(source / "product_properties.parquet", index=False)
    for event in EVENTS:
        path = source / "input" / f"{event}.parquet"
        frame = pd.read_parquet(path)
        frame["timestamp"] = frame["timestamp"].replace({"2022-10-20T00:00:00Z": "2022-10-01T00:00:00Z"})
        if event == "product_buy":
            extra = pd.DataFrame({"client_id": [14, 11, 12, 13, 16],
                "timestamp": ["2022-10-12T23:59:59Z", "2022-10-13T00:00:00Z",
                              "2022-10-26T23:59:59Z", "2022-10-27T00:00:00Z", "2022-10-14T00:00:00Z"],
                "sku": [100, 100, 200, 200, 100]})
            frame = pd.concat((frame, extra), ignore_index=True)
        frame.to_parquet(path, index=False)
    return source


def test_multiwindow_boundaries_shared_coordinates_and_user_split(tmp_path, monkeypatch):
    source = multiwindow_example(tmp_path)
    import extension.pretrain_encoder as module
    original = module.parquet.ParquetFile

    def no_official_targets(path, *args, **kwargs):
        assert "target" not in Path(path).parts
        return original(path, *args, **kwargs)

    monkeypatch.setattr(module.parquet, "ParquetFile", no_official_targets)
    cache = tmp_path / "windows"
    metadata = prepare_multiwindow_data(source, cache, sequence_length=8, batch_size=3,
                                       validation_fraction=0.25)
    assert metadata["windows"][0]["target_start_inclusive_utc"] == "2022-10-13T00:00:00+00:00"
    assert metadata["windows"][1]["target_start_inclusive_utc"] == "2022-10-27T00:00:00+00:00"
    np.testing.assert_array_equal(np.load(cache / "earlier_targets.npy")[:, 0], [1, 0, 0, 1, 1, 1, 0, 1])
    np.testing.assert_array_equal(np.load(cache / "later_targets.npy")[:, 0], [0, 1, 0, 0, 0, 1, 0, 1])
    np.testing.assert_array_equal(np.load(cache / "earlier_eligible.npy"), [True] * 6 + [False, False])
    np.testing.assert_array_equal(np.load(cache / "later_eligible.npy"), [True] * 7 + [False])
    assert 999 not in json.loads((cache / "classes.json").read_text())["sku"]
    states = [(cache / name / "feature_state.npz").read_bytes() for name in ("earlier", "later", "full")]
    assert states[0] == states[1] == states[2]
    columns = [json.loads((cache / name / "features.json").read_text()) for name in ("earlier", "later", "full")]
    assert columns[0] == columns[1] == columns[2]
    assert metadata["aggregate_dimensions"] == len(columns[0])
    assert all(np.load(cache / name / "embeddings.npy", mmap_mode="r").flags.c_contiguous
               for name in ("earlier", "later", "full"))
    validation = np.load(cache / "validation_users.npy")
    assert validation.sum() == 2  # One mask is reused for both windows.
    for window in metadata["windows"]:
        sequence = np.load(cache / f"{window['name']}_sequence.npy")
        present = sequence["timestamp"] != EMPTY_TIME
        assert (sequence["timestamp"][present] <= pd.Timestamp(window["input_cutoff_utc"]).value).all()
        search = sequence["event"] == 5
        assert (sequence["item"][search] > 0).all()
    assert metadata["official_target_directory_read"] is False


@pytest.mark.parametrize("include_delayed", [False, True])
def test_multiwindow_training_isolates_validation_and_full_history(tmp_path, include_delayed):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    source = multiwindow_example(tmp_path)
    cache, other_cache = tmp_path / "cache", tmp_path / "other_cache"
    first, second = tmp_path / "first", tmp_path / "second"
    prepare_multiwindow_data(source, cache, sequence_length=8, batch_size=4, validation_fraction=0.25,
                             class_selection="training_targets", include_delayed_window=include_delayed)
    metadata = json.loads((cache / "manifest.json").read_text())
    windows = load_windows(cache, metadata)
    pairs = 3 if include_delayed else 2
    assert len(windows) == pairs
    assert [window["head_group"] for window in windows[:2]] == [0, 0]
    if include_delayed:
        assert windows[2]["name"] == "earlier_delayed"
        assert windows[2]["matrix"] is windows[0]["matrix"]
        assert windows[2]["sequence"] is windows[0]["sequence"]
        assert windows[2]["eligible"] is windows[0]["eligible"]
        assert not windows[2]["eligible"][6] and windows[1]["eligible"][6]
        assert windows[2]["labels"] is windows[1]["labels"]
        assert windows[2]["input_cutoff_utc"] == windows[0]["input_cutoff_utc"]
        assert windows[2]["target_start_inclusive_utc"] == "2022-10-27T00:00:00+00:00"
        assert windows[2]["target_end_inclusive_utc"] == windows[1]["target_end_inclusive_utc"]
        assert windows[2]["forecast_gap_days"] == 14
        assert windows[2]["head_group"] == 1
    assert not (cache / "earlier_delayed").exists()
    assert metadata["class_selection_purchase_events"] == 6  # Later targets are not counted twice.
    assert json.loads((cache / "classes.json").read_text()) == {"sku": [999, 100, 200], "category": [9, 1, 2]}
    result = train_multiwindow_encoder(cache, first, epochs=1, batch_size=3, device="cpu", seed=7,
                                      churn_weight_mode="unit", churn_multiplier=3)
    validation = np.load(cache / "validation_users.npy")
    validation_ids = np.load(cache / "client_ids.npy")[validation]
    purchase_path = source / "input" / "product_buy.parquet"
    purchases = pd.read_parquet(purchase_path)
    changed = purchases.client_id.isin(validation_ids) & (pd.to_datetime(purchases.timestamp, utc=True)
                                                         >= pd.Timestamp("2022-10-13T00:00:00Z"))
    assert changed.any()
    purchases.loc[changed, "sku"] = 100
    purchases.to_parquet(purchase_path, index=False)
    prepare_multiwindow_data(source, other_cache, sequence_length=8, batch_size=4, validation_fraction=0.25,
                             class_selection="training_targets", include_delayed_window=include_delayed)
    assert (cache / "classes.json").read_bytes() == (other_cache / "classes.json").read_bytes()
    assert (cache / "earlier" / "feature_state.npz").read_bytes() == (other_cache / "earlier" / "feature_state.npz").read_bytes()
    np.testing.assert_array_equal(validation, np.load(other_cache / "validation_users.npy"))
    metadata = json.loads((other_cache / "manifest.json").read_text())
    assert metadata["class_selection"] == "training_targets"
    assert all(sum(window["label_support"]["training"]["sku_positive_users"]) > 0
               for window in metadata["windows"])
    for name in ("earlier", "later"):
        np.testing.assert_array_equal(np.load(cache / f"{name}_targets.npy")[~validation],
                                      np.load(other_cache / f"{name}_targets.npy")[~validation])
        labels = np.load(other_cache / f"{name}_targets.npy", mmap_mode="r+")
        for column in (0, 1, 2, 101, 102):
            labels[validation, column] = 1 - labels[validation, column]
        labels.flush()
        matrix = np.load(other_cache / name / "embeddings.npy", mmap_mode="r+")
        matrix[validation] += 10
        matrix.flush()
    full = np.load(other_cache / "full" / "embeddings.npy", mmap_mode="r+")
    full[:] += 100  # Neither inference rows nor validation features can fit training normalization.
    full.flush()
    train_multiwindow_encoder(other_cache, second, epochs=1, batch_size=3, device="cpu", seed=7,
                              churn_weight_mode="unit", churn_multiplier=3)
    left = torch.load(first / "encoder.pt", weights_only=True)
    right = torch.load(second / "encoder.pt", weights_only=True)
    for key in ("mean", "scale", "positive_weights"):
        assert torch.equal(left[key], right[key])
    for key in left["model"]:
        assert torch.equal(left["model"][key], right["model"][key])
    assert left["validation_weighted_bce"] != right["validation_weighted_bce"]
    assert result["training_window_rows"] == 6 * pairs
    assert result["validation_window_rows"] == 2 * pairs
    assert result["training_pairs"] == pairs
    assert result["normalization_input_windows"] == 2
    assert result["churn_positive_weight"] == 1
    assert set(result["epochs"][0]["validation"]["windows"]) == {window["name"] for window in windows}
    head_groups = {"immediate", "delayed"} if include_delayed else {"immediate"}
    assert set(result["epochs"][0]["validation"]["head_groups"]) == head_groups
    assert set(result["epochs"][0]["training_head_groups"]) == head_groups
    for group in range(len(head_groups)):
        selected = [window for window in windows if window["head_group"] == group]
        expected = training_positive_weights([window["labels"] for window in selected],
            [window["eligible"] for window in selected], np.flatnonzero(~validation), 3, "unit")
        np.testing.assert_array_equal(left["positive_weights"][group].numpy(), expected)
    manifest = extract_embeddings(cache, first, batch_size=3, device="cpu")
    values = np.load(first / "embeddings.npy")
    assert values.shape == (8, 594 if include_delayed else 393)
    assert np.isfinite(values).all()
    assert ((values[:, 192:] >= 0) & (values[:, 192:] <= 1)).all()
    assert manifest["blocks"][2] == {"name": "immediate_scores" if include_delayed else "prediction_scores",
                                     "start": 192, "stop": 393, "kind": "probability"}
    if include_delayed:
        assert manifest["blocks"][3] == {"name": "delayed_scores", "start": 393, "stop": 594, "kind": "probability"}
    assert len(json.loads((first / "features.json").read_text())) == values.shape[1]
    np.testing.assert_array_equal(np.load(first / "client_ids.npy"), np.arange(10, 18))


def test_explicit_churn_loss_multiplier_and_unit_weight():
    torch = pytest.importorskip("torch")
    labels = np.zeros((4, 201), dtype=np.uint8)
    labels[:3, 0] = 1
    weights = training_positive_weights(labels, np.ones(4, bool), np.arange(4), 2, "unit")
    assert weights[0] == 1
    gradients = []
    for multiplier in (1, 3):
        logits = torch.zeros((4, 201), requires_grad=True)
        multitask_loss(logits, torch.from_numpy(labels.astype(np.float32)), torch.ones(4),
                       torch.from_numpy(weights), {"category": 2, "sku": 2}, multiplier).backward()
        gradients.append(logits.grad)
    torch.testing.assert_close(gradients[1][:, 0], 3 * gradients[0][:, 0])
    torch.testing.assert_close(gradients[1][:, 1:], gradients[0][:, 1:])
    assert binary_auc([0, 1, 0, 1], [0, 0, 1, 1]) == 0.5
    assert binary_auc([1, 1], [0, 1]) is None


def test_attentive_encoder_ignores_padding_and_handles_empty_sequences():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    torch.manual_seed(2)
    model = make_encoder(4, "multiwindow").eval()
    aggregate = torch.ones((2, 4))
    item = torch.tensor([[2, 3, 777, 888], [0, 0, 0, 0]])
    category = torch.zeros_like(item)
    event = torch.tensor([[1, 2, 5, 5], [0, 0, 0, 0]])
    numeric = torch.zeros((2, 4, 3))
    with torch.no_grad():
        latent, logits = model(aggregate, item, category, event, numeric, torch.tensor([2, 0]))
        other = model(aggregate[:1], item[:1, :2], category[:1, :2], event[:1, :2], numeric[:1, :2], torch.tensor([2]))
    assert latent.shape == (2, 192) and logits.shape == (2, 201)
    assert torch.isfinite(latent).all() and torch.isfinite(logits).all()
    torch.testing.assert_close(latent[:1], other[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(logits[:1], other[1], atol=1e-5, rtol=1e-5)


def test_row_cache_layout_preserves_feature_values(tmp_path):
    expected = np.asfortranarray((np.arange(35, dtype=np.float32).reshape(7, 5) - 10) / 7)
    path = tmp_path / "features.npy"
    np.save(path, expected, allow_pickle=False)
    row_major_cache(path, batch_size=3)
    actual = np.load(path, mmap_mode="r")
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, expected)


def test_delayed_head_routes_gradients_and_extracts_both_horizons():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    model = make_encoder(4, "multiwindow", include_delayed_window=True).eval()
    inputs = (torch.ones((2, 4)), torch.ones((2, 3), dtype=torch.long),
              torch.ones((2, 3), dtype=torch.long), torch.ones((2, 3), dtype=torch.long),
              torch.zeros((2, 3, 3)), torch.tensor([3, 3]))
    latent, immediate = model(*inputs, head_group=0)
    immediate.sum().backward()
    assert model.churn_head.weight.grad is not None
    assert all(parameter.grad is None for parameter in model.delayed_heads.parameters())
    model.zero_grad(set_to_none=True)
    _, delayed = model(*inputs, head_group=1)
    delayed.sum().backward()
    assert model.churn_head.weight.grad is None
    assert model.category_head.weight.grad is None
    assert model.sku_head.weight.grad is None
    assert all(parameter.grad is not None for parameter in model.delayed_heads.parameters())
    assert model.shared[0].weight.grad is not None and model.churn[0].weight.grad is not None
    with torch.no_grad():
        both_latent, both = model(*inputs, head_group=None)
    torch.testing.assert_close(both_latent, latent)
    torch.testing.assert_close(both[:, :201], immediate)
    torch.testing.assert_close(both[:, 201:], delayed)
    assert torch.cat((both_latent, both.sigmoid()), dim=1).shape == (2, 594)
