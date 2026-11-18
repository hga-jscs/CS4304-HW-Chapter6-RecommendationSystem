import numpy as np
from scipy.sparse import csr_matrix

from extension.activity_repeat_scores import activity_at_anchor


def test_activity_windows_end_at_anchor_for_each_user():
    history = np.zeros((2, 155), np.int32)
    history[1, [26, 27, 28, 34, 35, 139, 140]] = [500, 100, 3, 2, 1, 8, 99]
    counts = csr_matrix(history)
    result = np.rint(np.expm1(activity_at_anchor(counts, np.array([1, 0]), age=28)))
    # Window of 7 days covers ages 28..34; 112 days covers ages 28..139.
    assert result[0, :2].tolist() == [5, 2]
    assert result[0, 8:10].tolist() == [14, 4]
    assert result[0, 10] == 0
    assert result[1, :10].tolist() == [0] * 10


def test_score_replacement_preserves_named_columns_and_row_order(tmp_path):
    import json
    from extension.activity_repeat_scores import replace_scores
    from extension.repeat_purchases import BASE_NAMES, POST_NAMES, PRIOR_NAMES, SCORE_NAMES

    directories = [tmp_path / name for name in ("previous", "scores", "combined")]
    previous, new, output = directories
    ids = np.array([31, 20, 75], dtype=np.int64)
    original_names = BASE_NAMES + POST_NAMES + PRIOR_NAMES + SCORE_NAMES
    old_values = np.arange(len(ids) * len(original_names), dtype=np.float16).reshape(len(ids), -1)
    new_values = np.full((len(ids), len(SCORE_NAMES)), -2, np.float16)
    for path, names, values in ((previous, original_names, old_values), (new, SCORE_NAMES, new_values)):
        path.mkdir()
        np.save(path / "client_ids.npy", ids)
        np.save(path / "embeddings.npy", values)
        (path / "features.json").write_text(json.dumps(names), encoding="utf-8")
    replace_scores(previous, new, output)
    result = np.load(output / "embeddings.npy")
    np.testing.assert_array_equal(result[:, :-4], old_values[:, :-4])
    np.testing.assert_array_equal(result[:, -4:], new_values)
    np.testing.assert_array_equal(np.load(output / "client_ids.npy"), ids)


def test_build_exports_scores_when_paired_comparison_is_negative(tmp_path, monkeypatch):
    import pandas as pd
    from extension import activity_repeat_scores as pipeline

    data = tmp_path / "data"
    source = data / "input"
    source.mkdir(parents=True)
    ids = np.arange(30, dtype=np.int64)
    np.save(source / "relevant_clients.npy", ids)
    pd.DataFrame({"sku": [7], "category": [8]}).to_parquet(data / "product_properties.parquet")
    for event in pipeline.EVENTS:
        field = (
            "sku"
            if event in ("product_buy", "add_to_cart", "remove_from_cart")
            else ("url" if event == "page_visit" else "query")
        )
        events = pd.DataFrame(
            {
                "client_id": np.r_[ids, ids[-1]],
                field: 7,
                "timestamp": [pd.Timestamp("2022-01-03")] * len(ids) + [pd.Timestamp("2022-05-01")],
            }
        )
        events.to_parquet(source / f"{event}.parquet")
    monkeypatch.setattr(pipeline, "compare", lambda *args: {"differences": {"all": {"gain": -0.02}}})

    def fake_fit(history, current, users, output, seed):
        values = np.zeros((len(users), len(pipeline.HORIZONS)), np.float16)
        np.save(output / "embeddings.partial.npy", values)
        return sum(len(record["ids"]) for record in history)

    monkeypatch.setattr(pipeline, "save_scores", fake_fit)
    output = tmp_path / "scores"
    record = pipeline.build(data, output, sample_users=len(ids))
    assert record["historical_ablation"]["differences"]["all"]["gain"] < 0
    assert record["dimensions"] == 4
    assert np.load(output / "embeddings.npy").shape == (len(ids), 4)
