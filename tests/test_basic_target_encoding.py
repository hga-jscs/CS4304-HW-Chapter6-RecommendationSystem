"""Label leakage boundaries and hand-calculated target-encoding values."""

import pickle

import numpy as np
import pandas as pd

from basic.target_encoding import TargetEncoder


def interactions():
    return pd.DataFrame({
        "msno": ["u", "u", "u", "v", "v", "w"],
        "song_id": ["a", "b", "a", "b", "c", "c"],
        "source_system_tab": ["home", "search", "home", "home", "search", "home"],
        "source_type": ["radio", "radio", "local", "local", "radio", "local"],
        "target": [1, 0, 1, 0, 0, 1],
        "row_id": np.arange(6),
    }), pd.Series(["A", "B", "A", "B", "C", "C"])


def test_five_fields_match_manual_smoothed_training_statistics():
    frame, artists = interactions()
    encoder = TargetEncoder()
    encoder.fit_transform(frame, artists)
    query = pd.DataFrame({"msno": ["u"], "song_id": ["b"],
                          "source_system_tab": ["home"], "source_type": ["radio"]})
    encoded = encoder.transform(query, pd.Series(["C"]))
    # The six training labels average 1/2. Counts/sums for this query are
    # user (3,2), song (2,0), artist (2,1), user/tab (2,2), user/type (2,1).
    expected = [12 / 23, 10 / 22, 11 / 22, 12 / 22, 11 / 22]
    assert list(encoded) == encoder.feature_names
    for name, value in zip(encoder.feature_names, expected):
        np.testing.assert_allclose(encoded[name], [value], rtol=1e-6)
        assert encoded[name].dtype == np.float32


def test_each_rows_label_is_excluded_from_its_random_oof_features():
    frame, artists = interactions()
    reference = TargetEncoder(seed=17).fit_transform(frame, artists)
    for position in range(len(frame)):
        changed = frame.copy()
        changed.loc[position, "target"] = 1 - changed.loc[position, "target"]
        altered = TargetEncoder(seed=17).fit_transform(changed, artists)
        for name in TargetEncoder.feature_names:
            assert altered[name][position] == reference[name][position]


def test_unseen_fold_categories_use_only_other_folds_label_prior():
    frame = pd.DataFrame({"msno": list("abcde"), "song_id": list("ABCDE"),
                          "source_system_tab": ["home"] * 5,
                          "source_type": ["radio"] * 5, "target": [0, 1, 1, 1, 1]})
    encoded = TargetEncoder(folds=5).fit_transform(frame, pd.Series(list("vwxyz")))
    # Each fold contains one row and all five key types are unique per row.
    # Its own row is absent: the first prior is 1, the other priors are 3/4.
    for values in encoded.values():
        np.testing.assert_array_equal(values, [1, 0.75, 0.75, 0.75, 0.75])


def test_row_order_uses_only_preceding_blocks_and_restores_input_order():
    chronological = pd.DataFrame({
        "msno": ["u"] * 10, "song_id": ["s"] * 10,
        "source_system_tab": ["home"] * 10, "source_type": ["radio"] * 10,
        "target": [0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
        "global_row_id": np.arange(100, 110), "row_id": np.arange(10)[::-1],
    })
    order = np.array([8, 0, 5, 3, 9, 1, 7, 2, 6, 4])
    frame = chronological.iloc[order]
    artists = pd.Series(["A"] * len(frame))
    encoder = TargetEncoder()
    reference = encoder.fit_transform(frame, artists, split_mode="row_order")
    expected_by_time = np.repeat([0.5, 0, 0.5, 2 / 3, 0.5], 2)
    for values in reference.values():
        np.testing.assert_allclose(values, expected_by_time[order], rtol=1e-6)

    last_block_changed = frame.copy()
    last_block_changed.loc[last_block_changed.global_row_id >= 108, "target"] = 0
    altered = TargetEncoder().fit_transform(last_block_changed, artists, split_mode="row_order")
    # No block may use labels from the last block, including that block itself.
    for name in encoder.feature_names:
        np.testing.assert_array_equal(altered[name], reference[name])

    earlier_block_changed = frame.copy()
    earlier_block_changed.loc[earlier_block_changed.global_row_id.isin([106, 107]), "target"] = 1
    altered = TargetEncoder().fit_transform(earlier_block_changed, artists, split_mode="row_order")
    earlier = frame.global_row_id.to_numpy() < 108
    for name in encoder.feature_names:
        np.testing.assert_array_equal(altered[name][earlier], reference[name][earlier])
        np.testing.assert_allclose(altered[name][~earlier], 0.75)


def test_unknown_keys_missing_categories_and_prediction_batches():
    frame, artists = interactions()
    frame.loc[0:1, "source_system_tab"] = None
    artists.iloc[0:1] = None
    encoder = TargetEncoder()
    encoder.fit_transform(frame, artists)
    encoder = pickle.loads(pickle.dumps(encoder))
    query = pd.DataFrame({"msno": ["missing", "u", "w"], "song_id": ["missing", "a", "c"],
                          "source_system_tab": ["missing", pd.NA, "search"],
                          "source_type": ["missing", "radio", "local"], "target": [99, 99, 99]})
    query_artists = pd.Series(["missing", pd.NA, "C"])
    expected = encoder.transform(query, query_artists)
    changed_labels = encoder.transform(query.assign(target=-999), query_artists)
    first = encoder.transform(query.iloc[:1], query_artists.iloc[:1])
    rest = encoder.transform(query.iloc[1:], query_artists.iloc[1:])
    for name in encoder.feature_names:
        assert expected[name][0] == 0.5
        assert np.isfinite(expected[name]).all()
        np.testing.assert_array_equal(expected[name], changed_labels[name])
        np.testing.assert_array_equal(expected[name], np.concatenate([first[name], rest[name]]))
    # The missing tab occurs twice for user u, with one positive label.
    assert expected["target_user_source_system_tab"][1] == 0.5
    # Both w and search are known separately, but this particular pair is unseen.
    assert expected["target_user_source_system_tab"][2] == 0.5
    # Missing artist is a real fitted category with one positive label.
    np.testing.assert_allclose(expected["target_artist_name"][1], 11 / 21)


def test_singleton_oof_uses_half_prior_and_inference_uses_fitted_labels():
    frame, artists = interactions()
    frame, artists = frame.iloc[:1], artists.iloc[:1]
    for mode in ("random", "row_order"):
        encoder = TargetEncoder()
        output = encoder.fit_transform(frame, artists, split_mode=mode)
        predicted = encoder.transform(frame, artists)
        for name in encoder.feature_names:
            np.testing.assert_array_equal(output[name], [0.5])
            np.testing.assert_array_equal(predicted[name], [1.0])
