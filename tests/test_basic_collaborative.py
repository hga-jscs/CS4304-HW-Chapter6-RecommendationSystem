"""Check sparse factorization semantics rather than benchmark scores."""

import pickle

import numpy as np
import pandas as pd

from basic.collaborative import CollaborativeFeatures, InteractionEmbedding


def test_rank_one_reconstruction_includes_singular_values():
    users = pd.Series(["u0", "u0", "u1", "u1"])
    items = pd.Series(["s0", "s1", "s0", "s1"])
    fitted = InteractionEmbedding(32).fit(users, items)
    dot, cosine = fitted.transform(users, items)
    # Every normalized interaction row is [1/sqrt(2), 1/sqrt(2)].
    # Multiplying bare U and V would incorrectly return 1/2 here.
    np.testing.assert_allclose(dot, np.full(4, 1 / np.sqrt(2)), atol=2e-6)
    np.testing.assert_allclose(cosine, np.ones(4), atol=2e-6)
    assert fitted.metadata["rank"] == 1
    assert fitted.user_factors.dtype == fitted.item_factors.dtype == np.float32


def test_unknown_ids_rank_zero_and_missing_items_are_finite():
    fitted = InteractionEmbedding(32).fit(pd.Series(["only"]), pd.Series(["one"]))
    dot, cosine = fitted.transform(pd.Series(["only", "new"]), pd.Series(["one", "new"]))
    np.testing.assert_array_equal(dot, [0, 0])
    np.testing.assert_array_equal(cosine, [0, 0])
    assert fitted.metadata["rank"] == 0
    fitted.fit(pd.Series(["u0", "u1"]), pd.Series([None, None]))
    assert fitted.metadata["shape"] == [2, 0]
    dot, cosine = fitted.transform(pd.Series(["u0", "new"]), pd.Series([None, "x"]))
    assert np.isfinite(dot).all() and np.isfinite(cosine).all()


def test_target_independence_unknown_ids_and_batch_consistency():
    frame = pd.DataFrame({"msno": ["u0", "u0", "u1", "u2", "u2"],
                          "song_id": ["s0", "s1", "s1", "s0", "s2"],
                          "target": [0, 1, 0, 1, 0]})
    artists = pd.Series(["a0", "a1", "a1", "a0", "a2"])
    fitted = CollaborativeFeatures().fit(frame, artists)
    changed = CollaborativeFeatures().fit(frame.assign(target=1-frame.target), artists)
    expected = fitted.transform(frame, artists)
    altered = changed.transform(frame, artists)
    restored = pickle.loads(pickle.dumps(fitted))
    first = restored.transform(frame.iloc[:2], artists.iloc[:2])
    second = restored.transform(frame.iloc[2:], artists.iloc[2:])
    for name in fitted.feature_names:
        np.testing.assert_allclose(altered[name], expected[name], atol=1e-6)
        np.testing.assert_allclose(np.concatenate([first[name], second[name]]), expected[name], atol=1e-6)
        assert expected[name].dtype == np.float32
    unseen = pd.DataFrame({"msno": ["missing", "u0"], "song_id": ["s0", "missing"]})
    values = fitted.transform(unseen, pd.Series(["a0", "missing"]))
    for column in values.values():
        np.testing.assert_array_equal(column, [0, 0])
