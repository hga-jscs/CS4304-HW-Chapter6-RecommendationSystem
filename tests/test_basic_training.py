"""Small deterministic fixtures test contracts, not competition performance."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from basic.features import FeatureBuilder, load_training, read_interactions
from basic.predict import predict
from basic.train import parser, train


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "data"
    folder.mkdir()
    # Unique user/song pairs with a deliberately learnable synthetic target.
    # This fixture only verifies pipeline behavior; its AUC is not a result.
    records = []
    for user in range(20):
        for song in range(20):
            target = int((user + song) % 3 == 0)
            records.append({"msno": f"user_{user}", "song_id": f"song_{song}",
                "source_system_tab": "library", "source_screen_name": "main",
                "source_type": "playlist" if target else "radio", "target": target})
    pd.DataFrame(records).to_csv(folder / "train.csv", index=False)
    pd.DataFrame([
        {"id": 17, "msno": "user_0", "song_id": "song_0", "source_system_tab": "library", "source_screen_name": "main", "source_type": "playlist"},
        {"id": 4, "msno": "unknown_user", "song_id": "unknown_song", "source_system_tab": "unseen", "source_screen_name": "", "source_type": "radio"},
        {"id": 99, "msno": "user_3", "song_id": "song_8", "source_system_tab": "library", "source_screen_name": "main", "source_type": "radio"},
    ]).to_csv(folder / "test.csv", index=False)
    pd.DataFrame({"id": [17, 4, 99], "target": [0, 0, 0]}).to_csv(folder / "sample_submission.csv", index=False)
    pd.DataFrame([{"msno": f"user_{user}", "city": user % 3, "bd": 20+user,
        "gender": "male" if user % 2 else "female", "registered_via": 7,
        "registration_init_time": 20150101, "expiration_date": 20170101} for user in range(20)]).to_csv(folder / "members.csv", index=False)
    pd.DataFrame([{"song_id": f"song_{song}", "song_length": 180000+song,
        "genre_ids": "1|2" if song % 2 else "1", "artist_name": f"artist_{song%5}",
        "composer": "composer", "lyricist": "lyricist", "language": 3} for song in range(20)]).to_csv(folder / "songs.csv", index=False)
    pd.DataFrame([{"song_id": f"song_{song}", "name": f"歌曲 {song}", "isrc": "TWAAA1600001"} for song in range(20)]).to_csv(folder / "song_extra_info.csv", index=False)
    return folder


def test_uniform_sample_covers_whole_file_and_is_reproducible(data_dir):
    first, info = load_training(data_dir / "train.csv", max_rows=40, seed=42, chunksize=33)
    second, _ = load_training(data_dir / "train.csv", max_rows=40, seed=42, chunksize=77)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 40
    assert first.row_id.max() > 350
    assert first.row_id.is_unique
    assert info["available_rows"] == 400
    assert info["sampling"] == "uniform_without_replacement"


def test_feature_fit_does_not_observe_holdout_or_labels(data_dir):
    frame, _ = load_training(data_dir / "train.csv")
    training = frame.iloc[:200]
    heldout = frame.iloc[200:]
    builder = FeatureBuilder(data_dir, "enhanced").fit(training)
    transformed = builder.transform(heldout)
    assert (transformed["msno"] == -1).all()
    assert (transformed["msno_frequency"] == 0).all()
    assert (transformed["song_id_frequency"] == 10).all()
    changed_labels = training.assign(target=1-training.target)
    other = FeatureBuilder(data_dir, "enhanced").fit(changed_labels)
    pd.testing.assert_frame_equal(transformed, other.transform(heldout))
    assert "target" not in transformed
    assert "row_id" not in transformed


def test_duplicate_metadata_is_rejected(data_dir):
    members = pd.read_csv(data_dir / "members.csv")
    pd.concat([members, members.iloc[:1]]).to_csv(data_dir / "members.csv", index=False)
    with pytest.raises(ValueError, match="unique"):
        FeatureBuilder(data_dir)


def test_disabling_frequencies_only_removes_five_columns(data_dir):
    frame, _ = load_training(data_dir / "train.csv")
    training, heldout = frame.iloc[:200], frame.iloc[200:]
    complete = FeatureBuilder(data_dir, "enhanced").fit(training)
    ablation = FeatureBuilder(data_dir, "enhanced", disable_frequency_features=True).fit(training)
    expected_removed = {name + "_frequency" for name in ["msno", "song_id", "artist_name", "genre_ids", "source_type"]}
    assert set(complete.feature_names) - set(ablation.feature_names) == expected_removed
    assert len(complete.feature_names) - len(ablation.feature_names) == 5
    assert ablation.frequencies == {}
    assert complete.categorical_columns == ablation.categorical_columns
    for name in complete.categorical_columns:
        pd.testing.assert_index_equal(complete.categories[name], ablation.categories[name])
    expected = complete.transform(heldout).drop(columns=sorted(expected_removed))
    pd.testing.assert_frame_equal(ablation.transform(heldout), expected)
    restored = pickle.loads(pickle.dumps(ablation))
    assert restored.disable_frequency_features is True
    pd.testing.assert_frame_equal(restored.transform(heldout), expected)


def test_user_context_probabilities_leave_out_self_and_use_training_only(data_dir):
    frame, _ = load_training(data_dir / "train.csv")
    training = frame.iloc[:4].copy()
    training["msno"] = pd.Series(["user_0", "user_0", "user_0", "user_1"], dtype="string[pyarrow]")
    training["source_type"] = pd.Series(["a", "a", "b", None], dtype="string[pyarrow]")
    builder = FeatureBuilder(data_dir, user_context_features=True).fit(training)
    column = "msno_source_type_probability"
    fitted = builder.transform(training)
    leave_one_out = builder.transform(training, training_context=True)
    assert fitted[column].iloc[0] == pytest.approx((2 + 10 * 0.5) / (3 + 10))
    assert leave_one_out[column].iloc[0] == pytest.approx((1 + 10 * 0.5) / (2 + 10))
    assert leave_one_out[column].iloc[3] == pytest.approx(0.25)
    unseen = training.copy()
    unseen["msno"] = pd.Series(["new_user", "new_user", "user_0", "new_user"], dtype="string[pyarrow]")
    unseen["source_type"] = pd.Series(["a", "new_context", "new_context", None], dtype="string[pyarrow]")
    np.testing.assert_allclose(builder.transform(unseen)[column], [0.5, 0.0, 0.0, 0.25])
    relabeled = FeatureBuilder(data_dir, user_context_features=True).fit(training.assign(target=1-training.target))
    pd.testing.assert_frame_equal(builder.transform(unseen), relabeled.transform(unseen))
    restored = pickle.loads(pickle.dumps(builder))
    pd.testing.assert_frame_equal(restored.transform(unseen), builder.transform(unseen))
    with pytest.raises(ValueError, match="included"):
        builder.transform(unseen, training_context=True)


def test_default_and_old_feature_states_keep_existing_output(data_dir):
    frame, _ = load_training(data_dir / "train.csv")
    builder = FeatureBuilder(data_dir, "enhanced").fit(frame)
    expected = builder.transform(frame)
    assert not any(column.endswith("_probability") for column in expected)
    del builder.user_context_features
    pd.testing.assert_frame_equal(builder.transform(frame), expected)


def test_invalid_training_target_is_rejected_before_narrowing(data_dir):
    frame = pd.read_csv(data_dir / "train.csv")
    frame.loc[0, "target"] = 256
    frame.to_csv(data_dir / "train.csv", index=False)
    with pytest.raises(ValueError, match="only 0 and 1"):
        list(read_interactions(data_dir / "train.csv", training=True))


@pytest.mark.parametrize("feature_mode,user_context_features,collaborative_features", [
    ("baseline", False, False), ("enhanced", False, False),
    ("enhanced", True, False), ("enhanced", True, True),
])
def test_training_and_chunked_prediction(data_dir, tmp_path, feature_mode, user_context_features, collaborative_features):
    model_dir = tmp_path / feature_mode
    args = parser().parse_args(["--data-dir", str(data_dir), "--output-dir", str(model_dir),
        "--feature-mode", feature_mode, "--threads", "1", "--num-boost-round", "12",
        "--early-stopping-rounds", "4", "--num-leaves", "7", "--min-child-samples", "5"])
    args.user_context_features = user_context_features
    args.collaborative_features = collaborative_features
    metrics = train(args)
    assert metrics["training_rows"] + metrics["validation_rows"] == 400
    assert metrics["final_training_rows"] == 400
    assert metrics["platform_score"] is None
    assert metrics["split"] == "stratified_random_holdout_not_chronological"
    assert 1 <= metrics["best_iteration"] <= 12
    output = tmp_path / f"{feature_mode}.csv"
    result = predict(data_dir, model_dir, output, chunk_size=2, threads=1)
    submission = pd.read_csv(output)
    assert result["rows"] == 3
    assert submission.id.tolist() == [17, 4, 99]
    assert submission.target.between(0, 1).all()
    assert not output.with_name(output.name+".partial").exists()
    manifest = json.loads((model_dir/"data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["train.csv"]["bytes"] == (data_dir / "train.csv").stat().st_size
    assert manifest["train.csv"]["mtime_ns"] == (data_dir / "train.csv").stat().st_mtime_ns
    assert "sha256" not in manifest["train.csv"]
    configuration = json.loads((model_dir/"configuration.json").read_text(encoding="utf-8"))
    assert configuration["user_context_features"] is user_context_features
    if collaborative_features:
        assert metrics["validation_collaborative_fit"]["song"]["rank"] > 0
        assert metrics["final_collaborative_fit"]["artist"]["rank"] > 0


def test_no_refit_and_explicit_overwrite_preserve_matching_state(data_dir, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("keep")
    args = parser().parse_args(["--data-dir", str(data_dir), "--output-dir", str(output),
        "--split-mode", "row_order", "--no-refit", "--threads", "1", "--num-boost-round", "5",
        "--num-leaves", "7", "--min-child-samples", "5"])
    metrics = train(args)
    assert metrics["final_training_rows"] == metrics["training_rows"] == 320
    assert metrics["refit_performed"] is False
    with (output / "features.pkl").open("rb") as stream:
        builder = pickle.load(stream)
    assert len(builder.categories["msno"]) == 16
    with pytest.raises(ValueError, match="overwrite"):
        train(args)
    args.overwrite = True
    train(args)
    assert (output / "keep.txt").read_text() == "keep"
    prediction = tmp_path / "prediction.csv"
    partial = prediction.with_name("prediction.csv.partial")
    partial.write_text("unfinished")
    predict(data_dir, output, prediction, threads=1)
    assert not partial.exists()
    with pytest.raises(ValueError, match="overwrite"):
        predict(data_dir, output, prediction, threads=1)
    before = prediction.read_bytes()
    predict(data_dir, output, prediction, threads=1, overwrite=True)
    assert prediction.read_bytes() == before


def test_calendar_features_keep_days_and_reference_year_order(data_dir):
    members = pd.read_csv(data_dir / "members.csv")
    members.loc[:2, "registration_init_time"] = [20161231, 20170101, 20170102]
    members.to_csv(data_dir / "members.csv", index=False)
    extra = pd.read_csv(data_dir / "song_extra_info.csv")
    extra.loc[:2, "isrc"] = ["TWAAA9900001", "TWAAA1700001", "invalid"]
    extra.to_csv(data_dir / "song_extra_info.csv", index=False)
    frame, _ = load_training(data_dir / "train.csv")
    rows = frame.iloc[[0, 21, 42]]
    values = FeatureBuilder(data_dir, "enhanced").fit(rows).transform(rows)
    np.testing.assert_array_equal(np.diff(values.registration_days), [1, 1])
    np.testing.assert_array_equal(values.isrc_reference_year.iloc[:2], [1999, 2017])
    assert np.isnan(values.isrc_reference_year.iloc[2])
    legacy = FeatureBuilder(data_dir, "enhanced", numeric_features="legacy").fit(rows)
    expected = legacy.transform(rows)
    del legacy.numeric_features
    pd.testing.assert_frame_equal(pickle.loads(pickle.dumps(legacy)).transform(rows), expected)


def test_relative_frequency_full_counts_and_preferences(data_dir):
    frame, _ = load_training(data_dir / "train.csv")
    rows = frame.iloc[:10]
    builder = FeatureBuilder(data_dir, "enhanced", frequency_mode="relative",
        context_counting="full", user_context_features=True, user_preference_features=True).fit(rows)
    expected = builder.transform(rows)
    pd.testing.assert_frame_equal(builder.transform(rows, training_context=True), expected)
    assert (expected.msno_frequency == 1).all()
    assert (expected.song_id_frequency == np.float32(0.1)).all()
    assert "msno_artist_name_probability" in expected
    assert "msno_genre_ids_probability" in expected
    assert expected.msno_artist_name_probability.iloc[0] == pytest.approx(0.2)
    other = FeatureBuilder(data_dir, "enhanced", frequency_mode="relative",
        context_counting="full", user_context_features=True, user_preference_features=True).fit(rows.assign(target=1-rows.target))
    pd.testing.assert_frame_equal(other.transform(rows), expected)


def test_tail_sampling_preserves_original_row_numbers(data_dir):
    frame, info = load_training(data_dir / "train.csv", max_rows=37, chunksize=41, sampling_mode="tail")
    assert frame.row_id.tolist() == list(range(363, 400))
    assert info["sampling"] == "tail_rows_not_verified_time"
