"""Small deterministic fixtures test contracts, not competition performance."""

from __future__ import annotations

import json
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


def test_invalid_training_target_is_rejected_before_narrowing(data_dir):
    frame = pd.read_csv(data_dir / "train.csv")
    frame.loc[0, "target"] = 256
    frame.to_csv(data_dir / "train.csv", index=False)
    with pytest.raises(ValueError, match="only 0 and 1"):
        list(read_interactions(data_dir / "train.csv", training=True))


@pytest.mark.parametrize("feature_mode", ["baseline", "enhanced"])
def test_training_and_chunked_prediction(data_dir, tmp_path, feature_mode):
    model_dir = tmp_path / feature_mode
    args = parser().parse_args(["--data-dir", str(data_dir), "--output-dir", str(model_dir),
        "--feature-mode", feature_mode, "--threads", "1", "--num-boost-round", "12",
        "--early-stopping-rounds", "4", "--num-leaves", "7", "--min-child-samples", "5"])
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
    assert len(manifest["train.csv"]["sha256"]) == 64


def test_training_refuses_to_overwrite_existing_experiment(data_dir, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("keep")
    args = parser().parse_args(["--data-dir", str(data_dir), "--output-dir", str(output)])
    with pytest.raises(ValueError, match="not empty"):
        train(args)
    assert (output / "keep.txt").read_text() == "keep"
