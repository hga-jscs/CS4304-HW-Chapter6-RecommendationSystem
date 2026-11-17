import json

import numpy as np

from extension.standardize import standardize_embeddings


def test_standardization_matches_independent_column_calculation_and_preserves_order(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    values = np.array([[0, 2, 5], [1, 2, 1], [3, 2, 9], [8, 2, 3]], dtype=np.float16)
    np.save(source / "embeddings.npy", values)
    np.save(source / "client_ids.npy", np.array([8, 3, 12, 1], dtype=np.int64))
    (source / "features.json").write_text('["a", "constant", "b"]')
    expected = values.astype(np.float64)
    scale = expected.std(axis=0)
    scale[scale == 0] = 1
    expected = ((expected - expected.mean(axis=0)) / scale).astype(np.float16)
    for batch_size in (1, 3):
        output = tmp_path / str(batch_size)
        standardize_embeddings(source, output, batch_size=batch_size)
        np.testing.assert_array_equal(np.load(output / "embeddings.npy"), expected)
        np.testing.assert_array_equal(np.load(output / "client_ids.npy"), [8, 3, 12, 1])
        assert json.loads((output / "manifest.json").read_text())["target_data_used_for_embeddings"] is False
    # Overwriting an experiment does not erase unrelated notes.
    (output / "notes.txt").write_text("keep")
    standardize_embeddings(source, output, overwrite=True)
    assert (output / "notes.txt").read_text() == "keep"
    np.testing.assert_array_equal(np.load(source / "embeddings.npy"), values)
    (source / "features.json").unlink()
    standardize_embeddings(source, output, overwrite=True)
    assert not (output / "features.json").exists()


def test_block_scaling_preserves_embedding_direction_and_zero_history(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    values = np.array([[1, 100, 1], [2, -50, 1], [3, 0, 0]], dtype=np.float32)
    np.save(source / "embeddings.npy", values)
    np.save(source / "client_ids.npy", np.array([4, 9, 2]))
    blocks = [{"name": "count", "start": 0, "stop": 1, "kind": "statistics"},
              {"name": "graph", "start": 1, "stop": 3, "kind": "embedding"}]
    (source / "manifest.json").write_text(json.dumps({"blocks": blocks}))
    for batch_size in (1, 3):
        output = tmp_path / f"scaled_{batch_size}"
        standardize_embeddings(source, output, method="block", batch_size=batch_size)
        scaled = np.load(output / "embeddings.npy").astype(np.float32)
        np.testing.assert_allclose(scaled[:2, 1] / scaled[:2, 2], [100, -50], rtol=.001)
        np.testing.assert_allclose(np.linalg.norm(scaled[:2, 1:], axis=1), 2, rtol=.001)
        np.testing.assert_array_equal(scaled[2, 1:], [0, 0])
        np.testing.assert_allclose(scaled[:, 0], [-np.sqrt(1.5), 0, np.sqrt(1.5)], rtol=.001)
    np.testing.assert_array_equal(np.load(source / "embeddings.npy"), values)


def test_combining_probability_features_does_not_apply_sigmoid_twice(tmp_path):
    history, core, output = [tmp_path / name for name in ("history", "core", "output")]
    for directory in (history, core):
        directory.mkdir()
        np.save(directory / "client_ids.npy", np.array([3, 1]))
    np.save(history / "embeddings.npy", np.array([[1], [3]], np.float32))
    np.save(core / "embeddings.npy", np.array([[.1], [.9]], np.float32))
    for directory, kind in ((history, "statistics"), (core, "probability")):
        (directory / "features.json").write_text('["feature"]')
        (directory / "manifest.json").write_text(json.dumps({"blocks": [
            {"name": kind, "start": 0, "stop": 1, "kind": kind}]}))
    standardize_embeddings(history, output, method="block", pretrained_directory=core)
    np.testing.assert_allclose(np.load(output / "embeddings.npy"), [[.13, -1], [1.17, 1]], rtol=.001)
    np.testing.assert_array_equal(np.load(output / "client_ids.npy"), [3, 1])
    assert not (output / "combined.partial.npy").exists()
