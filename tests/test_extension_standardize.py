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
