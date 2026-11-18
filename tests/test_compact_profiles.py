import json

import numpy as np
import pytest

from extension.combine_profiles import combine
from extension.compact_profiles import fit_projection


def test_centered_projection_ignores_constant_columns_and_preserves_rank():
    samples = np.array([[2, 3, 6], [0, 3, 0], [1, 3, 3], [3, 3, 9]], np.float16)
    mean, basis, ratio, active = fit_projection(samples, 0, 3, np.arange(4), 1)
    centered = samples - mean
    assert active == 2
    assert ratio == pytest.approx(1.0)
    np.testing.assert_allclose(centered @ basis @ basis.T, centered, atol=1e-5)
    assert basis[1] == 0


def test_weighted_concatenation_preserves_identity_order(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    x = np.array([[2, -3], [4, 0]], dtype=np.float16)
    ids = np.array([12, 3], dtype=np.int64)
    np.save(source / "embeddings.npy", x)
    np.save(source / "client_ids.npy", ids)
    (source / "features.json").write_text(json.dumps(["a", "b"]))
    out = tmp_path / "weighted"
    combine([source, source], out, [1.0, 1.3])
    actual = np.load(out / "embeddings.npy")
    np.testing.assert_array_equal(actual[:, :2], x)
    np.testing.assert_array_equal(actual[:, 2:], (x.astype(np.float32) * 1.3).astype(np.float16))
    np.testing.assert_array_equal(np.load(out / "client_ids.npy"), ids)
    with pytest.raises(ValueError, match="one finite weight"):
        combine([source], tmp_path / "bad", [1.0, 2.0])
