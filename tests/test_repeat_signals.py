import numpy as np
import pandas as pd

from extension.repeat_signals import buyer_groups, encode


def test_gate_respects_calendar_days_and_historical_cutoff(tmp_path):
    path = tmp_path / "purchases.parquet"
    pd.DataFrame({
        "client_id": [10, 10, 20, 20, 20, 30, 99],
        "timestamp": pd.to_datetime(["2022-01-01 12:00", "2022-01-02 12:00", "2022-01-01 10:00",
                                     "2022-01-01 13:00", "2022-01-03 12:00", "2022-01-04 12:00",
                                     "2022-01-01 12:00"]),
    }).to_parquet(path, row_group_size=2)
    buyers, single = buyer_groups([30, 20, 10, 40], path, np.datetime64("2022-01-02", "D").astype(int))
    assert buyers.tolist() == [False, True, True, False]
    assert single.tolist() == [False, True, False, False]


def test_conditional_residuals_and_gated_thresholds():
    rng = np.random.default_rng(42)
    old = rng.normal(size=(400, 4))
    extra = rng.normal(size=old.shape)
    new = old + 0.4 * old[:, :1] + extra
    single = rng.random(len(old)) > 0.45
    encoded, _, _ = encode(old, new, single, 2)
    assert encoded.shape == (400, 28)
    assert np.max(np.abs(np.corrcoef(old.T, encoded[:, 4:8].T)[:4, 4:])) < 1e-3
    assert np.count_nonzero(encoded[~single, 8:12]) == 0
    np.testing.assert_allclose(encoded[single, 8:12].astype(float).std(axis=0), 2, atol=0.001)
    assert (np.diff(encoded[np.argsort(new[:, -1]), 12:], axis=0) >= 0).all()
