import numpy as np
from scipy.sparse import csr_matrix

from extension.activity_windows import daily_summary, outcomes, NAMES


def test_basket_counts_and_distinct_day_gaps():
    days = np.zeros((3, 112), np.int32)
    days[0, [0, 2, 6]] = [4, 2, 1]
    days[1, 15] = 1
    values = np.rint(np.expm1(daily_summary(days)))
    assert values[0, :2].tolist() == [7,3]
    assert values[0, -5:].tolist() == [0,6,2,4,3]
    assert values[1, -5:].tolist() == [15,15,112,112,112]
    assert values[2, -5:].tolist() == [112]*5
    assert len(NAMES) == values.shape[1]


def test_purchase_windows_exclude_anchor_and_include_boundaries():
    counts = np.zeros((6, 113), np.int32)
    counts[np.arange(6), [0,13,14,27,28,29]] = 1
    a,b = outcomes(csr_matrix(counts), 28)
    assert a.tolist() == [0,0,1,1,0,0]
    assert b.tolist() == [1,1,0,0,0,0]


def test_activity_window_boundaries_include_the_last_day():
    from extension.activity_windows import WINDOWS
    history = np.zeros((1, 120), np.int32)
    cutoff = 3
    history[0, cutoff-1] = 1000
    for size in WINDOWS:
        history[0, cutoff+size-1:cutoff+size+1] = 1
    values = np.rint(np.expm1(daily_summary(history[:, cutoff:cutoff+112])))
    assert values[0, :10:2].tolist() == [1, 3, 5, 7, 9]
