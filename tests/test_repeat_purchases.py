import numpy as np
import pandas as pd

from extension.build_embeddings import DAY_NS
from extension.repeat_purchases import BASE_NAMES, POST_NAMES, Snapshot, future_labels, prior_tables


def purchases(records):
    frame = pd.DataFrame(records, columns=["user", "day", "sku"])
    frame["category"] = frame.sku
    frame["time"] = frame.day * DAY_NS
    return frame


def test_repeat_priors_exclude_entire_recipient_fold_and_count_distinct_exposures():
    # Only the anchor at day 0 has a fully observed 28-day exposure window.
    rows = [(0, 0, 10), (5, 0, 10), (1, 0, 10), (2, 0, 10), (2, 0, 10), (1, 14, 10), (2, 1, 20), (2, 15, 10)]
    levels, values = prior_tables(purchases(rows), cutoff=14, first_day=-27)[("sku", 14)]
    # Customers 0 and 5 are in the same held-out fold. Of the two remaining distinct buyers,
    # one repurchased item 10, and both bought some item. Day 15 is past the horizon.
    np.testing.assert_allclose(values[0, levels.get_loc(10)], [0.5, 1.0])
    np.testing.assert_allclose(values[0, -1], [0.5, 1.0])


def test_continuation_deduplicates_dates_and_distinguishes_ties_and_nonbuyers():
    buys = purchases([(1, 0, 10), (2, 1, 10)])
    snapshot = Snapshot(np.array([1, 2, 3]), cutoff=10, buys=buys, first_day=0)
    snapshot.add_events("page_visit", np.array([1, 1, 1, 2, 3]), np.array([0, 1, 2, DAY_NS, 2]))
    snapshot.add_events("search_query", np.array([1, 1]), np.array([2, DAY_NS + 3]))
    snapshot.add_events("remove_from_cart", np.array([2]), np.array([DAY_NS]))
    x = snapshot.finish()
    names = BASE_NAMES + POST_NAMES
    np.testing.assert_allclose(x[:, names.index("log_page_visit_after_last_buy")], np.log1p([2, 0, 0]))
    np.testing.assert_allclose(x[:, names.index("log_active_days_after_last_buy")], np.log1p([2, 0, 0]))
    np.testing.assert_array_equal(x[:, names.index("latest_event_buy_tied_with_other")], [0, 1, 0])
    np.testing.assert_array_equal(x[:, names.index("latest_event_buy_only")], [0, 0, 0])
    outcome = future_labels(purchases([(1, 10, 10), (2, 24, 10), (3, 25, 10)]), snapshot)
    np.testing.assert_array_equal(outcome[:, 1], [False, True, False])
    np.testing.assert_array_equal(outcome[:, 3], [False, False, True])
