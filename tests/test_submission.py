import numpy as np
import pandas as pd
import pytest

from basic.submission import validate_submission


@pytest.fixture
def files(tmp_path):
    sample = tmp_path / "sample.csv"
    test = tmp_path / "test.csv"
    submission = tmp_path / "submission.csv"
    pd.DataFrame({"id": [2, 5, 8], "target": [0, 0, 0]}).to_csv(sample, index=False)
    pd.DataFrame({"id": [2, 5, 8], "msno": ["a", "b", "a"]}).to_csv(test, index=False)
    pd.DataFrame({"id": [2, 5, 8], "target": [0, .4, 1]}).to_csv(submission, index=False)
    return submission, sample, test


def test_valid_submission(files):
    report = validate_submission(*files)
    assert report["rows"] == 3 and report["valid"]


@pytest.mark.parametrize("ids,targets", [([2, 5], [0.1, 0.2]), ([2, 5, 5], [.1, .2, .3]),
    ([5, 2, 8], [.1, .2, .3]), ([2, 5, 8], [.1, np.nan, .3]),
    ([2, 5, 8], [.1, np.inf, .3]), ([2, 5, 8], [.1, -0.1, 1.1])])
def test_rejects_invalid_predictions(files, ids, targets):
    pd.DataFrame({"id": ids, "target": targets}).to_csv(files[0], index=False)
    with pytest.raises(ValueError):
        validate_submission(*files)


def test_rejects_extra_index_column(files):
    pd.read_csv(files[0]).to_csv(files[0], index=True)
    with pytest.raises(ValueError, match="header"):
        validate_submission(*files)


def test_rejects_wrong_sample(files):
    pd.DataFrame({"id": [2, 5, 9], "msno": ["a", "b", "a"]}).to_csv(files[2], index=False)
    with pytest.raises(ValueError, match="cover"):
        validate_submission(*files)


def test_rejects_extra_row_field_even_when_pandas_would_infer_index(files):
    files[0].write_text("id,target\n99,2,0.2\n98,5,0.4\n97,8,0.3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="two CSV fields"):
        validate_submission(*files)
