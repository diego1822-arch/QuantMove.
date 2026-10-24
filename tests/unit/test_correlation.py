import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, "/content/QuantMove/src")

from quantmove.risk.correlation import calculate_correlation


def make_returns(values):
    return pd.DataFrame(values)


def test_calculates_correlation_correctly():
    returns = make_returns({
        "ASSET_A": [1, 2, 3, 4],
        "ASSET_B": [2, 4, 6, 8],
        "ASSET_C": [4, 3, 2, 1],
    })

    result = calculate_correlation(returns)

    assert result.loc["ASSET_A", "ASSET_B"] == pytest.approx(1.0)
    assert result.loc["ASSET_A", "ASSET_C"] == pytest.approx(-1.0)
    assert result.loc["ASSET_B", "ASSET_C"] == pytest.approx(-1.0)


def test_correlation_matrix_is_symmetric():
    returns = make_returns({
        "A": [1, 3, 2, 5],
        "B": [2, 1, 4, 3],
        "C": [5, 2, 3, 1],
    })

    result = calculate_correlation(returns)

    np.testing.assert_allclose(result, result.T)


def test_diagonal_values_are_one():
    returns = make_returns({
        "A": [1, 2, 4],
        "B": [5, 3, 2],
    })

    result = calculate_correlation(returns)

    np.testing.assert_allclose(np.diag(result), [1.0, 1.0])


def test_rejects_non_dataframe():
    with pytest.raises(TypeError):
        calculate_correlation([0.01, 0.02])


def test_rejects_empty_dataframe():
    with pytest.raises(ValueError, match="empty"):
        calculate_correlation(pd.DataFrame())


def test_rejects_missing_returns():
    returns = make_returns({"A": [0.01, np.nan]})

    with pytest.raises(ValueError, match="missing"):
        calculate_correlation(returns)


def test_rejects_infinite_returns():
    returns = make_returns({"A": [0.01, np.inf]})

    with pytest.raises(ValueError, match="finite"):
        calculate_correlation(returns)


def test_rejects_nonnumeric_returns():
    returns = make_returns({"A": [0.01, "abc"]})

    with pytest.raises(ValueError, match="numeric"):
        calculate_correlation(returns)


def test_rejects_duplicate_asset_names():
    returns = pd.DataFrame(
        [[0.01, 0.02], [0.03, 0.04]],
        columns=["A", "A"],
    )

    with pytest.raises(ValueError, match="Asset names"):
        calculate_correlation(returns)


def test_requires_two_observations():
    returns = make_returns({"A": [0.01]})

    with pytest.raises(ValueError, match="At least two"):
        calculate_correlation(returns)


def test_rejects_constant_return_series():
    returns = make_returns({
        "A": [0.01, 0.01, 0.01],
        "B": [0.02, 0.03, 0.04],
    })

    with pytest.raises(ValueError, match="constant"):
        calculate_correlation(returns)
