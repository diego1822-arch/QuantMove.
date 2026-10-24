import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, "/content/QuantMove/src")

from quantmove.portfolio.portfolio_risk import (
    calculate_portfolio_volatility,
)


def make_returns():
    return pd.DataFrame({
        "A": [0.01, 0.03, -0.02, 0.04],
        "B": [-0.01, 0.02, 0.01, -0.03],
    })


def test_calculates_portfolio_volatility():
    returns = make_returns()
    weights = {"A": 0.6, "B": 0.4}

    result = calculate_portfolio_volatility(returns, weights)

    covariance = returns.cov(ddof=1).to_numpy()
    vector = np.array([0.6, 0.4])
    expected = np.sqrt(vector @ covariance @ vector)

    assert result == pytest.approx(expected)
    assert result >= 0


def test_weights_can_be_a_series():
    returns = make_returns()
    weights = pd.Series({"A": 0.6, "B": 0.4})

    result = calculate_portfolio_volatility(returns, weights)

    assert np.isfinite(result)


def test_weight_order_does_not_change_result():
    returns = make_returns()

    first = calculate_portfolio_volatility(
        returns, {"A": 0.6, "B": 0.4}
    )
    second = calculate_portfolio_volatility(
        returns, {"B": 0.4, "A": 0.6}
    )

    assert first == pytest.approx(second)


def test_rejects_non_dataframe_returns():
    with pytest.raises(TypeError, match="DataFrame"):
        calculate_portfolio_volatility([0.01, 0.02], {"A": 1.0})


def test_rejects_empty_returns():
    with pytest.raises(ValueError, match="empty"):
        calculate_portfolio_volatility(
            pd.DataFrame(), {}
        )


def test_rejects_missing_returns():
    returns = make_returns()
    returns.loc[0, "A"] = np.nan

    with pytest.raises(ValueError, match="missing"):
        calculate_portfolio_volatility(
            returns, {"A": 0.6, "B": 0.4}
        )


def test_rejects_infinite_returns():
    returns = make_returns()
    returns.loc[0, "A"] = np.inf

    with pytest.raises(ValueError, match="finite"):
        calculate_portfolio_volatility(
            returns, {"A": 0.6, "B": 0.4}
        )


def test_rejects_weights_that_do_not_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        calculate_portfolio_volatility(
            make_returns(), {"A": 0.5, "B": 0.4}
        )


def test_rejects_mismatched_asset_names():
    with pytest.raises(ValueError, match="match"):
        calculate_portfolio_volatility(
            make_returns(), {"A": 0.6, "C": 0.4}
        )


def test_rejects_nonfinite_weights():
    with pytest.raises(ValueError, match="finite"):
        calculate_portfolio_volatility(
            make_returns(), {"A": np.inf, "B": -np.inf}
        )


def test_rejects_only_one_observation():
    returns = pd.DataFrame({"A": [0.01]})

    with pytest.raises(ValueError, match="At least two"):
        calculate_portfolio_volatility(
            returns, {"A": 1.0}
        )
