import numpy as np
import pandas as pd
import pytest

from quantmove.risk.historical_var import calculate_historical_var_es
from quantmove.risk.var_backtesting import (
    backtest_historical_var,
    kupiec_test,
)


def test_backtest_uses_only_prior_observations():
    returns = pd.DataFrame(
        {"A": [0.01, -0.02, 0.015, -0.01, -0.50, 0.02, -0.03]}
    )
    result = backtest_historical_var(
        returns, [1.0], window_size=4, confidence=0.75
    )

    expected = calculate_historical_var_es(
        returns.iloc[:4],
        pd.Series([1.0], index=["A"]),
        confidence=0.75,
    )

    assert result["observations"].iloc[0]["var"] == pytest.approx(
        expected["var"]
    )
    # The large loss on the first test date must not affect its own VaR.
    assert result["observations"].index[0] == returns.index[4]


def test_backtest_output_and_summary():
    rng = np.random.default_rng(123)
    returns = pd.DataFrame(
        rng.normal(0.0002, 0.01, size=(120, 2)),
        columns=["A", "B"],
    )
    result = backtest_historical_var(
        returns, {"A": 0.6, "B": 0.4},
        window_size=50, confidence=0.95,
    )

    observations = result["observations"]
    summary = result["summary"]

    assert len(observations) == 70
    assert list(observations.columns) == ["var", "realized_loss", "exception"]
    assert summary["n_backtest_observations"] == 70
    assert summary["n_exceptions"] == int(observations["exception"].sum())
    assert summary["method"] == "historical_var"


def test_kupiec_zero_exceptions_is_finite():
    result = kupiec_test([0] * 100, confidence=0.95)

    assert result["n_exceptions"] == 0
    assert np.isfinite(result["likelihood_ratio"])
    assert np.isfinite(result["p_value"])
    assert 0 <= result["p_value"] <= 1


def test_kupiec_expected_rate_does_not_reject():
    # 5 exceptions out of 100 matches the 5% null rate exactly.
    result = kupiec_test([1] * 5 + [0] * 95, confidence=0.95)

    assert result["observed_exception_rate"] == pytest.approx(0.05)
    assert result["likelihood_ratio"] == pytest.approx(0.0)
    assert result["p_value"] == pytest.approx(1.0)
    assert result["reject_null"] is False


@pytest.mark.parametrize(
    "exceptions",
    [[], [[0, 1]], [0, 2], [0, np.nan], ["bad", 0]],
)
def test_kupiec_rejects_invalid_exceptions(exceptions):
    with pytest.raises(ValueError):
        kupiec_test(exceptions)


@pytest.mark.parametrize("window_size", [0, 1, 2.5, True])
def test_backtest_rejects_invalid_window_size(window_size):
    returns = pd.DataFrame({"A": [0.01, -0.02, 0.03, -0.01]})

    with pytest.raises(ValueError):
        backtest_historical_var(
            returns, [1.0], window_size=window_size
        )


def test_backtest_rejects_insufficient_data():
    returns = pd.DataFrame({"A": [0.01, -0.02, 0.03]})

    with pytest.raises(ValueError):
        backtest_historical_var(returns, [1.0], window_size=3)


def test_backtest_rejects_non_finite_returns():
    returns = pd.DataFrame({"A": [0.01, -0.02, np.nan, 0.03]})

    with pytest.raises(ValueError):
        backtest_historical_var(returns, [1.0], window_size=2)


def test_backtest_rejects_weights_not_summing_to_one():
    returns = pd.DataFrame(
        {"A": [0.01, -0.02, 0.03, -0.01, 0.02]}
    )

    with pytest.raises(ValueError):
        backtest_historical_var(returns, [0.7], window_size=2)
