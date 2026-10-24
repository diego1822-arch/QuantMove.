import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, "/content/QuantMove/src")

from quantmove.risk.historical_var import (
    calculate_historical_var_es,
)


def test_calculates_historical_var_and_es():
    returns = pd.DataFrame({
        "A": [0.10, 0.00, -0.02, -0.04],
    })

    result = calculate_historical_var_es(
        returns, {"A": 1.0}, confidence=0.75
    )

    assert result["var"] == pytest.approx(0.025)
    assert result["expected_shortfall"] == pytest.approx(0.04)


def test_accepts_series_weights():
    returns = pd.DataFrame({
        "A": [0.01, -0.02, 0.03],
        "B": [-0.01, 0.02, -0.01],
    })

    result = calculate_historical_var_es(
        returns, pd.Series({"A": 0.6, "B": 0.4})
    )

    assert np.isfinite(result["var"])
    assert np.isfinite(result["expected_shortfall"])


def test_weight_order_does_not_change_result():
    returns = pd.DataFrame({
        "A": [0.01, -0.02, 0.03],
        "B": [-0.01, 0.02, -0.01],
    })

    first = calculate_historical_var_es(
        returns, {"A": 0.6, "B": 0.4}
    )
    second = calculate_historical_var_es(
        returns, {"B": 0.4, "A": 0.6}
    )

    assert first["var"] == pytest.approx(second["var"])
    assert first["expected_shortfall"] == pytest.approx(
        second["expected_shortfall"]
    )


def test_rejects_non_dataframe_returns():
    with pytest.raises(TypeError, match="DataFrame"):
        calculate_historical_var_es([0.01, -0.02], {"A": 1.0})


def test_rejects_empty_returns():
    with pytest.raises(ValueError, match="empty"):
        calculate_historical_var_es(pd.DataFrame(), {})


def test_rejects_missing_returns():
    returns = pd.DataFrame({"A": [0.01, np.nan]})

    with pytest.raises(ValueError, match="missing"):
        calculate_historical_var_es(returns, {"A": 1.0})


def test_rejects_infinite_returns():
    returns = pd.DataFrame({"A": [0.01, np.inf]})

    with pytest.raises(ValueError, match="finite"):
        calculate_historical_var_es(returns, {"A": 1.0})


def test_rejects_nonnumeric_returns():
    returns = pd.DataFrame({"A": [0.01, "invalid"]})

    with pytest.raises(ValueError, match="numeric"):
        calculate_historical_var_es(returns, {"A": 1.0})


def test_rejects_weights_that_do_not_sum_to_one():
    returns = pd.DataFrame({
        "A": [0.01, -0.02],
        "B": [-0.01, 0.02],
    })

    with pytest.raises(ValueError, match="sum to 1"):
        calculate_historical_var_es(
            returns, {"A": 0.5, "B": 0.4}
        )


def test_rejects_mismatched_asset_names():
    returns = pd.DataFrame({"A": [0.01, -0.02]})

    with pytest.raises(ValueError, match="match"):
        calculate_historical_var_es(
            returns, {"B": 1.0}
        )


def test_rejects_nonfinite_weights():
    returns = pd.DataFrame({
        "A": [0.01, -0.02],
        "B": [-0.01, 0.02],
    })

    with pytest.raises(ValueError, match="finite"):
        calculate_historical_var_es(
            returns, {"A": np.inf, "B": -np.inf}
        )


@pytest.mark.parametrize("confidence", [0, 1, -0.1, 1.1, np.nan])
def test_rejects_invalid_confidence(confidence):
    returns = pd.DataFrame({"A": [0.01, -0.02]})

    with pytest.raises(ValueError, match="Confidence"):
        calculate_historical_var_es(
            returns, {"A": 1.0}, confidence=confidence
        )


def test_requires_two_observations():
    returns = pd.DataFrame({"A": [0.01]})

    with pytest.raises(ValueError, match="At least two"):
        calculate_historical_var_es(returns, {"A": 1.0})
def test_expected_shortfall_with_one_full_tail_observation():
    returns = pd.DataFrame({
        "ASSET_A": [0.10, 0.05, 0.02, -0.01, -0.20]
    })

    result = calculate_historical_var_es(
        returns, {"ASSET_A": 1.0}, confidence=0.80
    )

    assert np.isclose(result["var"], 0.048)
    assert np.isclose(result["expected_shortfall"], 0.20)


def test_expected_shortfall_with_fractional_tail_observation():
    returns = pd.DataFrame({
        "ASSET_A": [-0.30, -0.20, -0.10, 0.00, 0.10, 0.20]
    })

    result = calculate_historical_var_es(
        returns, {"ASSET_A": 1.0}, confidence=0.75
    )

    assert np.isclose(result["var"], 0.175)
    assert np.isclose(
        result["expected_shortfall"], 0.40 / 1.5
    )
    assert result["expected_shortfall"] >= result["var"]
