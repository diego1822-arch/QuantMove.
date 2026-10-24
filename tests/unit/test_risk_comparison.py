import numpy as np
import pandas as pd
import pytest

from quantmove.portfolio.risk_comparison import compare_risk_estimators


@pytest.fixture
def historical_returns():
    rng = np.random.default_rng(2026)
    data = rng.multivariate_normal(
        mean=[0.0005, 0.0008],
        cov=[
            [0.0004, 0.00012],
            [0.00012, 0.0009],
        ],
        size=600,
    )
    dates = pd.date_range("2023-01-01", periods=600, freq="D")
    return pd.DataFrame(data, index=dates, columns=["ASSET_A", "ASSET_B"])


def test_comparison_returns_both_estimators_and_metadata(historical_returns):
    result = compare_risk_estimators(
        historical_returns,
        [0.6, 0.4],
        confidence=0.95,
        n_simulations=20_000,
        random_state=42,
    )

    assert set(result) == {
        "historical", "monte_carlo", "differences", "metadata"
    }
    assert set(result["historical"]) == {"var", "expected_shortfall"}
    assert set(result["monte_carlo"]) == {"var", "expected_shortfall"}
    assert result["metadata"]["n_assets"] == 2
    assert result["metadata"]["n_observations"] == 600
    assert result["metadata"]["n_simulations"] == 20_000
    assert result["metadata"]["risk_horizon"] == "one observation period"


def test_comparison_is_reproducible(historical_returns):
    first = compare_risk_estimators(
        historical_returns, [0.6, 0.4],
        n_simulations=5000, random_state=123,
    )
    second = compare_risk_estimators(
        historical_returns, [0.6, 0.4],
        n_simulations=5000, random_state=123,
    )

    assert first == second


def test_differences_match_returned_estimates(historical_returns):
    result = compare_risk_estimators(
        historical_returns, [0.6, 0.4],
        n_simulations=10_000, random_state=42,
    )

    for metric in ("var", "expected_shortfall"):
        expected = abs(
            result["monte_carlo"][metric] - result["historical"][metric]
        )
        assert result["differences"][metric]["absolute"] == pytest.approx(
            expected
        )


def test_rejects_non_dataframe_returns():
    with pytest.raises(ValueError, match="DataFrame"):
        compare_risk_estimators(
            np.array([[0.01], [-0.02], [0.03]]),
            [1.0],
            n_simulations=100,
            random_state=42,
        )


def test_rejects_non_finite_returns(historical_returns):
    broken = historical_returns.copy()
    broken.iloc[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        compare_risk_estimators(
            broken, [0.6, 0.4],
            n_simulations=100, random_state=42,
        )
