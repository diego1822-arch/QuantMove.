import numpy as np
import pandas as pd
import pytest

from quantmove.portfolio.historical_risk_pipeline import (
    run_historical_risk_pipeline,
)


def make_prices(n_prices=90):
    rng = np.random.default_rng(123)
    returns = rng.normal(
        loc=[0.0003, 0.0002],
        scale=[0.01, 0.015],
        size=(n_prices - 1, 2),
    )
    initial = np.array([100.0, 200.0])
    values = np.vstack([
        initial,
        initial * np.cumprod(1 + returns, axis=0),
    ])
    return pd.DataFrame(
        values,
        index=pd.bdate_range("2025-01-02", periods=n_prices),
        columns=["ASSET_A", "ASSET_B"],
    )


def test_pipeline_integrates_risk_and_backtesting():
    result = run_historical_risk_pipeline(
        prices=make_prices(90),
        weights={"ASSET_A": 0.6, "ASSET_B": 0.4},
        window_size=30,
        n_simulations=1000,
        random_state=42,
    )

    assert result["metadata"]["n_price_observations"] == 90
    assert result["metadata"]["n_return_observations"] == 89
    assert result["metadata"]["n_assets"] == 2
    assert result["metadata"]["backtest_status"] == "completed"
    assert len(result["backtest"]["observations"]) == 59
    assert result["backtest_report"]["n_observations"] == 59
    assert result["correlation"].shape == (2, 2)
    assert np.isfinite(result["portfolio_volatility"])
    assert np.isfinite(result["historical_var_es"]["var"])
    assert np.isfinite(
        result["historical_var_es"]["expected_shortfall"]
    )


def test_pipeline_skips_backtest_when_history_is_insufficient():
    result = run_historical_risk_pipeline(
        prices=make_prices(30),
        weights={"ASSET_A": 0.6, "ASSET_B": 0.4},
        window_size=250,
        n_simulations=1000,
        random_state=42,
    )

    assert result["metadata"]["backtest_status"] == (
        "skipped_insufficient_history"
    )
    assert result["backtest"] is None
    assert result["backtest_report"] is None
    assert len(result["returns"]) == 29


def test_pipeline_rejects_weights_that_do_not_sum_to_one():
    with pytest.raises(ValueError):
        run_historical_risk_pipeline(
            prices=make_prices(30),
            weights={"ASSET_A": 0.6, "ASSET_B": 0.3},
            window_size=10,
            n_simulations=1000,
            random_state=42,
        )


def test_pipeline_rejects_nonpositive_prices():
    prices = make_prices(30)
    prices.iloc[5, 0] = 0.0

    with pytest.raises(ValueError):
        run_historical_risk_pipeline(
            prices=prices,
            weights={"ASSET_A": 0.6, "ASSET_B": 0.4},
            window_size=10,
            n_simulations=1000,
            random_state=42,
        )



@pytest.mark.parametrize(
    "kwargs",
    [
        {"confidence": 0},
        {"confidence": 1},
        {"confidence": 1.5},
        {"confidence": float("nan")},
        {"confidence": float("inf")},
        {"significance_level": 0},
        {"significance_level": 1},
        {"significance_level": float("nan")},
        {"window_size": 1},
        {"window_size": 2.5},
        {"window_size": True},
        {"n_simulations": 0},
        {"n_simulations": -10},
        {"n_simulations": 2.5},
        {"n_simulations": True},
        {"random_state": -1},
        {"random_state": 1.5},
        {"random_state": True},
    ],
)
def test_pipeline_rejects_invalid_parameters_even_with_short_history(kwargs):
    params = {
        "prices": make_prices(30),
        "weights": {"ASSET_A": 0.6, "ASSET_B": 0.4},
        "window_size": 250,
        "n_simulations": 1000,
        "random_state": 42,
    }

    # Sobrescribir el parámetro que se está probando,
    # sin pasarlo dos veces a la función.
    params.update(kwargs)

    with pytest.raises(ValueError):
        run_historical_risk_pipeline(**params)


def test_pipeline_accepts_valid_integer_parameters():
    result = run_historical_risk_pipeline(
        prices=make_prices(30),
        weights={"ASSET_A": 0.6, "ASSET_B": 0.4},
        confidence=0.95,
        window_size=250,
        significance_level=0.05,
        n_simulations=1000,
        random_state=42,
    )

    assert result["metadata"]["window_size"] == 250
    assert result["metadata"]["backtest_status"] == (
        "skipped_insufficient_history"
    )


def test_pipeline_is_reproducible_with_fixed_random_state():
    kwargs = {
        "prices": make_prices(90),
        "weights": {"ASSET_A": 0.6, "ASSET_B": 0.4},
        "window_size": 30,
        "n_simulations": 1000,
        "random_state": 42,
    }

    first = run_historical_risk_pipeline(**kwargs)
    second = run_historical_risk_pipeline(**kwargs)

    for section in ("historical", "monte_carlo"):
        for metric in ("var", "expected_shortfall"):
            assert (
                first["estimator_comparison"][section][metric]
                == second["estimator_comparison"][section][metric]
            )

    assert (
        first["portfolio_volatility"]
        == second["portfolio_volatility"]
    )
