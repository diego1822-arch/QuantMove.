import numpy as np
import pytest

from quantmove.portfolio.monte_carlo_risk import (
    calculate_monte_carlo_var_es,
    simulate_correlated_returns,
)


def test_simulated_shape_for_multiple_assets():
    historical = np.array([
        [0.01, 0.02],
        [-0.02, -0.01],
        [0.03, 0.01],
        [-0.01, -0.03],
        [0.02, 0.04],
    ])

    result = simulate_correlated_returns(
        historical, n_simulations=500, random_state=42
    )

    assert result.shape == (500, 2)
    assert np.isfinite(result).all()


def test_simulation_is_reproducible():
    historical = np.array([
        [0.01, 0.02],
        [-0.02, -0.01],
        [0.03, 0.01],
        [-0.01, -0.03],
    ])

    first = simulate_correlated_returns(historical, 100, random_state=123)
    second = simulate_correlated_returns(historical, 100, random_state=123)

    np.testing.assert_array_equal(first, second)


def test_simulation_approximately_preserves_mean_and_covariance():
    rng = np.random.default_rng(7)
    historical = rng.multivariate_normal(
        [0.001, 0.002],
        [[0.0004, 0.00024], [0.00024, 0.0009]],
        size=20_000,
    )

    simulated = simulate_correlated_returns(
        historical, n_simulations=100_000, random_state=42
    )

    np.testing.assert_allclose(
        simulated.mean(axis=0), historical.mean(axis=0), atol=0.00015
    )
    np.testing.assert_allclose(
        np.cov(simulated, rowvar=False, ddof=1),
        np.cov(historical, rowvar=False, ddof=1),
        rtol=0.04,
        atol=1e-7,
    )


def test_single_asset_keeps_two_dimensional_shape():
    historical = np.array([[0.01], [-0.02], [0.03], [-0.01]])

    result = simulate_correlated_returns(historical, 50, random_state=42)

    assert result.shape == (50, 1)


def test_portfolio_risk_returns_expected_keys_and_finite_values():
    rng = np.random.default_rng(5)
    historical = rng.multivariate_normal(
        [0.0005, 0.0008],
        [[0.0004, 0.00015], [0.00015, 0.0009]],
        size=1000,
    )

    result = calculate_monte_carlo_var_es(
        historical, [0.6, 0.4], confidence=0.95,
        n_simulations=20_000, random_state=42,
    )

    assert set(result) == {"var", "expected_shortfall"}
    assert np.isfinite(result["var"])
    assert np.isfinite(result["expected_shortfall"])
    assert result["expected_shortfall"] >= result["var"]


def test_risk_estimate_is_reproducible():
    rng = np.random.default_rng(9)
    historical = rng.normal(0.001, 0.02, size=(500, 1))

    first = calculate_monte_carlo_var_es(
        historical, [1.0], n_simulations=5000, random_state=11
    )
    second = calculate_monte_carlo_var_es(
        historical, [1.0], n_simulations=5000, random_state=11
    )

    assert first == second


@pytest.mark.parametrize(
    "bad_returns",
    [
        np.array([0.01, 0.02]),
        np.array([[0.01, np.nan], [0.02, 0.03]]),
        np.array([[0.01, np.inf], [0.02, 0.03]]),
        np.empty((0, 2)),
        np.array([[0.01], [np.nan]]),
    ],
)
def test_rejects_invalid_historical_returns(bad_returns):
    with pytest.raises(ValueError):
        simulate_correlated_returns(bad_returns, n_simulations=10)


@pytest.mark.parametrize("n_simulations", [0, -1, 1.5, True, "100"])
def test_rejects_invalid_simulation_count(n_simulations):
    historical = np.array([[0.01], [-0.02], [0.03]])

    with pytest.raises(ValueError):
        simulate_correlated_returns(historical, n_simulations=n_simulations)


@pytest.mark.parametrize(
    "weights",
    [
        [0.5, 0.4],
        [0.5],
        [[0.5, 0.5]],
        [np.nan, np.nan],
        [np.inf, 0.0],
    ],
)
def test_rejects_invalid_weights(weights):
    historical = np.array([
        [0.01, 0.02],
        [-0.02, -0.01],
        [0.03, 0.01],
    ])

    with pytest.raises(ValueError):
        calculate_monte_carlo_var_es(
            historical, weights, n_simulations=100, random_state=42
        )


@pytest.mark.parametrize("confidence", [0, 1, -0.1, 1.1, np.nan, True, "0.95"])
def test_rejects_invalid_confidence(confidence):
    historical = np.array([[0.01], [-0.02], [0.03]])

    with pytest.raises(ValueError):
        calculate_monte_carlo_var_es(
            historical, [1.0], confidence=confidence,
            n_simulations=100, random_state=42,
        )


def test_rejects_invalid_random_state():
    historical = np.array([[0.01], [-0.02], [0.03]])

    with pytest.raises(ValueError):
        simulate_correlated_returns(
            historical, n_simulations=10, random_state="42"
        )
