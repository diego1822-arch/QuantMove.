import numpy as np
import pytest

from quantmove.simulation.monte_carlo import simulate_normal_returns


def test_returns_requested_number_of_simulations():
    result = simulate_normal_returns(0.001, 0.02, n_simulations=500, random_state=42)

    assert isinstance(result, np.ndarray)
    assert result.shape == (500,)


def test_same_seed_produces_same_results():
    first = simulate_normal_returns(0.001, 0.02, 100, random_state=123)
    second = simulate_normal_returns(0.001, 0.02, 100, random_state=123)

    np.testing.assert_array_equal(first, second)


def test_zero_volatility_returns_constant_values():
    result = simulate_normal_returns(0.01, 0.0, 20, random_state=42)

    np.testing.assert_array_equal(result, np.full(20, 0.01))


def test_simulated_mean_and_volatility_approximate_parameters():
    mean_return = 0.001
    volatility = 0.02

    result = simulate_normal_returns(
        mean_return, volatility, n_simulations=100_000, random_state=42
    )

    assert abs(result.mean() - mean_return) < 0.0002
    assert abs(result.std(ddof=0) - volatility) < 0.0002


@pytest.mark.parametrize("mean_return", [np.nan, np.inf, -np.inf, True, "0.01"])
def test_rejects_invalid_mean_return(mean_return):
    with pytest.raises(ValueError):
        simulate_normal_returns(mean_return, 0.02)


@pytest.mark.parametrize("volatility", [-0.01, np.nan, np.inf, -np.inf, True])
def test_rejects_invalid_volatility(volatility):
    with pytest.raises(ValueError):
        simulate_normal_returns(0.001, volatility)


@pytest.mark.parametrize("n_simulations", [0, -1, 1.5, True, "100"])
def test_rejects_invalid_number_of_simulations(n_simulations):
    with pytest.raises(ValueError):
        simulate_normal_returns(0.001, 0.02, n_simulations=n_simulations)


def test_rejects_invalid_random_state():
    with pytest.raises(ValueError):
        simulate_normal_returns(0.001, 0.02, random_state="42")


def test_accepts_numpy_generator():
    rng = np.random.default_rng(42)

    result = simulate_normal_returns(
        0.001, 0.02, n_simulations=25, random_state=rng
    )

    assert result.shape == (25,)
    assert np.isfinite(result).all()
