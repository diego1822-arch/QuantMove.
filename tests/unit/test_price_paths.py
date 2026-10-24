import numpy as np
import pytest

from quantmove.simulation.price_paths import simulate_price_paths


def test_output_shape_and_initial_prices():
    paths = simulate_price_paths(100, 0.08, 0.2, 20, 500, random_state=42)

    assert paths.shape == (500, 21)
    np.testing.assert_array_equal(paths[:, 0], np.full(500, 100.0))


def test_same_seed_produces_same_paths():
    first = simulate_price_paths(100, 0.08, 0.2, 30, 100, random_state=123)
    second = simulate_price_paths(100, 0.08, 0.2, 30, 100, random_state=123)

    np.testing.assert_array_equal(first, second)


def test_all_simulated_prices_are_positive_and_finite():
    paths = simulate_price_paths(100, 0.08, 0.3, 252, 1000, random_state=42)

    assert np.isfinite(paths).all()
    assert (paths > 0).all()


def test_zero_volatility_gives_deterministic_paths():
    initial_price = 100.0
    drift = 0.10
    periods = 12
    periods_per_year = 12

    paths = simulate_price_paths(
        initial_price,
        drift,
        0.0,
        periods,
        5,
        periods_per_year=periods_per_year,
        random_state=42,
    )

    expected = initial_price * np.exp(
        drift * np.arange(periods + 1) / periods_per_year
    )

    np.testing.assert_allclose(
        paths, np.tile(expected, (5, 1)), rtol=1e-12, atol=1e-12
    )


def test_terminal_mean_approximates_gbm_theoretical_expectation():
    initial_price = 100.0
    drift = 0.08
    years = 1
    periods = 252

    paths = simulate_price_paths(
        initial_price,
        drift,
        0.2,
        periods,
        100_000,
        periods_per_year=252,
        random_state=42,
    )

    theoretical_mean = initial_price * np.exp(drift * years)

    assert abs(paths[:, -1].mean() - theoretical_mean) < 0.25


@pytest.mark.parametrize("initial_price", [0, -1, np.nan, np.inf, True, "100"])
def test_rejects_invalid_initial_price(initial_price):
    with pytest.raises(ValueError):
        simulate_price_paths(initial_price, 0.08, 0.2, 10)


@pytest.mark.parametrize("annual_drift", [np.nan, np.inf, -np.inf, True, "0.08"])
def test_rejects_invalid_drift(annual_drift):
    with pytest.raises(ValueError):
        simulate_price_paths(100, annual_drift, 0.2, 10)


@pytest.mark.parametrize(
    "annual_volatility", [-0.1, np.nan, np.inf, -np.inf, True, "0.2"]
)
def test_rejects_invalid_volatility(annual_volatility):
    with pytest.raises(ValueError):
        simulate_price_paths(100, 0.08, annual_volatility, 10)


@pytest.mark.parametrize("n_periods", [0, -1, 1.5, True, "10"])
def test_rejects_invalid_n_periods(n_periods):
    with pytest.raises(ValueError):
        simulate_price_paths(100, 0.08, 0.2, n_periods)


@pytest.mark.parametrize("n_simulations", [0, -1, 1.5, True, "100"])
def test_rejects_invalid_n_simulations(n_simulations):
    with pytest.raises(ValueError):
        simulate_price_paths(100, 0.08, 0.2, 10, n_simulations=n_simulations)


@pytest.mark.parametrize("periods_per_year", [0, -1, 1.5, True, "252"])
def test_rejects_invalid_periods_per_year(periods_per_year):
    with pytest.raises(ValueError):
        simulate_price_paths(
            100, 0.08, 0.2, 10, periods_per_year=periods_per_year
        )


def test_rejects_invalid_random_state():
    with pytest.raises(ValueError):
        simulate_price_paths(100, 0.08, 0.2, 10, random_state="42")


def test_accepts_numpy_generator():
    rng = np.random.default_rng(42)

    paths = simulate_price_paths(
        100, 0.08, 0.2, 10, n_simulations=25, random_state=rng
    )

    assert paths.shape == (25, 11)
    assert np.isfinite(paths).all()
