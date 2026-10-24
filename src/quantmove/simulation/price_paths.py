"""Geometric Brownian Motion price-path simulation."""

import numpy as np


def simulate_price_paths(
    initial_price: float,
    annual_drift: float,
    annual_volatility: float,
    n_periods: int,
    n_simulations: int = 10_000,
    periods_per_year: int = 252,
    random_state: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate positive asset-price paths under geometric Brownian motion.

    Parameters
    ----------
    initial_price:
        Initial asset price; must be positive and finite.
    annual_drift:
        Annualized arithmetic expected-return parameter, expressed
        as a decimal.
    annual_volatility:
        Annualized volatility, expressed as a decimal; must be
        non-negative.
    n_periods:
        Number of future time steps.
    n_simulations:
        Number of independent paths to generate.
    periods_per_year:
        Number of time steps per year, e.g. 252 for trading days.
    random_state:
        Optional integer seed or NumPy Generator.

    Returns
    -------
    np.ndarray
        Array of shape (n_simulations, n_periods + 1).
        The first column contains the initial price.

    Notes
    -----
    Uses the exact discretization of geometric Brownian motion:

        S(t + dt) = S(t) * exp(
            (mu - 0.5 * sigma**2) * dt
            + sigma * sqrt(dt) * Z
        )

    where Z is standard normal. This assumes constant parameters
    and independent normally distributed Brownian increments.
    It does not capture jumps, volatility clustering, or fat tails.
    """
    numeric_parameters = (
        ("initial_price", initial_price),
        ("annual_drift", annual_drift),
        ("annual_volatility", annual_volatility),
    )

    for name, value in numeric_parameters:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value)
        ):
            raise ValueError(f"{name} must be a finite number.")

    if initial_price <= 0:
        raise ValueError("initial_price must be strictly positive.")

    if annual_volatility < 0:
        raise ValueError("annual_volatility must be non-negative.")

    for name, value in (
        ("n_periods", n_periods),
        ("n_simulations", n_simulations),
        ("periods_per_year", periods_per_year),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer.")

    if random_state is not None and not isinstance(
        random_state, (int, np.integer, np.random.Generator)
    ):
        raise ValueError(
            "random_state must be an integer, a NumPy Generator, or None."
        )

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )

    dt = 1.0 / periods_per_year
    shocks = rng.standard_normal((n_simulations, n_periods))

    increments = (
        (annual_drift - 0.5 * annual_volatility**2) * dt
        + annual_volatility * np.sqrt(dt) * shocks
    )

    log_growth = np.cumsum(increments, axis=1)

    paths = np.empty((n_simulations, n_periods + 1), dtype=float)
    paths[:, 0] = initial_price
    paths[:, 1:] = initial_price * np.exp(log_growth)

    return paths
