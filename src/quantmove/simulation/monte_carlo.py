"""Monte Carlo simulation utilities."""

import numpy as np


def simulate_normal_returns(
    mean_return: float,
    volatility: float,
    n_simulations: int = 10_000,
    random_state: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate one-period returns under a normal distribution.

    Parameters
    ----------
    mean_return:
        Arithmetic mean return per period, expressed as a decimal.
    volatility:
        Standard deviation of returns per period, expressed as a decimal.
        Must be non-negative.
    n_simulations:
        Number of independent return scenarios to generate.
    random_state:
        Optional integer seed or NumPy Generator for reproducibility.

    Returns
    -------
    np.ndarray
        One-dimensional array containing simulated returns.

    Notes
    -----
    This model assumes normally distributed, independent returns.
    It is a modeling assumption, not a guarantee about market behavior.
    """
    for name, value in (
        ("mean_return", mean_return),
        ("volatility", volatility),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value)
        ):
            raise ValueError(f"{name} must be a finite number.")

    if volatility < 0:
        raise ValueError("volatility must be non-negative.")

    if (
        isinstance(n_simulations, (bool, np.bool_))
        or not isinstance(n_simulations, (int, np.integer))
        or n_simulations < 1
    ):
        raise ValueError("n_simulations must be a positive integer.")

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

    simulated_returns = rng.normal(
        loc=float(mean_return),
        scale=float(volatility),
        size=int(n_simulations),
    )

    return simulated_returns
