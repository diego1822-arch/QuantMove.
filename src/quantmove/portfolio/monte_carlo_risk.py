"""Correlated Monte Carlo simulation and portfolio risk estimation."""

import numpy as np


def _validate_returns(returns):
    """Validate and normalize a historical returns matrix."""
    try:
        values = np.asarray(returns, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("returns must contain numeric values.") from exc

    if values.ndim != 2:
        raise ValueError("returns must be a two-dimensional matrix.")

    if values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError(
            "returns must contain at least two observations and one asset."
        )

    if not np.isfinite(values).all():
        raise ValueError("returns must contain only finite values.")

    return values


def simulate_correlated_returns(
    historical_returns,
    n_simulations=100_000,
    random_state=None,
):
    """Simulate joint one-period asset returns using a multivariate normal.

    Parameters
    ----------
    historical_returns:
        Two-dimensional matrix shaped (observations, assets).
        Rows are observations and columns are assets.
    n_simulations:
        Number of scenarios to generate.
    random_state:
        Optional integer seed or NumPy Generator.

    Returns
    -------
    np.ndarray
        Simulated returns shaped (n_simulations, n_assets).

    Notes
    -----
    The historical arithmetic mean vector and sample covariance matrix
    (ddof=1) parameterize a multivariate normal distribution.

    The model preserves linear dependence represented by the covariance
    matrix, but assumes normally distributed joint returns. It does not
    model serial dependence, volatility clustering, jumps, or fat tails.
    """
    values = _validate_returns(historical_returns)

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

    mean = values.mean(axis=0)
    covariance = np.atleast_2d(np.cov(values, rowvar=False, ddof=1))

    scenarios = rng.multivariate_normal(
        mean=mean,
        cov=covariance,
        size=int(n_simulations),
        check_valid="raise",
    )

    # NumPy may return a one-dimensional array for a single asset.
    return np.asarray(scenarios, dtype=float).reshape(
        int(n_simulations), values.shape[1]
    )


def calculate_monte_carlo_var_es(
    historical_returns,
    weights,
    confidence=0.95,
    n_simulations=100_000,
    random_state=None,
):
    """Estimate one-period portfolio VaR and Expected Shortfall.

    Weights must sum to one within numerical tolerance. Short positions
    are permitted. Risk is reported as a fraction of current portfolio
    value, not as a currency amount.

    VaR is the linear empirical quantile of simulated losses. Expected
    Shortfall is the average of the worst (1 - confidence) fraction of
    losses, with fractional weighting at the tail boundary.

    Returns
    -------
    dict
        Contains 'var' and 'expected_shortfall', both as decimal returns.
        Positive values represent losses; negative values represent gains.
    """
    values = _validate_returns(historical_returns)

    try:
        portfolio_weights = np.asarray(weights, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("weights must be a numeric one-dimensional array.") from exc

    if portfolio_weights.ndim != 1:
        raise ValueError("weights must be one-dimensional.")

    if portfolio_weights.size != values.shape[1]:
        raise ValueError("weights length must match the number of assets.")

    if not np.isfinite(portfolio_weights).all():
        raise ValueError("weights must contain only finite values.")

    if not np.isclose(portfolio_weights.sum(), 1.0, rtol=1e-9, atol=1e-9):
        raise ValueError("weights must sum to 1.")

    if (
        isinstance(confidence, (bool, np.bool_))
        or not isinstance(confidence, (int, float, np.integer, np.floating))
        or not np.isfinite(confidence)
        or not 0 < confidence < 1
    ):
        raise ValueError("confidence must be finite and strictly between 0 and 1.")

    simulated_returns = simulate_correlated_returns(
        historical_returns=values,
        n_simulations=n_simulations,
        random_state=random_state,
    )

    portfolio_returns = simulated_returns @ portfolio_weights
    losses = -portfolio_returns

    var = float(np.quantile(losses, confidence, method="linear"))

    # Average exactly the worst tail fraction, including a partial
    # weight for the boundary observation when needed.
    ordered_losses = np.sort(losses)[::-1]
    tail_count = (1.0 - float(confidence)) * ordered_losses.size
    full_count = int(np.floor(tail_count))
    fractional_count = tail_count - full_count

    tail_sum = float(ordered_losses[:full_count].sum())

    if fractional_count > 0:
        tail_sum += fractional_count * float(ordered_losses[full_count])

    expected_shortfall = tail_sum / tail_count

    return {
        "var": var,
        "expected_shortfall": float(expected_shortfall),
    }
