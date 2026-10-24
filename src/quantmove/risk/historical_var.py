import numpy as np
import pandas as pd


def calculate_historical_var_es(
    returns: pd.DataFrame,
    weights: dict | pd.Series,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Calculate historical portfolio VaR and ES per observation period.

    VaR is the linearly interpolated quantile of portfolio losses.
    ES is the average loss over the worst (1 - confidence) fraction
    of observations, using fractional weighting at the tail boundary.

    Losses are expressed as return fractions.
    """

    if not isinstance(returns, pd.DataFrame):
        raise TypeError("Returns must be a pandas DataFrame.")

    if returns.empty:
        raise ValueError("The returns dataset cannot be empty.")

    if returns.columns.has_duplicates:
        raise ValueError("Asset names must be unique.")

    if returns.isna().any().any():
        raise ValueError("Returns cannot contain missing values.")

    try:
        numeric_returns = returns.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Returns must contain numeric values."
        ) from error

    if not np.isfinite(numeric_returns.to_numpy()).all():
        raise ValueError("Returns must be finite numbers.")

    if len(numeric_returns) < 2:
        raise ValueError(
            "At least two observations are required "
            "to calculate historical VaR and ES."
        )

    if not isinstance(weights, (dict, pd.Series)):
        raise TypeError("Weights must be a dictionary or pandas Series.")

    portfolio_weights = pd.Series(weights, dtype=float)

    if portfolio_weights.index.has_duplicates:
        raise ValueError("Weight asset names must be unique.")

    if set(portfolio_weights.index) != set(numeric_returns.columns):
        raise ValueError(
            "Weights must match the return asset names exactly."
        )

    if not np.isfinite(portfolio_weights.to_numpy()).all():
        raise ValueError("Weights must be finite numbers.")

    if not np.isclose(portfolio_weights.sum(), 1.0):
        raise ValueError("Portfolio weights must sum to 1.")

    if (
        not isinstance(confidence, (int, float, np.integer, np.floating))
        or isinstance(confidence, (bool, np.bool_))
        or not np.isfinite(confidence)
        or not 0 < confidence < 1
    ):
        raise ValueError("Confidence must be between 0 and 1.")

    portfolio_weights = portfolio_weights.reindex(
        numeric_returns.columns
    )

    portfolio_returns = (
        numeric_returns @ portfolio_weights
    ).to_numpy()

    losses = -portfolio_returns

    var = float(np.quantile(losses, confidence, method="linear"))

    # Expected Shortfall using fractional empirical tail weighting.
    sorted_losses = np.sort(losses)[::-1]
    tail_mass = (1.0 - float(confidence)) * len(sorted_losses)

    whole_observations = int(np.floor(tail_mass))
    fractional_observation = tail_mass - whole_observations

    tail_sum = float(sorted_losses[:whole_observations].sum())

    if (
        fractional_observation > 0
        and whole_observations < len(sorted_losses)
    ):
        tail_sum += (
            fractional_observation
            * sorted_losses[whole_observations]
        )

    expected_shortfall = float(tail_sum / tail_mass)

    return {
        "var": var,
        "expected_shortfall": expected_shortfall,
    }
