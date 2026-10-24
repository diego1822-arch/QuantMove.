import numpy as np
import pandas as pd


def calculate_portfolio_volatility(
    returns: pd.DataFrame,
    weights: dict | pd.Series,
) -> float:
    """Calculate portfolio volatility per return period."""

    if not isinstance(returns, pd.DataFrame):
        raise TypeError("Returns must be a pandas DataFrame.")

    if returns.empty:
        raise ValueError("The returns dataset cannot be empty.")

    if returns.columns.has_duplicates:
        raise ValueError("Asset names must be unique.")

    if returns.isna().any().any():
        raise ValueError("Returns cannot contain missing values.")

    try:
        values = returns.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Returns must contain numeric values."
        ) from error

    if not np.isfinite(values).all():
        raise ValueError("Returns must be finite numbers.")

    if len(returns) < 2:
        raise ValueError(
            "At least two observations are required "
            "to calculate portfolio volatility."
        )

    if not isinstance(weights, (dict, pd.Series)):
        raise TypeError("Weights must be a dictionary or pandas Series.")

    portfolio_weights = pd.Series(weights, dtype=float)

    if portfolio_weights.index.has_duplicates:
        raise ValueError("Weight asset names must be unique.")

    if set(portfolio_weights.index) != set(returns.columns):
        raise ValueError(
            "Weights must match the return asset names exactly."
        )

    if not np.isfinite(portfolio_weights.to_numpy()).all():
        raise ValueError("Weights must be finite numbers.")

    if not np.isclose(portfolio_weights.sum(), 1.0):
        raise ValueError("Portfolio weights must sum to 1.")

    portfolio_weights = portfolio_weights.reindex(returns.columns)
    covariance = returns.astype(float).cov(ddof=1).to_numpy()
    weight_values = portfolio_weights.to_numpy()

    variance = float(
        weight_values @ covariance @ weight_values
    )

    if variance < -1e-12:
        raise ValueError("Calculated portfolio variance is invalid.")

    return float(np.sqrt(max(variance, 0.0)))
