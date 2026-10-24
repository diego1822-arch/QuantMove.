
import numpy as np
import pandas as pd


def validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Validate historical financial price data."""

    if not isinstance(prices, pd.DataFrame):
        raise TypeError("Prices must be a pandas DataFrame.")

    if prices.empty:
        raise ValueError("The price dataset cannot be empty.")

    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("Prices must use a DatetimeIndex.")

    if prices.index.has_duplicates:
        raise ValueError("Price timestamps must be unique.")

    if not prices.index.is_monotonic_increasing:
        raise ValueError("Prices must be ordered chronologically.")

    if prices.columns.has_duplicates:
        raise ValueError("Asset names must be unique.")

    if prices.isna().any().any():
        raise ValueError("Prices cannot contain missing values.")

    try:
        values = prices.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Prices must contain numeric values.") from error

    if not np.isfinite(values).all():
        raise ValueError("Prices must be finite numbers.")

    if (values <= 0).any():
        raise ValueError("Asset prices must be greater than zero.")

    return prices.astype(float).copy()


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate simple percentage returns."""

    clean_prices = validate_prices(prices)

    if len(clean_prices) < 2:
        raise ValueError(
            "At least two price observations are required "
            "to calculate returns."
        )

    returns = clean_prices.pct_change().iloc[1:]

    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Calculated returns contain invalid values.")

    return returns
