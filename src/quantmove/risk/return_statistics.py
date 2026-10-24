import numpy as np
import pandas as pd


def calculate_return_statistics(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate descriptive statistics for asset returns."""

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
            "At least two return observations are required "
            "to calculate sample volatility."
        )

    numeric_returns = returns.astype(float)

    statistics = pd.DataFrame({
        "mean": numeric_returns.mean(),
        "volatility": numeric_returns.std(ddof=1),
        "minimum": numeric_returns.min(),
        "maximum": numeric_returns.max(),
    })

    return statistics
