import numpy as np
import pandas as pd


def calculate_correlation(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate the Pearson correlation matrix of asset returns."""

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
            "to calculate correlation."
        )

    numeric_returns = returns.astype(float)

    if (numeric_returns.std(ddof=1) == 0).any():
        raise ValueError(
            "Correlation is undefined for constant return series."
        )

    correlation = numeric_returns.corr(method="pearson")

    if not np.isfinite(correlation.to_numpy()).all():
        raise ValueError(
            "The correlation matrix contains invalid values."
        )

    return correlation
