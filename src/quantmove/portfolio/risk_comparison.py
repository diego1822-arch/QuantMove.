"""Compare historical and Monte Carlo portfolio risk estimates."""

import numpy as np
import pandas as pd

from quantmove.portfolio.monte_carlo_risk import (
    calculate_monte_carlo_var_es,
)
from quantmove.risk.historical_var import calculate_historical_var_es


def compare_risk_estimators(
    returns,
    weights,
    confidence=0.95,
    n_simulations=100_000,
    random_state=None,
):
    """Compare historical and Monte Carlo one-period portfolio risk.

    Parameters
    ----------
    returns:
        DataFrame of historical asset returns. Rows are observations
        and columns are assets.
    weights:
        Portfolio weights in the same order as returns columns.
        Must sum to one.
    confidence:
        Confidence level strictly between zero and one.
    n_simulations:
        Number of Monte Carlo scenarios.
    random_state:
        Optional integer seed or NumPy Generator.

    Returns
    -------
    dict
        Historical estimates, Monte Carlo estimates, absolute differences,
        relative differences, and comparison metadata.

        Relative differences are None when the historical estimate is
        numerically zero. They are descriptive comparisons, not measures
        of model accuracy.
    """
    if not isinstance(returns, pd.DataFrame):
        raise ValueError("returns must be a pandas DataFrame.")

    if returns.empty or returns.shape[1] == 0:
        raise ValueError("returns must contain observations and assets.")

    if not returns.columns.is_unique:
        raise ValueError("returns must have unique asset columns.")

    try:
        numeric_returns = returns.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("returns must contain numeric values.") from exc

    if not np.isfinite(numeric_returns).all():
        raise ValueError("returns must contain only finite values.")

    # Normalize weights while preserving the asset-column order.
    if isinstance(weights, pd.Series):
        if not weights.index.equals(returns.columns):
            if (
                weights.index.is_unique
                and set(weights.index) == set(returns.columns)
            ):
                weights = weights.reindex(returns.columns)
            else:
                raise ValueError(
                    "Weight labels must match the return columns."
                )
        weight_series = weights.astype(float)
    elif isinstance(weights, dict):
        if set(weights) != set(returns.columns):
            raise ValueError(
                "Weight keys must match the return columns."
            )
        weight_series = pd.Series(weights, dtype=float).reindex(
            returns.columns
        )
    else:
        weight_array = np.asarray(weights, dtype=float)
        if weight_array.ndim != 1 or len(weight_array) != returns.shape[1]:
            raise ValueError(
                "weights must contain one value per asset."
            )
        weight_series = pd.Series(
            weight_array, index=returns.columns, dtype=float
        )

    weight_array = weight_series.to_numpy(dtype=float)

    # Each estimator receives the weight format it expects.
    historical = calculate_historical_var_es(
        returns,
        weight_series,
        confidence=confidence,
    )

    monte_carlo = calculate_monte_carlo_var_es(
        numeric_returns,
        weight_array,
        confidence=confidence,
        n_simulations=n_simulations,
        random_state=random_state,
    )

    differences = {}
    for metric in ("var", "expected_shortfall"):
        historical_value = float(historical[metric])
        monte_carlo_value = float(monte_carlo[metric])
        absolute_difference = abs(monte_carlo_value - historical_value)

        if np.isclose(historical_value, 0.0, atol=1e-12, rtol=0.0):
            relative_difference = None
        else:
            relative_difference = (
                absolute_difference / abs(historical_value)
            )

        differences[metric] = {
            "absolute": float(absolute_difference),
            "relative": (
                None
                if relative_difference is None
                else float(relative_difference)
            ),
        }

    return {
        "historical": {
            "var": float(historical["var"]),
            "expected_shortfall": float(historical["expected_shortfall"]),
        },
        "monte_carlo": {
            "var": float(monte_carlo["var"]),
            "expected_shortfall": float(
                monte_carlo["expected_shortfall"]
            ),
        },
        "differences": differences,
        "metadata": {
            "confidence": float(confidence),
            "n_simulations": int(n_simulations),
            "n_observations": int(returns.shape[0]),
            "n_assets": int(returns.shape[1]),
            "risk_horizon": "one observation period",
        },
    }
