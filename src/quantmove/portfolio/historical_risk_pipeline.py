"""Integrated historical portfolio risk analysis pipeline."""

import math
import numbers

from quantmove.data.data_validation import (
    validate_prices,
    calculate_returns,
)
from quantmove.risk.return_statistics import calculate_return_statistics
from quantmove.risk.correlation import calculate_correlation
from quantmove.risk.historical_var import calculate_historical_var_es
from quantmove.portfolio.portfolio_risk import (
    calculate_portfolio_volatility,
)
from quantmove.portfolio.risk_comparison import compare_risk_estimators
from quantmove.risk.var_backtesting import backtest_historical_var
from quantmove.risk.var_backtesting_report import var_backtesting_report


def run_historical_risk_pipeline(
    prices,
    weights,
    confidence=0.95,
    window_size=250,
    significance_level=0.05,
    n_simulations=100_000,
    random_state=None,
):
    """Run an integrated portfolio risk analysis from historical prices.

    Returns decimal return fractions for risk estimates, not currency
    amounts. Backtesting is skipped when the return history is too
    short for the requested estimation window.
    """
    # Validate configuration before running any risk calculations.
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, numbers.Real)
        or not math.isfinite(confidence)
        or not 0 < confidence < 1
    ):
        raise ValueError("confidence must be finite and strictly between 0 and 1.")

    if (
        isinstance(significance_level, bool)
        or not isinstance(significance_level, numbers.Real)
        or not math.isfinite(significance_level)
        or not 0 < significance_level < 1
    ):
        raise ValueError(
            "significance_level must be finite and strictly between 0 and 1."
        )

    if (
        isinstance(window_size, bool)
        or not isinstance(window_size, numbers.Integral)
        or window_size < 2
    ):
        raise ValueError("window_size must be an integer greater than or equal to 2.")

    if (
        isinstance(n_simulations, bool)
        or not isinstance(n_simulations, numbers.Integral)
        or n_simulations < 1
    ):
        raise ValueError("n_simulations must be a positive integer.")

    if random_state is not None and (
        isinstance(random_state, bool)
        or not isinstance(random_state, numbers.Integral)
        or random_state < 0
    ):
        raise ValueError("random_state must be None or a non-negative integer.")

    validated_prices = validate_prices(prices)
    returns = calculate_returns(validated_prices)

    if len(returns) < 2:
        raise ValueError(
            "At least three price observations are required "
            "to calculate returns and sample statistics."
        )

    statistics = calculate_return_statistics(returns)
    correlation = calculate_correlation(returns)

    portfolio_volatility = calculate_portfolio_volatility(
        returns,
        weights,
    )

    historical_risk = calculate_historical_var_es(
        returns,
        weights,
        confidence=confidence,
    )

    estimator_comparison = compare_risk_estimators(
        returns,
        weights,
        confidence=confidence,
        n_simulations=n_simulations,
        random_state=random_state,
    )

    backtest = None
    backtest_report = None

    if len(returns) > window_size:
        backtest = backtest_historical_var(
            returns=returns,
            weights=weights,
            window_size=window_size,
            confidence=confidence,
            significance_level=significance_level,
        )

        backtest_report = var_backtesting_report(
            backtest["observations"]["exception"].to_numpy(),
            confidence=confidence,
            significance_level=significance_level,
        )

    return {
        "metadata": {
            "n_price_observations": len(validated_prices),
            "n_return_observations": len(returns),
            "n_assets": len(returns.columns),
            "confidence": float(confidence),
            "window_size": int(window_size),
            "backtest_status": (
                "completed"
                if backtest is not None
                else "skipped_insufficient_history"
            ),
            "risk_units": "decimal return fractions",
        },
        "prices": validated_prices,
        "returns": returns,
        "return_statistics": statistics,
        "correlation": correlation,
        "portfolio_volatility": portfolio_volatility,
        "historical_var_es": historical_risk,
        "estimator_comparison": estimator_comparison,
        "backtest": backtest,
        "backtest_report": backtest_report,
    }
