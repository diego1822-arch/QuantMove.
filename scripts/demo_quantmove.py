
"""Reproducible demonstration of QuantMove portfolio risk analysis."""

import numpy as np
import pandas as pd

from quantmove.portfolio.historical_risk_pipeline import (
    run_historical_risk_pipeline,
)


def generate_synthetic_prices(n_prices: int = 300) -> pd.DataFrame:
    """Generate deterministic synthetic prices for two assets."""
    rng = np.random.default_rng(123)

    simulated_returns = rng.normal(
        loc=[0.0003, 0.0002],
        scale=[0.01, 0.015],
        size=(n_prices - 1, 2),
    )

    initial_prices = np.array([100.0, 200.0])

    price_values = np.vstack(
        [
            initial_prices,
            initial_prices * np.cumprod(
                1 + simulated_returns, axis=0
            ),
        ]
    )

    dates = pd.bdate_range("2025-01-02", periods=n_prices)

    return pd.DataFrame(
        price_values,
        index=dates,
        columns=["ASSET_A", "ASSET_B"],
    )


def print_percentage(label: str, value: float) -> None:
    """Display a decimal return fraction as a percentage."""
    print(f"{label:<34} {value * 100:>9.3f}%")


def main() -> None:
    """Run the integrated risk analysis and display the results."""
    prices = generate_synthetic_prices()

    result = run_historical_risk_pipeline(
        prices=prices,
        weights={"ASSET_A": 0.60, "ASSET_B": 0.40},
        confidence=0.95,
        window_size=60,
        significance_level=0.05,
        n_simulations=10_000,
        random_state=42,
    )

    metadata = result["metadata"]
    historical = result["estimator_comparison"]["historical"]
    monte_carlo = result["estimator_comparison"]["monte_carlo"]
    report = result["backtest_report"]

    print("=" * 62)
    print("QUANTMOVE | PORTFOLIO RISK DEMO")
    print("=" * 62)
    print("Data source: synthetic and reproducible")
    print(f"Price observations: {metadata['n_price_observations']}")
    print(f"Return observations: {metadata['n_return_observations']}")
    print(f"Assets: {metadata['n_assets']}")
    print(f"Confidence level: {metadata['confidence']:.0%}")
    print(f"Backtesting status: {metadata['backtest_status']}")
    print("-" * 62)

    print("PORTFOLIO RISK")
    print_percentage(
        "Portfolio volatility:",
        result["portfolio_volatility"],
    )
    print_percentage("Historical VaR:", historical["var"])
    print_percentage(
        "Historical Expected Shortfall:",
        historical["expected_shortfall"],
    )
    print("-" * 62)

    print("ESTIMATOR COMPARISON")
    print_percentage("Monte Carlo VaR:", monte_carlo["var"])
    print_percentage(
        "Monte Carlo Expected Shortfall:",
        monte_carlo["expected_shortfall"],
    )
    print("-" * 62)

    print("BACKTESTING DIAGNOSTICS")

    if report is None:
        print("Backtesting report unavailable.")
    else:
        diagnostics = report["diagnostics"]

        print(f"Test observations: {report['n_observations']}")
        print(f"VaR exceptions: {report['n_exceptions']}")
        print(
            "Expected exception rate: "
            f"{report['expected_exception_rate']:.2%}"
        )
        print(
            "Observed exception rate: "
            f"{report['observed_exception_rate']:.2%}"
        )
        print(
            "Coverage status: "
            f"{diagnostics['coverage_status']}"
        )
        print(
            "Independence status: "
            f"{diagnostics['independence_status']}"
        )
        print(
            "Conditional coverage status: "
            f"{diagnostics['conditional_coverage_status']}"
        )

        print("\nInterpretation warning:")
        print(diagnostics["interpretation_warning"])

    print("-" * 62)
    print("IMPORTANT")
    print("All data are synthetic; results are illustrative.")
    print("Risk estimates do not guarantee future outcomes.")
    print("VaR and ES are losses per observation period, not cash amounts.")
    print("=" * 62)


if __name__ == "__main__":
    main()
