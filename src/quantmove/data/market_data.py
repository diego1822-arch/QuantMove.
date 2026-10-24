"""Market data download utilities using Yahoo Finance."""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from quantmove.data.data_validation import validate_prices


def download_adjusted_close(
    tickers: list[str],
    period: str = "1y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Download and validate adjusted closing prices."""

    # 1. Validate tickers.
    if not isinstance(tickers, list) or not tickers:
        raise ValueError("tickers must be a non-empty list.")

    if any(
        not isinstance(ticker, str) or not ticker.strip()
        for ticker in tickers
    ):
        raise ValueError("Every ticker must be a non-empty string.")

    normalized_tickers = [
        ticker.strip().upper() for ticker in tickers
    ]

    if len(set(normalized_tickers)) != len(normalized_tickers):
        raise ValueError("Ticker symbols must be unique.")

    # 2. Validate download parameters.
    if not isinstance(period, str) or not period.strip():
        raise ValueError("period must be a non-empty string.")

    if not isinstance(interval, str) or not interval.strip():
        raise ValueError("interval must be a non-empty string.")

    # 3. Download data.
    try:
        raw = yf.download(
            tickers=normalized_tickers,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as error:
        raise RuntimeError(
            "Yahoo Finance price download failed."
        ) from error

    if raw is None or raw.empty:
        raise ValueError("Yahoo Finance returned no price data.")

    # 4. Extract closing prices, including MultiIndex layouts.
    if isinstance(raw.columns, pd.MultiIndex):
        close_level = None

        for level in range(raw.columns.nlevels):
            if "Close" in raw.columns.get_level_values(level):
                close_level = level
                break

        if close_level is None:
            raise ValueError(
                "Downloaded data has no Close prices."
            )

        prices = raw.xs(
            "Close",
            axis=1,
            level=close_level,
        ).copy()

    else:
        if len(normalized_tickers) != 1 or "Close" not in raw.columns:
            raise ValueError(
                "Unexpected Yahoo Finance column structure."
            )

        prices = raw[["Close"]].rename(
            columns={"Close": normalized_tickers[0]}
        )

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=normalized_tickers[0])

    # 5. Check that all requested assets are present.
    missing_tickers = [
        ticker
        for ticker in normalized_tickers
        if ticker not in prices.columns
    ]

    if missing_tickers:
        raise ValueError(
            f"Missing downloaded tickers: {missing_tickers}"
        )

    prices = prices.loc[:, normalized_tickers].copy()
    prices.columns = normalized_tickers

    # 6. Validate and order dates.
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError(
            "Downloaded prices must have a DatetimeIndex."
        )

    prices = prices.sort_index()

    # 7. Remove only trailing rows where every asset is missing.
    all_missing = prices.isna().all(axis=1)

    if all_missing.any():
        first_missing_position = all_missing.to_numpy().argmax()

        trailing_rows = prices.iloc[first_missing_position:]

        if not trailing_rows.isna().all(axis=1).all():
            raise ValueError(
                "Downloaded prices contain fully missing rows "
                "inside the historical data."
            )

        prices = prices.iloc[:first_missing_position].copy()

    # 8. Reject any remaining missing prices.
    if prices.isna().any().any():
        raise ValueError(
            "Downloaded prices contain partially missing values. "
            "Inspect the source data before continuing."
        )

    # 9. Apply central project validation.
    return validate_prices(prices)
