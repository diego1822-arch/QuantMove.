import numpy as np
import pandas as pd
import pytest

from quantmove.data.market_data import download_adjusted_close


@pytest.fixture
def market_data():
    dates = pd.to_datetime([
        "2026-01-05",
        "2026-01-06",
        "2026-01-07",
    ])
    columns = pd.MultiIndex.from_product(
        [["Close", "High"], ["AAPL", "MSFT"]],
        names=["Price", "Ticker"],
    )
    values = [
        [100.0, 200.0, 110.0, 210.0],
        [101.0, 201.0, 111.0, 211.0],
        [102.0, 202.0, 112.0, 212.0],
    ]
    return pd.DataFrame(values, index=dates, columns=columns)


def test_download_adjusted_close_returns_close_prices(
    market_data, monkeypatch
):
    def fake_download(**kwargs):
        assert kwargs["auto_adjust"] is True
        return market_data

    monkeypatch.setattr(
        "quantmove.data.market_data.yf.download",
        fake_download,
    )

    prices = download_adjusted_close(["AAPL", "MSFT"])

    assert prices.shape == (3, 2)
    assert prices.columns.tolist() == ["AAPL", "MSFT"]
    assert prices.iloc[0].tolist() == [100.0, 200.0]
    assert prices.index.is_monotonic_increasing
    assert not prices.index.has_duplicates
    assert np.isfinite(prices.to_numpy()).all()


@pytest.mark.parametrize(
    "tickers",
    [
        [],
        ["AAPL", "AAPL"],
        ["AAPL", " aapl "],
        [""],
        ["   "],
    ],
)
def test_invalid_tickers_are_rejected(tickers):
    with pytest.raises(ValueError):
        download_adjusted_close(tickers)


def test_empty_download_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "quantmove.data.market_data.yf.download",
        lambda **kwargs: pd.DataFrame(),
    )

    with pytest.raises(ValueError, match="no price data"):
        download_adjusted_close(["AAPL"])


def test_missing_prices_are_rejected(monkeypatch):
    dates = pd.to_datetime(["2026-01-05", "2026-01-06"])
    columns = pd.MultiIndex.from_product(
        [["Close"], ["AAPL", "MSFT"]],
        names=["Price", "Ticker"],
    )
    data = pd.DataFrame(
        [[100.0, 200.0], [101.0, np.nan]],
        index=dates,
        columns=columns,
    )

    monkeypatch.setattr(
        "quantmove.data.market_data.yf.download",
        lambda **kwargs: data,
    )

    with pytest.raises(ValueError, match="missing values"):
        download_adjusted_close(["AAPL", "MSFT"])


def test_download_exception_is_wrapped(monkeypatch):
    def fail_download(**kwargs):
        raise OSError("network unavailable")

    monkeypatch.setattr(
        "quantmove.data.market_data.yf.download",
        fail_download,
    )

    with pytest.raises(RuntimeError, match="download failed"):
        download_adjusted_close(["AAPL"])


def test_single_ticker_download(monkeypatch):
    dates = pd.to_datetime(["2026-01-05", "2026-01-06"])
    data = pd.DataFrame(
        {"Close": [100.0, 101.0]},
        index=dates,
    )

    monkeypatch.setattr(
        "quantmove.data.market_data.yf.download",
        lambda **kwargs: data,
    )

    prices = download_adjusted_close(["AAPL"])

    assert prices.columns.tolist() == ["AAPL"]
    assert prices.shape == (2, 1)
