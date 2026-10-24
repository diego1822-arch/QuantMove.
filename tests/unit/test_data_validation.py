
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, "/content/QuantMove/src")

from quantmove.data.data_validation import (
    validate_prices,
    calculate_returns,
)


DATES = pd.to_datetime([
    "2026-01-05",
    "2026-01-06",
    "2026-01-07",
])


def make_prices(values, index=None, columns=None):
    if index is None:
        if isinstance(values, dict):
            n_rows = len(next(iter(values.values())))
        else:
            n_rows = len(values)

        index = pd.date_range(
            start="2026-01-05",
            periods=n_rows,
            freq="D",
        )

    return pd.DataFrame(
        values,
        index=index,
        columns=columns,
    )


def test_validate_prices_returns_float_copy():
    prices = make_prices({"ASSET_A": [100, 105]})

    result = validate_prices(prices)

    assert result["ASSET_A"].dtype == float
    assert result is not prices


def test_rejects_empty_dataframe():
    with pytest.raises(ValueError):
        validate_prices(pd.DataFrame())


def test_rejects_non_dataframe():
    with pytest.raises(TypeError):
        validate_prices([100, 105])


def test_rejects_non_datetime_index():
    prices = pd.DataFrame({"ASSET_A": [100, 105]})

    with pytest.raises(TypeError, match="DatetimeIndex"):
        validate_prices(prices)


def test_rejects_duplicate_asset_names():
    prices = make_prices(
        [[100, 200]],
        index=DATES[:1],
        columns=["ASSET_A", "ASSET_A"],
    )

    with pytest.raises(ValueError, match="Asset names"):
        validate_prices(prices)


def test_rejects_duplicate_timestamps():
    dates = pd.to_datetime(["2026-01-05", "2026-01-05"])
    prices = make_prices({"ASSET_A": [100, 105]}, index=dates)

    with pytest.raises(ValueError, match="unique"):
        validate_prices(prices)


def test_rejects_unsorted_timestamps():
    dates = pd.to_datetime(["2026-01-06", "2026-01-05"])
    prices = make_prices({"ASSET_A": [105, 100]}, index=dates)

    with pytest.raises(ValueError, match="chronologically"):
        validate_prices(prices)


def test_rejects_missing_values():
    prices = make_prices({"ASSET_A": [100, np.nan]})

    with pytest.raises(ValueError):
        validate_prices(prices)


def test_rejects_infinite_values():
    prices = make_prices({"ASSET_A": [100, np.inf]})

    with pytest.raises(ValueError):
        validate_prices(prices)


@pytest.mark.parametrize("invalid_price", [0, -10])
def test_rejects_nonpositive_prices(invalid_price):
    prices = make_prices({"ASSET_A": [100, invalid_price]})

    with pytest.raises(ValueError):
        validate_prices(prices)


def test_calculates_returns_correctly():
    prices = make_prices({
        "ASSET_A": [100, 105, 102],
        "ASSET_B": [200, 202, 210],
    })

    result = calculate_returns(prices)

    expected = pd.DataFrame({
        "ASSET_A": [0.05, 102 / 105 - 1],
        "ASSET_B": [0.01, 210 / 202 - 1],
    }, index=prices.index[1:])

    pd.testing.assert_frame_equal(result, expected)


def test_returns_have_no_missing_values():
    prices = make_prices({"ASSET_A": [100, 105, 110]})

    result = calculate_returns(prices)

    assert not result.isna().any().any()



def test_rejects_nonnumeric_prices():
    prices = make_prices({"ASSET_A": [100, "abc"]})

    with pytest.raises(ValueError, match="numeric values"):
        validate_prices(prices)


def test_rejects_returns_with_only_one_observation():
    prices = make_prices({"ASSET_A": [100]})

    with pytest.raises(ValueError, match="At least two"):
        calculate_returns(prices)
