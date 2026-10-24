import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, "/content/QuantMove/src")

from quantmove.risk.return_statistics import (
    calculate_return_statistics,
)


def make_returns(values):
    return pd.DataFrame(
        values,
        index=pd.date_range(
            "2026-01-05",
            periods=len(next(iter(values.values()))),
            freq="D",
        ),
    )


def test_calculates_statistics_correctly():
    returns = make_returns({
        "ASSET_A": [0.01, 0.03, 0.05],
        "ASSET_B": [-0.02, 0.00, 0.02],
    })

    result = calculate_return_statistics(returns)

    assert result.loc["ASSET_A", "mean"] == pytest.approx(0.03)
    assert result.loc["ASSET_A", "volatility"] == pytest.approx(0.02)
    assert result.loc["ASSET_A", "minimum"] == pytest.approx(0.01)
    assert result.loc["ASSET_A", "maximum"] == pytest.approx(0.05)

    assert result.loc["ASSET_B", "mean"] == pytest.approx(0.00)
    assert result.loc["ASSET_B", "minimum"] == pytest.approx(-0.02)
    assert result.loc["ASSET_B", "maximum"] == pytest.approx(0.02)


def test_rejects_non_dataframe():
    with pytest.raises(TypeError):
        calculate_return_statistics([0.01, 0.02])


def test_rejects_empty_dataframe():
    with pytest.raises(ValueError, match="empty"):
        calculate_return_statistics(pd.DataFrame())


def test_rejects_missing_returns():
    returns = make_returns({"ASSET_A": [0.01, np.nan]})

    with pytest.raises(ValueError, match="missing"):
        calculate_return_statistics(returns)


def test_rejects_infinite_returns():
    returns = make_returns({"ASSET_A": [0.01, np.inf]})

    with pytest.raises(ValueError, match="finite"):
        calculate_return_statistics(returns)


def test_rejects_nonnumeric_returns():
    returns = make_returns({"ASSET_A": [0.01, "abc"]})

    with pytest.raises(ValueError, match="numeric"):
        calculate_return_statistics(returns)


def test_rejects_duplicate_asset_names():
    returns = pd.DataFrame(
        [[0.01, 0.02], [0.03, 0.04]],
        columns=["ASSET_A", "ASSET_A"],
    )

    with pytest.raises(ValueError, match="Asset names"):
        calculate_return_statistics(returns)


def test_requires_two_observations():
    returns = make_returns({"ASSET_A": [0.01]})

    with pytest.raises(ValueError, match="At least two"):
        calculate_return_statistics(returns)
