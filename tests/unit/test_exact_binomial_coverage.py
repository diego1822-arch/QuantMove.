import numpy as np
import pytest
from scipy.stats import binomtest

from quantmove.risk.var_backtesting import exact_binomial_coverage_test


def test_exact_binomial_matches_scipy_reference():
    exceptions = [1] * 5 + [0] * 95

    result = exact_binomial_coverage_test(
        exceptions,
        confidence=0.95,
    )
    expected = binomtest(
        k=5,
        n=100,
        p=0.05,
        alternative="two-sided",
    )

    assert result["n_observations"] == 100
    assert result["n_exceptions"] == 5
    assert result["expected_exception_rate"] == pytest.approx(0.05)
    assert result["observed_exception_rate"] == pytest.approx(0.05)
    assert result["p_value"] == pytest.approx(expected.pvalue)
    assert result["reject_null"] is False
    assert result["p_value_method"] == "exact_binomial_two_sided"


def test_exact_binomial_zero_exceptions_is_finite():
    result = exact_binomial_coverage_test([0] * 100)

    assert result["n_exceptions"] == 0
    assert np.isfinite(result["p_value"])
    assert 0 <= result["p_value"] <= 1


def test_exact_binomial_all_exceptions_is_finite():
    result = exact_binomial_coverage_test([1] * 20)

    assert result["n_exceptions"] == 20
    assert np.isfinite(result["p_value"])
    assert 0 <= result["p_value"] <= 1
    assert result["reject_null"] is True


@pytest.mark.parametrize(
    "exceptions",
    [[], [[0, 1]], [0, 2], [0, np.nan], ["bad", 0]],
)
def test_exact_binomial_rejects_invalid_exceptions(exceptions):
    with pytest.raises(ValueError):
        exact_binomial_coverage_test(exceptions)


@pytest.mark.parametrize("confidence", [0, 1, -0.1, 1.1, True, np.nan])
def test_exact_binomial_rejects_invalid_confidence(confidence):
    with pytest.raises(ValueError):
        exact_binomial_coverage_test([0, 1, 0, 1], confidence=confidence)


@pytest.mark.parametrize(
    "significance_level",
    [0, 1, -0.1, 1.1, True, np.nan],
)
def test_exact_binomial_rejects_invalid_significance(significance_level):
    with pytest.raises(ValueError):
        exact_binomial_coverage_test(
            [0, 1, 0, 1],
            significance_level=significance_level,
        )
