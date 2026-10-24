
import numpy as np
import pytest

from quantmove.risk.var_backtesting import (
    christoffersen_conditional_coverage_test,
)


def test_conditional_coverage_returns_expected_structure():
    exceptions = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0] * 10

    result = christoffersen_conditional_coverage_test(
        exceptions,
        confidence=0.80,
    )

    assert set(result) == {
        "n_observations",
        "n_exceptions",
        "expected_exception_rate",
        "observed_exception_rate",
        "kupiec_lr_statistic",
        "independence_lr_statistic",
        "lr_statistic",
        "degrees_of_freedom",
        "p_value",
        "significance_level",
        "reject_null",
        "kupiec_p_value",
        "independence_p_value",
        "kupiec_reject_null",
        "independence_reject_null",
        "independence_identifiable",
    }


def test_conditional_coverage_lr_is_sum_of_components():
    exceptions = [0, 0, 1, 0, 0, 1, 0, 0, 0, 1] * 10

    result = christoffersen_conditional_coverage_test(
        exceptions,
        confidence=0.80,
    )

    assert result["lr_statistic"] == pytest.approx(
        result["kupiec_lr_statistic"]
        + result["independence_lr_statistic"]
    )
    assert result["degrees_of_freedom"] == 2
    assert 0 <= result["p_value"] <= 1


def test_conditional_coverage_uses_two_degree_chi_square_p_value():
    exceptions = [0, 0, 0, 1, 0, 0, 1, 0, 0, 0] * 10

    result = christoffersen_conditional_coverage_test(
        exceptions,
        confidence=0.80,
    )

    # Survival function of chi-square(2): exp(-LR / 2).
    expected_p_value = np.exp(-result["lr_statistic"] / 2)

    assert result["p_value"] == pytest.approx(expected_p_value)


def test_conditional_coverage_rejects_invalid_confidence():
    with pytest.raises(ValueError):
        christoffersen_conditional_coverage_test(
            [0, 1, 0, 1],
            confidence=1.0,
        )


def test_conditional_coverage_requires_two_observations():
    with pytest.raises(ValueError):
        christoffersen_conditional_coverage_test([1])


def test_conditional_coverage_rejects_invalid_exceptions():
    with pytest.raises((TypeError, ValueError)):
        christoffersen_conditional_coverage_test([0, 2, 1])


def test_conditional_coverage_validates_significance_level():
    with pytest.raises(ValueError):
        christoffersen_conditional_coverage_test(
            [0, 1, 0, 1],
            significance_level=0,
        )
