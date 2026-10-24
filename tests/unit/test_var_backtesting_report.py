import numpy as np
import pytest

from quantmove.risk.var_backtesting_report import (
    var_backtesting_report,
)


def test_report_contains_all_four_tests():
    exceptions = [0] * 95 + [1] * 5

    report = var_backtesting_report(exceptions)

    assert report["n_observations"] == 100
    assert report["n_exceptions"] == 5
    assert report["confidence"] == 0.95
    assert set(report["tests"]) == {
        "kupiec",
        "exact_binomial",
        "christoffersen_independence",
        "christoffersen_conditional_coverage",
    }


def test_report_rates_are_consistent():
    report = var_backtesting_report([0, 0, 1, 0])

    assert report["expected_exception_rate"] == pytest.approx(0.05)
    assert report["observed_exception_rate"] == pytest.approx(0.25)


def test_report_flags_disagreement_between_coverage_tests():
    report = var_backtesting_report([1] + [0] * 99)

    assert report["diagnostics"]["coverage_tests_disagree"] is True
    assert (
        report["diagnostics"]["coverage_status"]
        == "coverage_tests_disagree"
    )


def test_report_marks_independence_unidentifiable_for_no_exceptions():
    report = var_backtesting_report([0] * 100)

    assert (
        report["diagnostics"]["independence_status"]
        == "independence_not_identifiable"
    )


def test_report_accepts_boolean_exceptions():
    report = var_backtesting_report([False, True, False, False])

    assert report["n_observations"] == 4
    assert report["n_exceptions"] == 1


@pytest.mark.parametrize(
    "exceptions",
    [
        [],
        [[0, 1]],
        [0, 2],
        [0, np.nan],
        ["bad", 0],
    ],
)
def test_report_rejects_invalid_exceptions(exceptions):
    with pytest.raises(ValueError):
        var_backtesting_report(exceptions)


@pytest.mark.parametrize("confidence", [0, 1, -0.1, 1.1, True])
def test_report_rejects_invalid_confidence(confidence):
    with pytest.raises(ValueError):
        var_backtesting_report([0, 1, 0], confidence=confidence)


@pytest.mark.parametrize(
    "significance_level", [0, 1, -0.1, 1.1, True]
)
def test_report_rejects_invalid_significance(significance_level):
    with pytest.raises(ValueError):
        var_backtesting_report(
            [0, 1, 0],
            significance_level=significance_level,
        )
