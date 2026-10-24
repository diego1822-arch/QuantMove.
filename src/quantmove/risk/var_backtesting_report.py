"""Unified reporting for VaR exception backtesting."""

from quantmove.risk.var_backtesting import (
    kupiec_test,
    exact_binomial_coverage_test,
    christoffersen_independence_test,
    christoffersen_conditional_coverage_test,
)


def var_backtesting_report(
    exceptions,
    confidence=0.95,
    significance_level=0.05,
):
    """Run VaR backtesting tests and return a unified report.

    The report separates unconditional coverage, temporal
    independence, and conditional coverage conclusions.

    Non-rejection is not proof that a model is correctly specified.
    Asymptotic tests can be unreliable with small samples or few
    exceptions. The exact binomial test assumes independent Bernoulli
    observations with a constant exception probability.

    Parameters
    ----------
    exceptions:
        One-dimensional sequence of binary VaR exceptions.
    confidence:
        VaR confidence level, strictly between 0 and 1.
    significance_level:
        Significance threshold, strictly between 0 and 1.

    Returns
    -------
    dict
        Input-independent metadata, individual test results,
        and a concise diagnostic summary.
    """
    kupiec = kupiec_test(
        exceptions,
        confidence=confidence,
        significance_level=significance_level,
    )
    exact = exact_binomial_coverage_test(
        exceptions,
        confidence=confidence,
        significance_level=significance_level,
    )
    independence = christoffersen_independence_test(
        exceptions,
        significance_level=significance_level,
    )
    conditional = christoffersen_conditional_coverage_test(
        exceptions,
        confidence=confidence,
        significance_level=significance_level,
    )

    coverage_disagreement = (
        kupiec["reject_null"] != exact["reject_null"]
    )

    if kupiec["reject_null"] and exact["reject_null"]:
        coverage_status = "coverage_rejected_by_both_tests"
    elif coverage_disagreement:
        coverage_status = "coverage_tests_disagree"
    else:
        coverage_status = "coverage_not_rejected_by_either_test"

    if not independence["independence_identifiable"]:
        independence_status = "independence_not_identifiable"
    elif independence["reject_null"]:
        independence_status = "independence_rejected"
    else:
        independence_status = "independence_not_rejected"

    if conditional["reject_null"]:
        conditional_status = "conditional_coverage_rejected"
    else:
        conditional_status = "conditional_coverage_not_rejected"

    return {
        "n_observations": kupiec["n_observations"],
        "n_exceptions": kupiec["n_exceptions"],
        "confidence": float(confidence),
        "significance_level": float(significance_level),
        "expected_exception_rate": (
            kupiec["expected_exception_rate"]
        ),
        "observed_exception_rate": (
            kupiec["observed_exception_rate"]
        ),
        "tests": {
            "kupiec": kupiec,
            "exact_binomial": exact,
            "christoffersen_independence": independence,
            "christoffersen_conditional_coverage": conditional,
        },
        "diagnostics": {
            "coverage_status": coverage_status,
            "coverage_tests_disagree": bool(
                coverage_disagreement
            ),
            "independence_status": independence_status,
            "conditional_coverage_status": conditional_status,
            "interpretation_warning": (
                "Failure to reject a null hypothesis does not prove "
                "that the model is correct. Interpret p-values in "
                "light of sample size and test assumptions."
            ),
        },
    }
