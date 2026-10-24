"""Historical Value-at-Risk backtesting and Kupiec coverage test."""

from math import erfc, sqrt

import numpy as np
import pandas as pd

from quantmove.risk.historical_var import calculate_historical_var_es


def _validate_confidence(confidence):
    """Validate a VaR confidence level."""
    if (
        isinstance(confidence, (bool, np.bool_))
        or not isinstance(
            confidence, (int, float, np.integer, np.floating)
        )
        or not np.isfinite(confidence)
        or not 0 < confidence < 1
    ):
        raise ValueError("confidence must be finite and between 0 and 1.")


def _normalize_weights(weights, columns):
    """Normalize portfolio weights to the return-column order."""
    if isinstance(weights, pd.Series):
        if not weights.index.is_unique:
            raise ValueError("Weight asset names must be unique.")
        if set(weights.index) != set(columns):
            raise ValueError("Weights must match the return asset names.")
        result = weights.reindex(columns).astype(float)

    elif isinstance(weights, dict):
        if len(weights) != len(set(weights)):
            raise ValueError("Weight asset names must be unique.")
        if set(weights) != set(columns):
            raise ValueError("Weights must match the return asset names.")
        result = pd.Series(weights, dtype=float).reindex(columns)

    else:
        try:
            values = np.asarray(weights, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("weights must be numeric.") from exc

        if values.ndim != 1 or len(values) != len(columns):
            raise ValueError("weights must contain one value per asset.")

        result = pd.Series(values, index=columns, dtype=float)

    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("weights must contain only finite values.")

    if not np.isclose(result.sum(), 1.0):
        raise ValueError("Portfolio weights must sum to 1.")

    return result


def kupiec_test(exceptions, confidence=0.95, significance_level=0.05):
    """Run the Kupiec unconditional-coverage likelihood-ratio test.

    Parameters
    ----------
    exceptions:
        One-dimensional sequence of 0/1 values or booleans. A value of
        1/True means the realized loss exceeded the VaR forecast.
    confidence:
        VaR confidence level. The expected exception rate is 1-confidence.
    significance_level:
        Significance threshold used for the rejection decision.

    Returns
    -------
    dict
        Exception counts, empirical and expected rates, LR statistic,
        asymptotic chi-square(1) p-value, and rejection decision.

    Notes
    -----
    The null hypothesis is that the exception probability equals
    1-confidence. This test evaluates unconditional coverage, not
    independence or conditional coverage.
    """
    _validate_confidence(confidence)

    if (
        isinstance(significance_level, (bool, np.bool_))
        or not isinstance(
            significance_level,
            (int, float, np.integer, np.floating),
        )
        or not np.isfinite(significance_level)
        or not 0 < significance_level < 1
    ):
        raise ValueError("significance_level must be between 0 and 1.")

    try:
        values = np.asarray(exceptions)
    except (TypeError, ValueError) as exc:
        raise ValueError("exceptions must be a one-dimensional 0/1 sequence.") from exc

    if values.ndim != 1 or values.size == 0:
        raise ValueError("exceptions must be a non-empty one-dimensional sequence.")

    if values.dtype.kind not in "biuf":
        raise ValueError("exceptions must contain only 0/1 values.")

    if not np.isfinite(values.astype(float)).all():
        raise ValueError("exceptions must contain finite values.")

    if not np.isin(values, [0, 1, False, True]).all():
        raise ValueError("exceptions must contain only 0/1 values.")

    values = values.astype(int)
    n = int(values.size)
    x = int(values.sum())
    expected_rate = 1.0 - float(confidence)
    observed_rate = x / n

    # Bernoulli log-likelihood, with 0*log(0) defined as zero.
    def log_likelihood(successes, trials, probability):
        failures = trials - successes
        result = 0.0

        if successes:
            result += successes * np.log(probability)

        if failures:
            result += failures * np.log1p(-probability)

        return float(result)

    null_log_likelihood = log_likelihood(x, n, expected_rate)

    if x == 0 or x == n:
        alternative_log_likelihood = 0.0
    else:
        alternative_log_likelihood = log_likelihood(
            x, n, observed_rate
        )

    lr_statistic = max(
        0.0,
        -2.0 * (null_log_likelihood - alternative_log_likelihood),
    )

    # Survival function of chi-square with one degree of freedom.
    p_value = float(erfc(sqrt(lr_statistic / 2.0)))

    return {
        "n_observations": n,
        "n_exceptions": x,
        "expected_exception_rate": expected_rate,
        "observed_exception_rate": float(observed_rate),
        "likelihood_ratio": float(lr_statistic),
        "p_value": p_value,
        "significance_level": float(significance_level),
        "reject_null": bool(p_value < significance_level),
        "p_value_method": "asymptotic_chi_square_1_df",
        "asymptotic_p_value_caution": (
            "La aproximación asintótica puede ser poco fiable "
            "en muestras pequeñas o con pocos eventos."
        ),
    }


def backtest_historical_var(
    returns,
    weights,
    window_size=250,
    confidence=0.95,
    significance_level=0.05,
):
    """Backtest historical VaR using a strictly trailing estimation window.

    For each forecast date t, the VaR is estimated from observations
    strictly before t. The realized portfolio loss at t is then compared
    with that forecast.

    Returns
    -------
    dict
        'observations': DataFrame with VaR, realized loss, and exception.
        'summary': aggregate coverage metrics and Kupiec test results.

    VaR and realized loss are decimal return fractions, not currency
    amounts. A strict exceedance (loss > VaR) counts as an exception.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")

    if returns.empty or returns.shape[1] == 0:
        raise ValueError("returns must contain observations and assets.")

    if not returns.columns.is_unique:
        raise ValueError("returns must have unique asset columns.")

    if (
        isinstance(window_size, (bool, np.bool_))
        or not isinstance(window_size, (int, np.integer))
        or window_size < 2
    ):
        raise ValueError("window_size must be an integer of at least 2.")

    _validate_confidence(confidence)

    if (
        isinstance(significance_level, (bool, np.bool_))
        or not isinstance(
            significance_level,
            (int, float, np.integer, np.floating),
        )
        or not np.isfinite(significance_level)
        or not 0 < significance_level < 1
    ):
        raise ValueError("significance_level must be between 0 and 1.")

    try:
        numeric_returns = returns.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("returns must contain numeric values.") from exc

    if not np.isfinite(numeric_returns.to_numpy()).all():
        raise ValueError("returns must contain only finite values.")

    if len(numeric_returns) <= window_size:
        raise ValueError(
            "returns must contain more observations than window_size."
        )

    portfolio_weights = _normalize_weights(
        weights, numeric_returns.columns
    )

    forecast_var = []
    realized_losses = []
    exceptions = []
    forecast_index = numeric_returns.index[window_size:]

    for t in range(window_size, len(numeric_returns)):
        # The current observation is deliberately excluded from estimation.
        estimation_window = numeric_returns.iloc[t - window_size:t]

        historical_estimate = calculate_historical_var_es(
            estimation_window,
            portfolio_weights,
            confidence=confidence,
        )
        var_value = float(historical_estimate["var"])

        realized_return = float(
            numeric_returns.iloc[t].to_numpy()
            @ portfolio_weights.to_numpy()
        )
        realized_loss = -realized_return
        exception = bool(realized_loss > var_value)

        forecast_var.append(var_value)
        realized_losses.append(realized_loss)
        exceptions.append(exception)

    observations = pd.DataFrame(
        {
            "var": forecast_var,
            "realized_loss": realized_losses,
            "exception": exceptions,
        },
        index=forecast_index,
    )
    observations.index.name = returns.index.name

    coverage = kupiec_test(
        observations["exception"].to_numpy(),
        confidence=confidence,
        significance_level=significance_level,
    )

    summary = {
        "method": "historical_var",
        "confidence": float(confidence),
        "window_size": int(window_size),
        "n_backtest_observations": len(observations),
        **coverage,
    }

    return {
        "observations": observations,
        "summary": summary,
    }



def christoffersen_independence_test(
    exceptions,
    significance_level=0.05,
):
    """
    Test de independencia de Christoffersen para excepciones de VaR.

    H0: las excepciones son independientes en el tiempo.
    H1: la probabilidad de excepción depende del estado anterior.

    independence_identifiable indica si la secuencia contiene ambos
    estados (0 y 1), condición necesaria para comparar transiciones
    desde ambos estados.
    """
    from math import erfc, isfinite, log, sqrt

    import numpy as np

    if (
        isinstance(significance_level, (bool, np.bool_))
        or not isinstance(significance_level, (int, float, np.integer, np.floating))
        or not isfinite(float(significance_level))
        or not 0 < significance_level < 1
    ):
        raise ValueError(
            "significance_level debe estar estrictamente entre 0 y 1."
        )

    values = np.asarray(exceptions)

    if values.ndim != 1:
        raise ValueError("exceptions debe ser un vector unidimensional.")

    if values.size < 2:
        raise ValueError(
            "Se necesitan al menos dos observaciones para analizar transiciones."
        )

    if values.dtype.kind not in "biuf":
        raise TypeError("exceptions debe contener únicamente valores numéricos 0/1.")

    if not np.isfinite(values).all():
        raise ValueError("exceptions contiene valores no finitos.")

    if not np.isin(values, [0, 1]).all():
        raise ValueError("exceptions solo puede contener valores 0 y 1.")

    values = values.astype(np.int8)
    independence_identifiable = bool(
        np.any(values == 0) and np.any(values == 1)
    )

    previous = values[:-1]
    current = values[1:]

    n_00 = int(np.sum((previous == 0) & (current == 0)))
    n_01 = int(np.sum((previous == 0) & (current == 1)))
    n_10 = int(np.sum((previous == 1) & (current == 0)))
    n_11 = int(np.sum((previous == 1) & (current == 1)))

    n_transitions = len(values) - 1
    n_0 = n_00 + n_01
    n_1 = n_10 + n_11
    n_switch = n_01 + n_11

    pi = n_switch / n_transitions
    pi_0 = n_01 / n_0 if n_0 else 0.0
    pi_1 = n_11 / n_1 if n_1 else 0.0

    def log_likelihood(successes, failures, probability):
        result = 0.0

        if successes:
            if probability == 0:
                return float("-inf")
            result += successes * log(probability)

        if failures:
            if probability == 1:
                return float("-inf")
            result += failures * log(1 - probability)

        return result

    ll_null = log_likelihood(
        n_switch,
        n_transitions - n_switch,
        pi,
    )

    ll_alternative = (
        log_likelihood(n_01, n_00, pi_0)
        + log_likelihood(n_11, n_10, pi_1)
    )

    if not independence_identifiable:
        lr_statistic = 0.0
        p_value = 1.0
    elif ll_null == float("-inf") and ll_alternative == float("-inf"):
        lr_statistic = 0.0
        p_value = 1.0
    else:
        lr_statistic = max(0.0, 2 * (ll_alternative - ll_null))
        p_value = erfc(sqrt(lr_statistic / 2))

    return {
        "n_observations": int(len(values)),
        "n_transitions": int(n_transitions),
        "n_00": n_00,
        "n_01": n_01,
        "n_10": n_10,
        "n_11": n_11,
        "lr_statistic": float(lr_statistic),
        "p_value": float(p_value),
        "significance_level": float(significance_level),
        "reject_null": bool(p_value < significance_level),
        "independence_identifiable": independence_identifiable,
    }


def christoffersen_conditional_coverage_test(
    exceptions,
    confidence=0.95,
    significance_level=0.05,
):
    """
    Test de cobertura condicional de Christoffersen.

    Combina el test de cobertura incondicional de Kupiec y el test
    de independencia de Christoffersen.

    H0: las excepciones tienen la frecuencia esperada y son
    independientes en el tiempo.

    El estadístico conjunto se compara asintóticamente con una
    distribución chi-cuadrado de dos grados de libertad.
    """
    from math import exp

    import numpy as np

    # Validación común del nivel de confianza.
    _validate_confidence(confidence)

    if (
        isinstance(significance_level, (bool, np.bool_))
        or not isinstance(
            significance_level,
            (int, float, np.integer, np.floating),
        )
        or not np.isfinite(significance_level)
        or not 0 < significance_level < 1
    ):
        raise ValueError(
            "significance_level must be strictly between 0 and 1."
        )

    # Kupiec valida la secuencia y la cobertura incondicional.
    kupiec = kupiec_test(
        exceptions,
        confidence=confidence,
        significance_level=significance_level,
    )

    # Christoffersen exige al menos dos observaciones.
    independence = christoffersen_independence_test(
        exceptions,
        significance_level=significance_level,
    )

    kupiec_lr = float(kupiec["likelihood_ratio"])
    independence_lr = float(independence["lr_statistic"])
    combined_lr = kupiec_lr + independence_lr

    # La supervivencia de chi-cuadrado(2) es exp(-LR / 2).
    p_value = exp(-combined_lr / 2.0)

    return {
        "n_observations": int(kupiec["n_observations"]),
        "n_exceptions": int(kupiec["n_exceptions"]),
        "expected_exception_rate": float(
            kupiec["expected_exception_rate"]
        ),
        "observed_exception_rate": float(
            kupiec["observed_exception_rate"]
        ),
        "kupiec_lr_statistic": kupiec_lr,
        "independence_lr_statistic": independence_lr,
        "lr_statistic": float(combined_lr),
        "degrees_of_freedom": 2,
        "p_value": float(p_value),
        "significance_level": float(significance_level),
        "reject_null": bool(p_value < significance_level),
        "kupiec_p_value": float(kupiec["p_value"]),
        "independence_p_value": float(independence["p_value"]),
        "kupiec_reject_null": bool(kupiec["reject_null"]),
        "independence_reject_null": bool(independence["reject_null"]),
        "independence_identifiable": bool(
            independence["independence_identifiable"]
        ),
    }
def exact_binomial_coverage_test(
    exceptions,
    confidence=0.95,
    significance_level=0.05,
):
    """Run an exact binomial test of unconditional VaR coverage.

    Parameters
    ----------
    exceptions:
        One-dimensional sequence of 0/1 values or booleans. A value
        of 1/True indicates a VaR exception.
    confidence:
        VaR confidence level. The null exception probability is
        1 - confidence.
    significance_level:
        Significance threshold for the rejection decision.

    Returns
    -------
    dict
        Exception counts, expected and observed rates, exact two-sided
        binomial p-value, and rejection decision.

    Notes
    -----
    This is an exact binomial test of unconditional coverage. It is
    distinct from the asymptotic likelihood-ratio test of Kupiec.
    The binomial model assumes a constant exception probability and
    independent Bernoulli observations. It does not test independence.
    """
    from scipy.stats import binomtest

    _validate_confidence(confidence)

    if (
        isinstance(significance_level, (bool, np.bool_))
        or not isinstance(
            significance_level,
            (int, float, np.integer, np.floating),
        )
        or not np.isfinite(significance_level)
        or not 0 < significance_level < 1
    ):
        raise ValueError("significance_level must be between 0 and 1.")

    try:
        values = np.asarray(exceptions)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "exceptions must be a one-dimensional 0/1 sequence."
        ) from exc

    if values.ndim != 1 or values.size == 0:
        raise ValueError(
            "exceptions must be a non-empty one-dimensional sequence."
        )

    if values.dtype.kind not in "biuf":
        raise ValueError("exceptions must contain only 0/1 values.")

    if not np.isfinite(values.astype(float)).all():
        raise ValueError("exceptions must contain finite values.")

    if not np.isin(values, [0, 1, False, True]).all():
        raise ValueError("exceptions must contain only 0/1 values.")

    values = values.astype(int)
    n = int(values.size)
    x = int(values.sum())
    expected_rate = 1.0 - float(confidence)
    observed_rate = x / n

    test_result = binomtest(
        k=x,
        n=n,
        p=expected_rate,
        alternative="two-sided",
    )
    p_value = float(test_result.pvalue)

    return {
        "n_observations": n,
        "n_exceptions": x,
        "expected_exception_rate": float(expected_rate),
        "observed_exception_rate": float(observed_rate),
        "p_value": p_value,
        "significance_level": float(significance_level),
        "reject_null": bool(p_value < significance_level),
        "p_value_method": "exact_binomial_two_sided",
        "assumption": (
            "Constant exception probability and independent "
            "Bernoulli observations."
        ),
    }
