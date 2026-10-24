
import numpy as np
import pytest

from quantmove.risk.var_backtesting import (
    christoffersen_independence_test,
)


def test_independence_test_returns_expected_structure():
    exceptions = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0]

    result = christoffersen_independence_test(exceptions)

    expected_keys = {
        "n_observations",
        "n_transitions",
        "n_00",
        "n_01",
        "n_10",
        "n_11",
        "lr_statistic",
        "p_value",
        "significance_level",
        "reject_null",
        "independence_identifiable",
    }

    assert set(result.keys()) == expected_keys
    assert result["n_observations"] == len(exceptions)
    assert result["n_transitions"] == len(exceptions) - 1
    assert 0 <= result["p_value"] <= 1
    assert isinstance(result["reject_null"], bool)


def test_independence_test_detects_clustered_exceptions():
    # Excepciones concentradas en bloques: patrón de dependencia.
    exceptions = [0] * 100 + [1] * 20 + [0] * 100 + [1] * 20

    result = christoffersen_independence_test(exceptions)

    assert result["lr_statistic"] > 0
    assert result["p_value"] < 0.05
    assert result["reject_null"] is True


def test_independence_test_accepts_balanced_transitions():
    # Transiciones con frecuencias iguales: máxima verosimilitud
    # bajo independencia cuando las probabilidades de transición
    # estimadas coinciden.
    exceptions = [0, 0, 1, 1] * 50

    result = christoffersen_independence_test(exceptions)

    assert result["lr_statistic"] >= 0
    assert 0 <= result["p_value"] <= 1


@pytest.mark.parametrize(
    "exceptions",
    [
        [],
        [[0, 1], [1, 0]],
        [0, 2, 1],
        [0, 1, np.nan],
        [0, 1, "x"],
    ],
)
def test_independence_test_rejects_invalid_inputs(exceptions):
    with pytest.raises((TypeError, ValueError)):
        christoffersen_independence_test(exceptions)


def test_independence_test_rejects_invalid_significance_level():
    with pytest.raises(ValueError):
        christoffersen_independence_test(
            [0, 1, 0, 1],
            significance_level=1.0,
        )


def test_independence_test_requires_two_observations():
    with pytest.raises(ValueError):
        christoffersen_independence_test([1])


def test_independence_test_flags_sequence_without_exceptions():
    result = christoffersen_independence_test([0] * 100)

    assert result["independence_identifiable"] is False


def test_independence_test_flags_sequence_with_only_exceptions():
    result = christoffersen_independence_test([1] * 100)

    assert result["independence_identifiable"] is False
