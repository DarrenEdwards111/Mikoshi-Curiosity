import pytest

from mikoshi_curiosity.amplification import (
    one_hot,
    run_amplification_audit,
    shared_index_dag,
    shared_index_gate_count,
)


def test_explicit_shared_dag_computes_index():
    data = (False, True, False, True)
    assert not shared_index_dag(data, one_hot(0, 4))
    assert shared_index_dag(data, one_hot(1, 4))
    assert shared_index_dag(data, one_hot(3, 4))


@pytest.mark.parametrize("bits", range(1, 9))
def test_linear_amplification_upper_bound_is_exhaustively_exact(bits):
    audit = run_amplification_audit(bits)
    assert audit.exact
    assert audit.checked_rows_and_queries == (2 ** bits) * bits
    assert audit.explicit_upper_bound_gates == 2 * bits - 1
    assert audit.cnf_clauses == 3 * bits
    assert not audit.superpolynomial_amplification_possible_for_this_family


def test_bad_dimensions_are_rejected():
    with pytest.raises(ValueError):
        shared_index_gate_count(0)
    with pytest.raises(ValueError):
        shared_index_dag((True,), (False, True))
