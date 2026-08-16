import pytest

from mikoshi_curiosity.tableau import (
    counter_step,
    counter_trace,
    ripple_transition_gate_bound,
    run_tableau_recovery_audit,
)


def test_counter_transition_and_full_cycle():
    assert counter_step((False, False, False)) == (True, False, False)
    assert counter_step((True, False, False)) == (False, True, False)
    assert counter_step((True, True, True)) == (False, False, False)
    assert len(set(counter_trace(4, 15))) == 16


@pytest.mark.parametrize("width", range(2, 9))
def test_exponential_recovery_requires_exponential_unrolling(width):
    steps = 2 ** width - 1
    audit = run_tableau_recovery_audit(width, steps, polynomial_degree=2)
    assert audit.exponential_state_coverage
    assert audit.frontier_bits == width
    assert audit.reusable_transition_gates == 2 * width
    assert audit.unrolled_gate_events == steps * 2 * width
    assert audit.ordinary_tableau_clauses_lower_bound == steps * width
    if steps > width ** 2:
        assert audit.hides_exponential_time_in_succinct_step_count


@pytest.mark.parametrize("width", range(2, 9))
def test_polynomial_time_yields_only_polynomially_many_recoveries(width):
    steps = width ** 2
    audit = run_tableau_recovery_audit(width, steps, polynomial_degree=2)
    assert audit.time_is_polynomial_in_width
    assert audit.unique_states <= steps + 1
    assert audit.unrolled_gate_events <= 2 * width ** 3


def test_invalid_parameters():
    with pytest.raises(ValueError):
        counter_trace(0, 1)
    with pytest.raises(ValueError):
        run_tableau_recovery_audit(2, -1)
