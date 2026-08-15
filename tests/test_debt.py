import math

import pytest

from mikoshi_curiosity import (
    class_count,
    duplicate_fanout,
    fanout_is_information_neutral,
    gate_refinement_bound,
    refine_signatures,
    run_debt_experiment,
    terminal_rows_distinguished,
    unresolved_debt,
)


def test_debt_reaches_zero_only_for_discrete_partition():
    assert unresolved_debt(4, (0, 0, 0, 0)) == 2.0
    assert unresolved_debt(4, (0, 1, 2, 3)) == 0.0
    assert math.isclose(unresolved_debt(3, (0, 0, 1)), math.log2(3) - 1)


def test_one_boolean_gate_can_at_most_double_classes():
    before = (0, 0, 1, 1)
    gate = (False, True, False, True)
    after = refine_signatures(before, gate)
    assert class_count(after) == 4
    assert gate_refinement_bound(before, gate)


def test_duplicate_fanout_is_semantically_neutral():
    signatures = ((0, False), (0, True), (1, False))
    assert class_count(duplicate_fanout(signatures)) == 3
    assert fanout_is_information_neutral(signatures)


def test_single_output_cannot_distinguish_three_rows():
    assert not terminal_rows_distinguished((False, True, False))


def test_finite_model_experiment_validates_local_rules_and_breaks_terminal_load():
    two_rows = run_debt_experiment(2)
    assert two_rows.terminal_load.status == "exhausted"
    result = run_debt_experiment(3)
    assert result.gate_bound.status == "exhausted"
    assert result.fanout_bound.status == "exhausted"
    assert result.terminal_load.status == "counterexample"


def test_mismatched_signatures_are_rejected():
    with pytest.raises(ValueError):
        refine_signatures((0,), (False, True))
