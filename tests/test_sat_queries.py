import pytest

from mikoshi_curiosity.sat_queries import (
    IndexSATQuery,
    bit_cnf,
    cnf_satisfiable,
    correctness_forces_orientation,
    index_answer,
    orientation_is_injective,
    query_family,
    run_sat_orientation_experiment,
    sat_orientation,
)


def test_bit_cnf_is_satisfiable_exactly_for_true():
    assert bit_cnf(True) == ()
    assert bit_cnf(False) == ((),)
    assert cnf_satisfiable(bit_cnf(True))
    assert not cnf_satisfiable(bit_cnf(False))


def test_sat_orientation_recovers_the_flattened_index_row():
    data = (True, False, False, True)
    assert query_family(2, 2) == (
        IndexSATQuery(0, 0), IndexSATQuery(0, 1),
        IndexSATQuery(1, 0), IndexSATQuery(1, 1),
    )
    assert sat_orientation(data, 2, 2) == data
    assert correctness_forces_orientation(data, 2, 2)


def test_orientation_is_injective_but_one_query_is_not():
    assert orientation_is_injective((False, False), (False, True), 2, 1)
    assert index_answer((False, False), 2, 1, IndexSATQuery(0, 0)) == index_answer(
        (False, True), 2, 1, IndexSATQuery(0, 0)
    )


@pytest.mark.parametrize("width,copies", [(1, 1), (2, 1), (2, 2), (3, 2)])
def test_finite_orientation_audit(width, copies):
    result = run_sat_orientation_experiment(width, copies)
    assert result.correctness.status == "exhausted"
    assert result.full_family_injective.status == "exhausted"
    expected = "exhausted" if width * copies == 1 else "counterexample"
    assert result.single_query_injective.status == expected


def test_dimensions_and_queries_are_checked():
    with pytest.raises(ValueError):
        query_family(0, 1)
    with pytest.raises(ValueError):
        sat_orientation((True,), 2, 1)
    with pytest.raises(ValueError):
        index_answer((True,), 1, 1, IndexSATQuery(1, 0))
