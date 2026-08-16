import itertools

import pytest

from mikoshi_curiosity.solver_capture import (
    audit_hankel_capture,
    capture_doubles,
    hankel_load,
    kronecker,
    rational_rank,
)


def test_exact_rational_rank_and_product_identity():
    relation = ((1, 0), (0, 1))
    product = kronecker(relation, relation)
    assert rational_rank(relation) == 2
    assert rational_rank(product) == 4
    assert hankel_load(relation) == 1
    assert hankel_load(product) == 3


def test_copying_a_stage_does_not_force_capture_doubling():
    relation = ((1, 0), (0, 1))
    audit = audit_hankel_capture(relation)
    assert not audit.copied_doubles
    assert audit.product_doubles


def test_every_full_rank_two_by_two_relation_has_the_same_boundary():
    checked = 0
    for entries in itertools.product((0, 1), repeat=4):
        relation = (entries[:2], entries[2:])
        if rational_rank(relation) != 2:
            continue
        checked += 1
        audit = audit_hankel_capture(relation)
        assert audit.product_rank == 4
        assert not audit.copied_doubles
        assert audit.product_doubles
    assert checked == 6


def test_rank_one_seed_cannot_satisfy_positive_base_load():
    relation = ((1, 1), (1, 1))
    assert hankel_load(relation) == 0
    assert not capture_doubles(relation, kronecker(relation, relation))


@pytest.mark.parametrize("bad", [(), ((1, 0), (1,)), ((2,),)])
def test_bad_relations_are_rejected(bad):
    with pytest.raises(ValueError):
        rational_rank(bad)

