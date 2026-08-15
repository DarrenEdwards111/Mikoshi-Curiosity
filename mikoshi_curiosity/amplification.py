"""Quantitative audits for proposed INDEX-to-SAT amplification routes.

The locally split CNF family is useful for communication lower bounds, but its
SAT predicate is an INDEX multiplexer.  This module constructs the explicit
linear-size shared DAG, preventing accidental claims of superpolynomial
amplification for that particular family.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Sequence, Tuple


def one_hot(position: int, size: int) -> Tuple[bool, ...]:
    if size <= 0 or not 0 <= position < size:
        raise ValueError("position must address a positive vector")
    return tuple(index == position for index in range(size))


def shared_index_dag(data: Sequence[bool], selector: Sequence[bool]) -> bool:
    """The output of N parallel AND gates followed by a shared OR tree."""
    if not data or len(data) != len(selector):
        raise ValueError("data and selector must have the same positive length")
    return any(bool(bit) and bool(chosen) for bit, chosen in zip(data, selector))


def shared_index_gate_count(data_bits: int) -> int:
    """N AND gates plus N-1 binary OR gates (zero constant-folding assumed)."""
    if data_bits <= 0:
        raise ValueError("data_bits must be positive")
    return data_bits + max(0, data_bits - 1)


@dataclass(frozen=True)
class AmplificationAudit:
    data_bits: int
    checked_rows_and_queries: int
    exact: bool
    forced_orientation_bits: int
    explicit_upper_bound_gates: int
    cnf_clauses: int

    @property
    def gate_ratio(self) -> float:
        return self.explicit_upper_bound_gates / self.data_bits

    @property
    def superpolynomial_amplification_possible_for_this_family(self) -> bool:
        return False


def run_amplification_audit(data_bits: int) -> AmplificationAudit:
    """Exhaustively validate the linear DAG on every row and legal selector."""
    if data_bits <= 0:
        raise ValueError("data_bits must be positive")
    checked = 0
    exact = True
    for data in itertools.product((False, True), repeat=data_bits):
        for position in range(data_bits):
            checked += 1
            if shared_index_dag(data, one_hot(position, data_bits)) != data[position]:
                exact = False
    return AmplificationAudit(
        data_bits=data_bits,
        checked_rows_and_queries=checked,
        exact=exact,
        forced_orientation_bits=data_bits,
        explicit_upper_bound_gates=shared_index_gate_count(data_bits),
        cnf_clauses=3 * data_bits,
    )
