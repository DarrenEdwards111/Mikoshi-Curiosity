"""Executable SAT-derived orientation families for direct-sum INDEX rows.

The family theorem is intentionally external: exact SAT correctness on every
generated residual query recovers the complete INDEX row.  It does not claim
that one execution of a SAT decider stores or exposes the whole row.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence, Tuple

from mikoshi_curiosity.model_finder import FiniteModelFinder, FiniteModelSpec, ModelSearchResult


Clause = Tuple[int, ...]
CNF = Tuple[Clause, ...]
IndexData = Tuple[bool, ...]
SATDecider = Callable[[CNF], bool]


@dataclass(frozen=True, order=True)
class IndexSATQuery:
    copy: int
    index: int


def bit_cnf(bit: bool) -> CNF:
    """A concrete variable-free CNF satisfiable exactly when ``bit`` is true."""
    return () if bit else ((),)


def cnf_satisfiable(cnf: CNF) -> bool:
    """Decide the variable-free CNFs produced by :func:`bit_cnf`."""
    if any(literal != 0 for clause in cnf for literal in clause):
        raise ValueError("cnf_satisfiable only accepts variable-free CNFs")
    return all(bool(clause) for clause in cnf)


def query_family(width: int, copies: int) -> Tuple[IndexSATQuery, ...]:
    if width <= 0 or copies <= 0:
        raise ValueError("width and copies must be positive")
    return tuple(IndexSATQuery(copy, index) for copy in range(copies) for index in range(width))


def _validate_data(data: Sequence[bool], width: int, copies: int) -> IndexData:
    query_family(width, copies)
    if len(data) != width * copies:
        raise ValueError("data must contain exactly width * copies bits")
    return tuple(bool(bit) for bit in data)


def index_answer(data: Sequence[bool], width: int, copies: int, query: IndexSATQuery) -> bool:
    row = _validate_data(data, width, copies)
    if not (0 <= query.copy < copies and 0 <= query.index < width):
        raise ValueError("query is outside the INDEX table")
    return row[query.copy * width + query.index]


def index_query_cnf(data: Sequence[bool], width: int, copies: int, query: IndexSATQuery) -> CNF:
    return bit_cnf(index_answer(data, width, copies, query))


def sat_orientation(
    data: Sequence[bool],
    width: int,
    copies: int,
    decider: SATDecider = cnf_satisfiable,
) -> Tuple[bool, ...]:
    """Observe SAT answers for every selector in canonical copy-major order."""
    return tuple(decider(index_query_cnf(data, width, copies, query)) for query in query_family(width, copies))


def correctness_forces_orientation(
    data: Sequence[bool],
    width: int,
    copies: int,
    decider: SATDecider = cnf_satisfiable,
) -> bool:
    """Exact correctness on the generated CNFs forces the INDEX orientation."""
    row = _validate_data(data, width, copies)
    return sat_orientation(row, width, copies, decider) == row


def orientation_is_injective(
    left: Sequence[bool],
    right: Sequence[bool],
    width: int,
    copies: int,
    decider: SATDecider = cnf_satisfiable,
) -> bool:
    lhs = _validate_data(left, width, copies)
    rhs = _validate_data(right, width, copies)
    return lhs == rhs or sat_orientation(lhs, width, copies, decider) != sat_orientation(rhs, width, copies, decider)


@dataclass(frozen=True)
class SATOrientationExperiment:
    width: int
    copies: int
    correctness: ModelSearchResult
    full_family_injective: ModelSearchResult
    single_query_injective: ModelSearchResult


def run_sat_orientation_experiment(width: int, copies: int, max_models: int = 100000) -> SATOrientationExperiment:
    """Audit full-family recovery and expose the one-query projection failure."""
    queries = query_family(width, copies)
    rows = tuple(itertools.product((False, True), repeat=width * copies))
    finder = FiniteModelFinder(max_models=max_models)
    correctness = finder.search(FiniteModelSpec(
        {"data": rows},
        lambda model: correctness_forces_orientation(model["data"], width, copies),
        "exact SAT answers recover every INDEX orientation bit",
    ))
    injective = finder.search(FiniteModelSpec(
        {"left": rows, "right": rows},
        lambda model: orientation_is_injective(model["left"], model["right"], width, copies),
        "the full SAT-query orientation separates distinct INDEX rows",
    ))
    first = queries[0]
    single = finder.search(FiniteModelSpec(
        {"left": rows, "right": rows},
        lambda model: (
            model["left"] == model["right"]
            or index_answer(model["left"], width, copies, first)
            != index_answer(model["right"], width, copies, first)
        ),
        "one fixed SAT residual query separates every INDEX row",
    ))
    return SATOrientationExperiment(width, copies, correctness, injective, single)
