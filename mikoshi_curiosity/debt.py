"""Executable fanout-neutral residual-row debt experiments.

This module deliberately separates a valid local information inequality from
the stronger terminal-load premise needed by circuit lower-bound arguments.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Hashable, Sequence, Tuple

from mikoshi_curiosity.model_finder import FiniteModelFinder, FiniteModelSpec, ModelSearchResult


Signature = Tuple[Hashable, ...]


def class_count(signatures: Sequence[Hashable]) -> int:
    """Number of residual-row equivalence classes induced by signatures."""
    return len(set(signatures))


def refine_signatures(boundary: Sequence[Hashable], gate_bits: Sequence[bool]) -> Signature:
    if len(boundary) != len(gate_bits):
        raise ValueError("boundary and gate signatures must cover the same residual rows")
    return tuple((boundary[index], bool(gate_bits[index])) for index in range(len(boundary)))


def duplicate_fanout(signatures: Sequence[Hashable]) -> Signature:
    """Expose a second wire carrying exactly the same semantic signature."""
    return tuple((value, value) for value in signatures)


def unresolved_debt(row_count: int, signatures: Sequence[Hashable]) -> float:
    """Bits still needed to distinguish every residual row.

    The quantity is zero only at a discrete partition and is never negative.
    """
    if row_count <= 0 or len(signatures) != row_count:
        raise ValueError("row_count must be positive and match the signatures")
    return max(0.0, math.log2(row_count) - math.log2(class_count(signatures)))


def gate_refinement_bound(boundary: Sequence[Hashable], gate_bits: Sequence[bool]) -> bool:
    """One Boolean observation creates at most twice as many classes."""
    return class_count(refine_signatures(boundary, gate_bits)) <= 2 * class_count(boundary)


def fanout_is_information_neutral(signatures: Sequence[Hashable]) -> bool:
    return class_count(duplicate_fanout(signatures)) == class_count(signatures)


def terminal_rows_distinguished(output_bits: Sequence[bool]) -> bool:
    return class_count(tuple(bool(bit) for bit in output_bits)) == len(output_bits)


@dataclass(frozen=True)
class DebtExperiment:
    rows: int
    gate_bound: ModelSearchResult
    fanout_bound: ModelSearchResult
    terminal_load: ModelSearchResult


def run_debt_experiment(rows: int, max_models: int = 100000) -> DebtExperiment:
    """Exhaustively audit the measure for a fixed number of residual rows.

    ``gate_bound`` and ``fanout_bound`` search for violations of the local
    claims. ``terminal_load`` searches for a counterexample to the universal
    assertion that an arbitrary one-bit output distinguishes all rows.
    """
    if rows <= 0:
        raise ValueError("rows must be positive")
    labels = tuple(itertools.product(range(rows), repeat=rows))
    bits = tuple(itertools.product((False, True), repeat=rows))
    nonconstant_bits = tuple(value for value in bits if len(set(value)) > 1)
    finder = FiniteModelFinder(max_models=max_models)
    gate = finder.search(FiniteModelSpec(
        {"boundary": labels, "gate": bits},
        lambda model: gate_refinement_bound(model["boundary"], model["gate"]),
        "one Boolean gate refines residual classes by at most a factor of two",
    ))
    fanout = finder.search(FiniteModelSpec(
        {"signature": bits},
        lambda model: fanout_is_information_neutral(model["signature"]),
        "duplicating an identical semantic wire adds no distinctions",
    ))
    terminal = finder.search(FiniteModelSpec(
        {"output": nonconstant_bits or bits},
        lambda model: terminal_rows_distinguished(model["output"]),
        "one nonconstant decision bit distinguishes every residual row",
    ))
    return DebtExperiment(rows, gate, fanout, terminal)
