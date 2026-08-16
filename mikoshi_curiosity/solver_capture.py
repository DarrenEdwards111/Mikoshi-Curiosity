"""Finite audits for representation-invariant Gödel-tower capture proposals.

The semantic prefix/continuation relation of a stage is represented by a
zero-one matrix.  Its exact rational rank ignores gate names and duplicated
fanout wires.  Tensoring two genuinely independent relations squares rank;
copying or re-encoding one relation does not.  This isolates product capture
as a necessary semantic obligation rather than silently assuming it.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Sequence, Tuple


Matrix = Tuple[Tuple[int, ...], ...]


def _matrix(rows: Iterable[Iterable[int]]) -> Matrix:
    value = tuple(tuple(int(entry) for entry in row) for row in rows)
    if not value or not value[0] or any(len(row) != len(value[0]) for row in value):
        raise ValueError("matrix must be nonempty and rectangular")
    if any(entry not in (0, 1) for row in value for entry in row):
        raise ValueError("semantic relations must be zero-one matrices")
    return value


def rational_rank(rows: Sequence[Sequence[int]]) -> int:
    """Exact Gaussian-elimination rank over Q."""
    matrix = _matrix(rows)
    work = [[Fraction(entry) for entry in row] for row in matrix]
    height, width = len(work), len(work[0])
    pivot_row = 0
    for column in range(width):
        pivot = next((row for row in range(pivot_row, height) if work[row][column]), None)
        if pivot is None:
            continue
        work[pivot_row], work[pivot] = work[pivot], work[pivot_row]
        scale = work[pivot_row][column]
        work[pivot_row] = [entry / scale for entry in work[pivot_row]]
        for row in range(height):
            if row == pivot_row or not work[row][column]:
                continue
            factor = work[row][column]
            work[row] = [left - factor * right for left, right in zip(work[row], work[pivot_row])]
        pivot_row += 1
        if pivot_row == height:
            break
    return pivot_row


def kronecker(left: Sequence[Sequence[int]], right: Sequence[Sequence[int]]) -> Matrix:
    """Independent product of two finite semantic relations."""
    a, b = _matrix(left), _matrix(right)
    return tuple(
        tuple(a_entry * b_entry for a_entry in a_row for b_entry in b_row)
        for a_row in a
        for b_row in b
    )


def hankel_load(relation: Sequence[Sequence[int]]) -> int:
    """Rank-minus-one load used by the generated Hankel candidate."""
    return max(0, rational_rank(relation) - 1)


def capture_doubles(old: Sequence[Sequence[int]], new: Sequence[Sequence[int]]) -> bool:
    return hankel_load(new) >= 2 * hankel_load(old) + 1


@dataclass(frozen=True)
class CaptureAudit:
    old_rank: int
    copied_rank: int
    product_rank: int
    copied_doubles: bool
    product_doubles: bool


def audit_hankel_capture(relation: Sequence[Sequence[int]]) -> CaptureAudit:
    """Compare mere reuse with a genuine independent-product construction."""
    old = _matrix(relation)
    product = kronecker(old, old)
    return CaptureAudit(
        old_rank=rational_rank(old),
        copied_rank=rational_rank(old),
        product_rank=rational_rank(product),
        copied_doubles=capture_doubles(old, old),
        product_doubles=capture_doubles(old, product),
    )

