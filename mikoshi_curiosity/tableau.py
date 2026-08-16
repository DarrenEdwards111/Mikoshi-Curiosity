"""Repeated-recovery audit for succinct transition-tableau proposals.

A fixed ripple-counter transition gadget can visit exponentially many states
with only linear reusable hardware and linear frontier width.  Ordinary
Cook--Levin/CNF tableaux, however, unroll time: their size and combinational
gate events grow linearly with the number of transitions.  This experiment
detects attempts to hide exponential work in a binary-encoded time bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple


State = Tuple[bool, ...]


def counter_step(state: Sequence[bool]) -> State:
    """Little-endian increment modulo ``2^width``."""
    if not state:
        raise ValueError("state must contain at least one bit")
    output = []
    carry = True
    for bit in state:
        output.append(bool(bit) != carry)
        carry = carry and bool(bit)
    return tuple(output)


def counter_trace(width: int, steps: int) -> Tuple[State, ...]:
    if width <= 0 or steps < 0:
        raise ValueError("width must be positive and steps nonnegative")
    state: State = (False,) * width
    trace = [state]
    for _ in range(steps):
        state = counter_step(state)
        trace.append(state)
    return tuple(trace)


def ripple_transition_gate_bound(width: int) -> int:
    """Safe linear bound for XOR/AND carry propagation."""
    if width <= 0:
        raise ValueError("width must be positive")
    return 2 * width


@dataclass(frozen=True)
class TableauRecoveryAudit:
    width: int
    steps: int
    unique_states: int
    frontier_bits: int
    reusable_transition_gates: int
    unrolled_gate_events: int
    ordinary_tableau_clauses_lower_bound: int
    time_is_polynomial_in_width: bool

    @property
    def exponential_state_coverage(self) -> bool:
        return self.unique_states == 2 ** self.width

    @property
    def hides_exponential_time_in_succinct_step_count(self) -> bool:
        return self.exponential_state_coverage and not self.time_is_polynomial_in_width


def run_tableau_recovery_audit(width: int, steps: int, polynomial_degree: int = 3) -> TableauRecoveryAudit:
    if polynomial_degree <= 0:
        raise ValueError("polynomial_degree must be positive")
    trace = counter_trace(width, steps)
    transition_gates = ripple_transition_gate_bound(width)
    return TableauRecoveryAudit(
        width=width,
        steps=steps,
        unique_states=len(set(trace)),
        frontier_bits=width,
        reusable_transition_gates=transition_gates,
        unrolled_gate_events=steps * transition_gates,
        # Every transition must constrain at least each next-state bit.
        ordinary_tableau_clauses_lower_bound=steps * width,
        time_is_polynomial_in_width=steps <= width ** polynomial_degree,
    )
