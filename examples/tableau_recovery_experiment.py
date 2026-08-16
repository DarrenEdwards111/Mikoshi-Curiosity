"""Compare polynomial and full-cycle repeated-recovery tableaux."""

from mikoshi_curiosity import run_tableau_recovery_audit


for width in range(2, 9):
    for label, steps in (("poly", width ** 2), ("cycle", 2 ** width - 1)):
        result = run_tableau_recovery_audit(width, steps, polynomial_degree=2)
        print(
            f"width={width} mode={label} steps={steps} unique={result.unique_states} "
            f"frontier={result.frontier_bits} hardware={result.reusable_transition_gates} "
            f"unrolled={result.unrolled_gate_events} tableau>={result.ordinary_tableau_clauses_lower_bound} "
            f"hidden_exp_time={result.hides_exponential_time_in_succinct_step_count}"
        )
