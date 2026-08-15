"""Show the explicit linear shared DAG for the split INDEX CNF slice."""

from mikoshi_curiosity import run_amplification_audit


for bits in range(1, 9):
    result = run_amplification_audit(bits)
    print(
        f"N={bits}: cases={result.checked_rows_and_queries}, exact={result.exact}, "
        f"forced_bits={result.forced_orientation_bits}, clauses={result.cnf_clauses}, "
        f"explicit_gates={result.explicit_upper_bound_gates}"
    )
