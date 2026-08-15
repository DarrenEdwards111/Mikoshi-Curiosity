"""Audit the external SAT residual-query orientation theorem on small rows."""

from mikoshi_curiosity import run_sat_orientation_experiment


for width, copies in ((1, 1), (2, 1), (2, 2), (3, 2)):
    result = run_sat_orientation_experiment(width, copies)
    bits = width * copies
    print(
        f"bits={bits}: correctness={result.correctness.status} "
        f"({result.correctness.checked}), full-family={result.full_family_injective.status} "
        f"({result.full_family_injective.checked}), one-query={result.single_query_injective.status} "
        f"({result.single_query_injective.checked})"
    )
    if result.single_query_injective.counterexample:
        print(f"  one-query collision: {dict(result.single_query_injective.counterexample)}")
