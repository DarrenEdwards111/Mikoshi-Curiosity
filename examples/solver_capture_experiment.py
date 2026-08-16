"""Audit the strongest finite-testable solver-capture candidate."""

from mikoshi_curiosity.solver_capture import audit_hankel_capture


for name, relation in (
    ("identity", ((1, 0), (0, 1))),
    ("xor", ((0, 1), (1, 0))),
    ("triangular", ((1, 1), (0, 1))),
):
    audit = audit_hankel_capture(relation)
    print(
        f"{name}: old_rank={audit.old_rank}, copied_rank={audit.copied_rank}, "
        f"product_rank={audit.product_rank}, copied_doubles={audit.copied_doubles}, "
        f"product_doubles={audit.product_doubles}"
    )
