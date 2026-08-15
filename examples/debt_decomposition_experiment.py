"""Exhaustively audit residual-row debt on small abstract fanout boundaries."""

from mikoshi_curiosity import run_debt_experiment

for rows in range(2, 6):
    result = run_debt_experiment(rows)
    print(
        f"rows={rows} "
        f"gate={result.gate_bound.status}:{result.gate_bound.checked} "
        f"fanout={result.fanout_bound.status}:{result.fanout_bound.checked} "
        f"terminal={result.terminal_load.status}:{result.terminal_load.counterexample}"
    )
