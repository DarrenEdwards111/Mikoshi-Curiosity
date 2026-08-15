"""Iterative Lean compiler feedback and source repair."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Sequence

from mikoshi_curiosity.research import Conjecture, ProofResult


class LeanRepairAdapter:
    name = "lean-repair"

    def __init__(self, command: Sequence[str], render: Callable[[Conjecture], str],
                 repair: Callable[[Conjecture, str, str, int], str], max_repairs: int = 3,
                 timeout: float = 60.0):
        if max_repairs < 0:
            raise ValueError("max_repairs cannot be negative")
        self.command = tuple(command)
        self.render = render
        self.repair = repair
        self.max_repairs = max_repairs
        self.timeout = timeout

    def check(self, conjecture: Conjecture) -> ProofResult:
        source = self.render(conjecture)
        diagnostics = ""
        with tempfile.TemporaryDirectory(prefix="mikoshi-lean-") as directory:
            path = Path(directory) / "Candidate.lean"
            for attempt in range(self.max_repairs + 1):
                path.write_text(source, encoding="utf-8")
                command = [part.replace("{file}", str(path)) for part in self.command]
                try:
                    run = subprocess.run(command, text=True, capture_output=True,
                                         timeout=self.timeout, check=False)
                except (OSError, subprocess.TimeoutExpired) as exc:
                    return ProofResult(self.name, "error", str(exc), source)
                diagnostics = (run.stdout + run.stderr).strip()
                if run.returncode == 0:
                    return ProofResult(self.name, "verified", f"kernel/checker accepted after {attempt} repairs\n{diagnostics}".strip(), source)
                if attempt < self.max_repairs:
                    source = self.repair(conjecture, source, diagnostics, attempt + 1)
        return ProofResult(self.name, "rejected", f"repair budget exhausted\n{diagnostics}".strip(), source)
