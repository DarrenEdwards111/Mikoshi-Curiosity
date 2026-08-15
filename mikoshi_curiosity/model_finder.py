"""Finite-domain counterexample search for executable conjectures."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence, Tuple

from mikoshi_curiosity.research import Conjecture, ProofResult


@dataclass(frozen=True)
class FiniteModelSpec:
    domains: Mapping[str, Sequence[object]]
    holds: Callable[[Mapping[str, object]], bool]
    description: str = ""


@dataclass(frozen=True)
class ModelSearchResult:
    status: str
    checked: int
    counterexample: Optional[Mapping[str, object]] = None
    reason: str = ""


class FiniteModelFinder:
    def __init__(self, max_models: int = 100000):
        if max_models <= 0:
            raise ValueError("max_models must be positive")
        self.max_models = max_models

    def search(self, spec: FiniteModelSpec) -> ModelSearchResult:
        names = tuple(spec.domains)
        domains = tuple(tuple(spec.domains[name]) for name in names)
        if any(not domain for domain in domains):
            return ModelSearchResult("exhausted", 0, reason="an input domain is empty")
        checked = 0
        for values in itertools.product(*domains):
            if checked >= self.max_models:
                return ModelSearchResult("incomplete", checked, reason="model limit reached")
            assignment = dict(zip(names, values))
            checked += 1
            try:
                holds = bool(spec.holds(assignment))
            except Exception as exc:
                return ModelSearchResult("error", checked, assignment, str(exc))
            if not holds:
                return ModelSearchResult("counterexample", checked, assignment)
        return ModelSearchResult("exhausted", checked)


class FiniteModelProofAdapter:
    """Falsify conjectures; finite exhaustion is deliberately not called a proof."""

    name = "finite-model"

    def __init__(self, spec_factory: Callable[[Conjecture], Optional[FiniteModelSpec]], max_models: int = 100000):
        self.spec_factory = spec_factory
        self.finder = FiniteModelFinder(max_models)

    def check(self, conjecture: Conjecture) -> ProofResult:
        spec = self.spec_factory(conjecture)
        if spec is None:
            return ProofResult(self.name, "skipped", "no executable finite model supplied")
        result = self.finder.search(spec)
        if result.status == "counterexample":
            return ProofResult(self.name, "rejected", f"counterexample after {result.checked} models: {dict(result.counterexample or {})}")
        if result.status == "error":
            return ProofResult(self.name, "error", result.reason)
        return ProofResult(self.name, "bounded", f"{result.status} after {result.checked} models; this is not a proof")
