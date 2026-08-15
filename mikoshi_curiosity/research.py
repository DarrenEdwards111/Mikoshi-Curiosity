"""Open-ended conjecture generation and falsification for research workflows.

The core curiosity engine explores neighbors supplied by a ``StateSpace``.
This module makes that neighborhood generative: candidates are typed
conjectures, fuzzy concept similarity proposes analogies, critics attack each
candidate, and optional proof adapters certify survivors.

No model or theorem prover is hard-coded.  Commercial applications inject an
LLM-backed ``ConjectureGenerator`` and Lean/SMT adapters; tests and offline
users can use the deterministic template implementations included here.
"""

from __future__ import annotations

import hashlib
import math
import re
import subprocess
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from mikoshi_curiosity.space import State, StateSpace


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", text.lower())


class HashEmbedder:
    """Dependency-free fuzzy text embedder using signed feature hashing."""

    def __init__(self, dimensions: int = 128):
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def __call__(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float64)
        for token in _tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            raw = int.from_bytes(digest, "big")
            vector[raw % self.dimensions] += 1.0 if raw & 1 else -1.0
        norm = float(np.linalg.norm(vector))
        return vector if norm == 0 else vector / norm


@dataclass(frozen=True)
class Concept:
    """A reusable research concept and its semantic representation."""

    name: str
    description: str
    tags: Tuple[str, ...] = ()
    source: str = ""

    @property
    def text(self) -> str:
        return " ".join((self.name, self.description, *self.tags))


class ConceptGraph:
    """Fuzzy concept graph supporting semantic jumps and analogical search."""

    def __init__(self, concepts: Iterable[Concept] = (), embedder: Optional[Callable[[str], np.ndarray]] = None):
        self.embedder = embedder or HashEmbedder()
        self._concepts: Dict[str, Concept] = {}
        self._embeddings: Dict[str, np.ndarray] = {}
        self._edges: Dict[str, Dict[str, float]] = {}
        for concept in concepts:
            self.add(concept)

    def add(self, concept: Concept) -> None:
        self._concepts[concept.name] = concept
        self._embeddings[concept.name] = np.asarray(self.embedder(concept.text), dtype=np.float64)
        self._edges.setdefault(concept.name, {})

    def connect(self, left: str, right: str, weight: float = 1.0) -> None:
        if left not in self._concepts or right not in self._concepts:
            raise KeyError("both concepts must be added before connecting them")
        self._edges[left][right] = float(weight)
        self._edges[right][left] = float(weight)

    def get(self, name: str) -> Concept:
        return self._concepts[name]

    def similar(self, query: str, n: int = 5, exclude: Iterable[str] = ()) -> List[Tuple[Concept, float]]:
        blocked = set(exclude)
        q = np.asarray(self.embedder(query), dtype=np.float64)
        qnorm = float(np.linalg.norm(q))
        scored = []
        for name, concept in self._concepts.items():
            if name in blocked:
                continue
            vector = self._embeddings[name]
            denom = qnorm * float(np.linalg.norm(vector))
            cosine = 0.0 if denom == 0 else float(np.dot(q, vector) / denom)
            scored.append((concept, cosine))
        return sorted(scored, key=lambda item: item[1], reverse=True)[: max(0, n)]

    def neighbors(self, name: str, n: int = 5) -> List[Tuple[Concept, float]]:
        if name not in self._concepts:
            return self.similar(name, n=n)
        explicit = [(self._concepts[target], weight) for target, weight in self._edges[name].items()]
        fuzzy = self.similar(self._concepts[name].text, n=n * 2, exclude=(name,))
        merged: Dict[str, Tuple[Concept, float]] = {c.name: (c, score) for c, score in fuzzy}
        for concept, weight in explicit:
            old = merged.get(concept.name)
            merged[concept.name] = (concept, max(weight, old[1] if old else -math.inf))
        return sorted(merged.values(), key=lambda item: item[1], reverse=True)[: max(0, n)]

    def __len__(self) -> int:
        return len(self._concepts)


@dataclass(frozen=True)
class Conjecture:
    """Typed intermediate representation for a generated research claim."""

    name: str
    statement: str
    definitions: Tuple[str, ...] = ()
    assumptions: Tuple[str, ...] = ()
    proof_sketch: Tuple[str, ...] = ()
    tags: Tuple[str, ...] = ()
    provenance: Tuple[str, ...] = ()
    parent_id: str = ""
    generation: int = 0

    @property
    def text(self) -> str:
        return "\n".join((
            self.name,
            self.statement,
            *self.definitions,
            *self.assumptions,
            *self.proof_sketch,
            *self.tags,
        ))

    @property
    def id(self) -> str:
        digest = hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]
        return f"conjecture:{digest}"


class ConjectureGenerator(Protocol):
    def generate(self, seed: Conjecture, concepts: Sequence[Concept], n: int) -> Sequence[Conjecture]: ...


class ConjectureMutator(Protocol):
    def mutate(self, conjecture: Conjecture, graph: ConceptGraph, n: int) -> Sequence[Conjecture]: ...


class CallableConjectureGenerator:
    """Adapt a Python callable to the conjecture-generator protocol."""

    def __init__(self, generator: Callable[[Conjecture, Sequence[Concept], int], Sequence[Conjecture]]):
        self._generator = generator

    def generate(self, seed: Conjecture, concepts: Sequence[Concept], n: int) -> Sequence[Conjecture]:
        return self._generator(seed, concepts, n)


@dataclass(frozen=True)
class Critique:
    critic: str
    severity: str
    message: str
    evidence: str = ""

    @property
    def fatal(self) -> bool:
        return self.severity.lower() in {"error", "fatal"}


class ResearchCritic(Protocol):
    name: str
    def review(self, conjecture: Conjecture) -> Sequence[Critique]: ...


class CallableResearchCritic:
    """Adapt an external checker or model callback to the critic protocol."""

    def __init__(self, name: str, reviewer: Callable[[Conjecture], Sequence[Critique]]):
        self.name = name
        self._reviewer = reviewer

    def review(self, conjecture: Conjecture) -> Sequence[Critique]:
        return self._reviewer(conjecture)


@dataclass(frozen=True)
class ProofResult:
    adapter: str
    status: str
    output: str = ""
    artifact: str = ""

    @property
    def verified(self) -> bool:
        return self.status == "verified"


class ProofAdapter(Protocol):
    name: str
    def check(self, conjecture: Conjecture) -> ProofResult: ...


@dataclass(frozen=True)
class CandidateEvaluation:
    verdict: str
    score: float
    critiques: Tuple[Critique, ...] = ()
    proofs: Tuple[ProofResult, ...] = ()


class CircularityCritic:
    """Flags assumptions that restate the requested conclusion."""

    name = "circularity"

    def __init__(self, target_terms: Sequence[str] = ("p != np", "p ≠ np", "sat not in p", "sat ∉ p")):
        self.target_terms = tuple(term.lower() for term in target_terms)

    def review(self, conjecture: Conjecture) -> Sequence[Critique]:
        issues = []
        for assumption in conjecture.assumptions:
            lowered = assumption.lower()
            if any(term in lowered for term in self.target_terms):
                issues.append(Critique(self.name, "error", "assumption restates the target", assumption))
            if any(marker in lowered for marker in ("assume hard", "assumed lower bound", "assume no sharing")):
                issues.append(Critique(self.name, "error", "load-bearing property is assumed", assumption))
        return issues


class KnownFailureCritic:
    """Pattern critic for domain-specific counterexamples and barriers."""

    name = "known-failures"

    def __init__(self, rules: Optional[Mapping[str, str]] = None):
        self.rules = dict(rules or {
            "local transition": "locality does not bound global semantic influence",
            "distinct residual": "syntactic distinctness is not computational independence",
            "one output bit": "decision correctness does not preserve every witness or residual answer",
            "different observer budget": "different budgets may still have identical reachable languages",
            "gate support additive": "fanout permits one gate to serve many consumers",
        })

    def review(self, conjecture: Conjecture) -> Sequence[Critique]:
        text = conjecture.text.lower()
        return [Critique(self.name, "warning", message, pattern)
                for pattern, message in self.rules.items() if pattern in text]


class CompletenessCritic:
    """Requires a conjecture to expose assumptions and a proof route."""

    name = "completeness"

    def review(self, conjecture: Conjecture) -> Sequence[Critique]:
        issues = []
        if not conjecture.statement.strip():
            issues.append(Critique(self.name, "error", "missing theorem statement"))
        if not conjecture.proof_sketch:
            issues.append(Critique(self.name, "warning", "missing proof decomposition"))
        if not conjecture.definitions:
            issues.append(Critique(self.name, "warning", "new primitives are not defined"))
        return issues


class CallableProofAdapter:
    """Wrap any local checker as a proof adapter."""

    def __init__(self, name: str, checker: Callable[[Conjecture], ProofResult]):
        self.name = name
        self._checker = checker

    def check(self, conjecture: Conjecture) -> ProofResult:
        result = self._checker(conjecture)
        if result.adapter != self.name:
            result = replace(result, adapter=self.name)
        return result


class CommandProofAdapter:
    """Run a local proof command against generated source.

    ``command`` is an argument vector.  ``{file}`` placeholders are replaced by
    a temporary source path.  No shell is used.
    """

    def __init__(self, name: str, command: Sequence[str], render: Callable[[Conjecture], str], suffix: str = ".txt", timeout: float = 30.0):
        self.name = name
        self.command = tuple(command)
        self.render = render
        self.suffix = suffix
        self.timeout = timeout

    def check(self, conjecture: Conjecture) -> ProofResult:
        with tempfile.TemporaryDirectory(prefix="mikoshi-proof-") as directory:
            path = Path(directory) / f"candidate{self.suffix}"
            path.write_text(self.render(conjecture), encoding="utf-8")
            command = [part.replace("{file}", str(path)) for part in self.command]
            try:
                run = subprocess.run(command, text=True, capture_output=True, timeout=self.timeout, check=False)
            except (OSError, subprocess.TimeoutExpired) as exc:
                return ProofResult(self.name, "error", str(exc))
            output = (run.stdout + run.stderr).strip()
            return ProofResult(self.name, "verified" if run.returncode == 0 else "rejected", output)


class ResearchEvaluator:
    """Adversarial evaluation pipeline for generated conjectures."""

    def __init__(self, critics: Sequence[ResearchCritic] = (), proof_adapters: Sequence[ProofAdapter] = ()):
        self.critics = tuple(critics)
        self.proof_adapters = tuple(proof_adapters)

    def evaluate(self, conjecture: Conjecture) -> CandidateEvaluation:
        critiques = tuple(issue for critic in self.critics for issue in critic.review(conjecture))
        if any(issue.fatal for issue in critiques):
            return CandidateEvaluation("rejected", -10.0 - len(critiques), critiques, ())
        proofs = tuple(adapter.check(conjecture) for adapter in self.proof_adapters)
        if any(result.status in {"rejected", "error"} for result in proofs):
            verdict = "falsified"
        elif proofs and all(result.verified for result in proofs):
            verdict = "verified"
        else:
            verdict = "survivor"
        score = 10.0 * sum(result.verified for result in proofs) - 2.0 * len(critiques)
        return CandidateEvaluation(verdict, score, critiques, proofs)


class TemplateGenerator:
    """Offline generator that converts fuzzy neighboring concepts into analogies."""

    def generate(self, seed: Conjecture, concepts: Sequence[Concept], n: int) -> Sequence[Conjecture]:
        generated = []
        for concept in concepts[:n]:
            generated.append(Conjecture(
                name=f"{seed.name} via {concept.name}",
                statement=f"Transfer {concept.name} to strengthen: {seed.statement}",
                definitions=seed.definitions + (f"Analogy primitive from {concept.name}: {concept.description}",),
                assumptions=seed.assumptions,
                proof_sketch=seed.proof_sketch + (f"Establish a structure-preserving map from {concept.name}.",),
                tags=tuple(dict.fromkeys(seed.tags + concept.tags + (concept.name,))),
                provenance=seed.provenance + (f"analogy:{concept.name}",),
                parent_id=seed.id,
                generation=seed.generation + 1,
            ))
        return generated


class FuzzyAnalogyMutator:
    def mutate(self, conjecture: Conjecture, graph: ConceptGraph, n: int) -> Sequence[Conjecture]:
        concepts = [item[0] for item in graph.similar(conjecture.text, n=n, exclude=conjecture.tags)]
        return TemplateGenerator().generate(conjecture, concepts, n)


class AssumptionChallengeMutator:
    """Creates candidates that attempt to derive, weaken, or falsify assumptions."""

    def mutate(self, conjecture: Conjecture, graph: ConceptGraph, n: int) -> Sequence[Conjecture]:
        out = []
        for index, assumption in enumerate(conjecture.assumptions[:n]):
            remaining = conjecture.assumptions[:index] + conjecture.assumptions[index + 1:]
            out.append(replace(
                conjecture,
                name=f"{conjecture.name} without assumption {index + 1}",
                assumptions=remaining,
                proof_sketch=conjecture.proof_sketch + (f"Derive or refute omitted assumption: {assumption}",),
                provenance=conjecture.provenance + (f"challenge:{assumption}",),
                parent_id=conjecture.id,
                generation=conjecture.generation + 1,
            ))
        return out


class ResearchStateSpace(StateSpace):
    """Dynamic, open-ended state space of generated and audited conjectures."""

    def __init__(self, graph: ConceptGraph, generator: Optional[ConjectureGenerator] = None,
                 mutators: Sequence[ConjectureMutator] = (), evaluator: Optional[ResearchEvaluator] = None,
                 embedder: Optional[Callable[[str], np.ndarray]] = None, archive=None):
        self.graph = graph
        self.generator = generator or TemplateGenerator()
        self.mutators = tuple(mutators) or (FuzzyAnalogyMutator(), AssumptionChallengeMutator())
        self.evaluator = evaluator or ResearchEvaluator((CompletenessCritic(), CircularityCritic(), KnownFailureCritic()))
        self.embedder = embedder or graph.embedder
        self.archive = archive
        self._states: Dict[str, State] = {}
        self._conjectures: Dict[str, Conjecture] = {}

    def add(self, conjecture: Conjecture) -> State:
        existing = self._states.get(conjecture.id)
        if existing is not None:
            return existing
        evaluation = self.evaluator.evaluate(conjecture)
        if self.archive is not None:
            self.archive.save(conjecture, evaluation)
        features = {
            "name": conjecture.name,
            "statement": conjecture.statement,
            "generation": conjecture.generation,
            "assumption_count": len(conjecture.assumptions),
            "tag_count": len(conjecture.tags),
            "verdict": evaluation.verdict,
            "research_score": evaluation.score,
        }
        state = State(conjecture.id, features, self.embedder(conjecture.text), {
            "conjecture": conjecture,
            "evaluation": evaluation,
        })
        self._states[state.id] = state
        self._conjectures[state.id] = conjecture
        return state

    def conjecture(self, state: State) -> Conjecture:
        return self._conjectures[state.id]

    def get_neighbors(self, state: State, n: int = 10) -> List[State]:
        seed = self.conjecture(state)
        fuzzy = [item[0] for item in self.graph.similar(seed.text, n=max(n, 1), exclude=seed.tags)]
        proposals = list(self.generator.generate(seed, fuzzy, n))
        for mutator in self.mutators:
            proposals.extend(mutator.mutate(seed, self.graph, n))
        unique: List[State] = []
        seen = set()
        for conjecture in proposals:
            if conjecture.id == state.id or conjecture.id in seen:
                continue
            seen.add(conjecture.id)
            unique.append(self.add(conjecture))
        unique.sort(key=lambda item: item.features["research_score"], reverse=True)
        return unique[: max(0, n)]

    def get_random(self, n: int = 10) -> List[State]:
        return list(self._states.values())[: max(0, n)]

    def get_state(self, id: str) -> State:
        return self._states[id]

    def embed(self, state: State) -> np.ndarray:
        return np.asarray(self.embedder(self.conjecture(state).text), dtype=np.float64)

    def size(self) -> int:
        return len(self._states)
