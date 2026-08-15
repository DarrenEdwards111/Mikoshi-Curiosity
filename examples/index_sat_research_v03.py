"""Run an LLM-driven research loop on the INDEX-to-SAT lift frontier.

Requires a local Ollama server by default. Set OLLAMA_MODEL and OLLAMA_URL to
select another locally installed model/server. The default is deliberately a
small model so the example runs on modest local hardware.
"""

import os
from pathlib import Path

from mikoshi_curiosity import (
    CircularityCritic, CompletenessCritic, Concept, ConceptGraph, Conjecture,
    CuriosityEngine, KnownFailureCritic, LLMConjectureGenerator, OllamaProvider,
    ResearchArchive, ResearchEvaluator, ResearchStateSpace,
)

graph = ConceptGraph([
    Concept("cross-frame monogamy", "charge independent orientation debt despite fanout", ("holography",)),
    Concept("information complexity", "charge revealed information rather than gate occurrences", ("direct-sum",)),
    Concept("random restrictions", "simplify circuits while retaining hard residuals", ("complexity",)),
    Concept("proof interpolation", "extract communication protocols from proof objects", ("proof-complexity",)),
    Concept("sheaf obstruction", "incompatible local views have no global section", ("topology",)),
    Concept("pebbling", "trade DAG reuse against time and retained state", ("time-space",)),
])

archive_path = Path(os.environ.get("RESEARCH_ARCHIVE", "index-sat-research.db"))
provider = OllamaProvider(
    model=os.environ.get("OLLAMA_MODEL", "llama3.2:latest"),
    base_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
)

with ResearchArchive(archive_path) as archive:
    generator = LLMConjectureGenerator(provider, failure_memory=archive)
    evaluator = ResearchEvaluator((CompletenessCritic(), CircularityCritic(), KnownFailureCritic()))
    space = ResearchStateSpace(graph, generator=generator, evaluator=evaluator, archive=archive)
    seed = space.add(Conjecture(
        "Unrestricted INDEX-to-SAT lift frontier",
        "Derive an unconditional general-circuit lower bound from the proved split-preserving INDEX-to-SAT embedding.",
        definitions=("The trusted slice has an injective residual-row map.",),
        assumptions=(
            "Lean proves directSumIndexSATEmbedding.",
            "Lean proves directSumIndexSATSlice_bits_ge for exact one-way protocols.",
        ),
        proof_sketch=(
            "Invent a computation-level invariant stable under arbitrary DAG fanout.",
            "Falsify it on finite circuits before attempting Lean formalization.",
        ),
        tags=("SAT", "INDEX", "general-circuits"),
    ))
    result = CuriosityEngine(space, strategy="balanced").explore(
        seed, budget=int(os.environ.get("RESEARCH_BUDGET", "20")), neighbors_per_step=5,
    )
    print(result.summary())
    print(f"Persistent candidates: {archive.count()}")
    for discovery in result.top(10):
        state = discovery.state
        print(state.features["verdict"], state.features["name"], discovery.score)
