"""Search for a non-circular solver-to-Godel-tower capture invariant."""

import os
from pathlib import Path

from mikoshi_curiosity import (
    AssumptionAuditCritic,
    CircularityCritic,
    CodexCLIProvider,
    CompletenessCritic,
    Concept,
    ConceptGraph,
    Conjecture,
    CuriosityEngine,
    KnownFailureCritic,
    LLMConjectureGenerator,
    ResearchArchive,
    ResearchEvaluator,
    ResearchStateSpace,
)


graph = ConceptGraph([
    Concept("solver-relative reflection", "diagonalize a bounded reflection stage against a fixed SAT circuit", ("Godel", "reflection")),
    Concept("residual equivalence", "quotient inputs by the future Boolean behaviour of a circuit state", ("semantics", "Myhill-Nerode")),
    Concept("information complexity", "charge conditional information rather than output alphabet size", ("communication", "direct-sum")),
    Concept("DAG bisimulation", "identify duplicated and recomputed nodes by semantic transition behaviour", ("fanout", "sharing")),
    Concept("proof complexity", "relate solver traces to bounded reflection and feasible interpolation", ("SAT", "reflection")),
    Concept("intensional diagonalization", "make the next instance depend on a solver description without presolving it", ("recursion-theorem", "uniformity")),
])

archive_path = Path(os.environ.get("RESEARCH_ARCHIVE", "godel-solver-capture.db"))
provider = CodexCLIProvider(
    model=os.environ.get("CODEX_MODEL", "gpt-5.6-sol"),
    timeout=float(os.environ.get("CODEX_TIMEOUT", "600")),
)

with ResearchArchive(archive_path) as archive:
    generator = LLMConjectureGenerator(provider, failure_memory=archive)
    evaluator = ResearchEvaluator((
        CompletenessCritic(),
        CircularityCritic(),
        KnownFailureCritic({
            "output-only": "a correct decider can expose only one output bit; constant load is a countermodel",
            "distinct stages": "syntactically distinct reflection stages need not consume independent circuit capacity",
            "incompleteness": "Godel incompleteness alone is proof-theoretic, not a Boolean circuit-size lower bound",
            "many queries": "separate executions do not imply simultaneous hardware load",
            "recomputation": "a small transition gadget can serialize exponentially many states when time is unrolled",
            "residual classes": "many semantic classes do not imply a large circuit without an extraction theorem",
        }),
        AssumptionAuditCritic({
            "solver capture doubling": "assumes the exact missing recurrence",
            "computation-faithful": "must be constructed and bounded for arbitrary DAG circuits, not postulated",
            "independent information": "must prove independence survives arbitrary fanout and representation changes",
            "cannot share": "assumes the required anti-sharing theorem",
            "each stage doubles": "restates the desired conclusion unless derived gate-by-gate",
            "arbitrary sat circuit": "must give a uniform construction for every unrestricted SAT circuit",
        }),
    ))
    space = ResearchStateSpace(graph, generator=generator, evaluator=evaluator, archive=archive)
    seed = space.add(Conjecture(
        name="Unrestricted solver capture for a succinct Godel tower",
        statement=(
            "Construct, from every polynomial-size SAT circuit C_n, a representation-invariant semantic load mu "
            "on a uniformly polynomial-size solver-relative reflection tower such that mu(0)>=1 and "
            "mu(i+1)>=2*mu(i)+1, without assuming anti-sharing or reading more than C_n computes."
        ),
        definitions=(
            "Exact correctness alone is insufficient: the output-only load is constantly one.",
            "The quantitative scale theorem 2^k <= mu(k) is already Lean-verified.",
            "Tower depth k=(log n)^2 suffices for a superpolynomial contradiction.",
        ),
        assumptions=(
            "Each tower instance is uniformly constructible with total encoded size polynomial in n and k.",
            "The candidate load must be invariant under DAG duplication, fanout, and equivalent circuit rewrites.",
        ),
        proof_sketch=(
            "Propose an explicit load definition computable on finite circuits.",
            "State a local lemma connecting one reflection step to load growth.",
            "Search first for finite counterexamples involving sharing, serialization, or constant outputs.",
        ),
        tags=("SAT", "Godel", "circuit-lower-bound", "capture-doubling"),
    ))
    result = CuriosityEngine(space, strategy="balanced").explore(
        seed,
        budget=int(os.environ.get("RESEARCH_BUDGET", "24")),
        neighbors_per_step=int(os.environ.get("RESEARCH_NEIGHBORS", "6")),
    )
    print(result.summary())
    print(f"Persistent candidates: {archive.count()}")
    for discovery in result.top(15):
        state = discovery.state
        evaluation = state.metadata["evaluation"]
        conjecture = state.metadata["conjecture"]
        print(f"\n[{evaluation.verdict}] {conjecture.name} score={discovery.score:.6f}")
        print(conjecture.statement)
        for critique in evaluation.critiques:
            print(f"  - {critique.severity}: {critique.message} ({critique.evidence})")
