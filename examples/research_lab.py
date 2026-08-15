"""Minimal open-ended conjecture laboratory."""

from mikoshi_curiosity import (
    CircularityCritic,
    CompletenessCritic,
    Concept,
    ConceptGraph,
    Conjecture,
    CuriosityEngine,
    KnownFailureCritic,
    ResearchEvaluator,
    ResearchStateSpace,
)

concepts = ConceptGraph([
    Concept("cross-frame monogamy", "a shared resource cannot erase independent debt in two transverse frames", ("holography", "rank")),
    Concept("communication direct sum", "independent rows require additive communication", ("communication",)),
    Concept("sheaf obstruction", "locally consistent views may lack a compatible global section", ("topology",)),
    Concept("pebbling tradeoff", "DAG evaluation exchanges time, space, and recomputation", ("complexity",)),
])
concepts.connect("cross-frame monogamy", "communication direct sum", 2.0)

evaluator = ResearchEvaluator((CompletenessCritic(), CircularityCritic(), KnownFailureCritic()))
space = ResearchStateSpace(concepts, evaluator=evaluator)
seed = space.add(Conjecture(
    name="Two-frame SAT debt",
    statement="A shared circuit cannot cheaply discharge independent residual debt in two transverse frames.",
    definitions=("Debt counts unresolved pairs merged by an observer view.",),
    assumptions=("The residual-row map is injective.",),
    proof_sketch=("Construct transverse views.", "Prove a composition law.", "Integrate the per-gate charge."),
    tags=("SAT", "observer", "anti-sharing"),
))

result = CuriosityEngine(space, strategy="balanced").explore(seed, budget=20, neighbors_per_step=6)
print(result.summary())
for discovery in result.top(5):
    evaluation = discovery.state.metadata["evaluation"]
    conjecture = discovery.state.metadata["conjecture"]
    print(f"\n{conjecture.name}\n  verdict={evaluation.verdict}\n  {conjecture.statement}")

