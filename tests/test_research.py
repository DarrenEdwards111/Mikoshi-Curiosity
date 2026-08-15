import sys

import numpy as np

from mikoshi_curiosity import (
    CallableConjectureGenerator,
    CallableProofAdapter,
    CallableResearchCritic,
    CircularityCritic,
    CommandProofAdapter,
    CompletenessCritic,
    Concept,
    ConceptGraph,
    Conjecture,
    CuriosityEngine,
    HashEmbedder,
    ProofResult,
    ResearchEvaluator,
    ResearchStateSpace,
)


def seed_conjecture():
    return Conjecture(
        name="Boundary capacity lemma",
        statement="Independent residual rows require distinct observer states.",
        definitions=("Observer state is a finite boundary label.",),
        assumptions=("Residual rows are injective.",),
        proof_sketch=("Pull back the observer map and count its image.",),
        tags=("communication", "boundary"),
    )


def test_hash_embedder_is_deterministic_and_normalized():
    embedder = HashEmbedder(32)
    left = embedder("semantic boundary rank")
    right = embedder("semantic boundary rank")
    assert np.array_equal(left, right)
    assert np.isclose(np.linalg.norm(left), 1.0)


def test_concept_graph_fuzzy_and_explicit_neighbors():
    graph = ConceptGraph([
        Concept("monogamy", "one resource cannot be shared freely", ("quantum",)),
        Concept("direct sum", "independent tasks require additive cost", ("communication",)),
        Concept("gardening", "growing vegetables", ("plants",)),
    ])
    graph.connect("monogamy", "direct sum", 4.0)
    names = [concept.name for concept, _ in graph.neighbors("monogamy", n=2)]
    assert names[0] == "direct sum"


def test_circularity_critic_rejects_target_assumption():
    candidate = Conjecture("bad", "P differs from NP", assumptions=("Assume SAT ∉ P.",))
    issues = CircularityCritic().review(candidate)
    assert issues and issues[0].fatal


def test_research_evaluator_verifies_survivor():
    adapter = CallableProofAdapter("toy", lambda _: ProofResult("wrong-name", "verified", "ok"))
    evaluation = ResearchEvaluator((CompletenessCritic(),), (adapter,)).evaluate(seed_conjecture())
    assert evaluation.verdict == "verified"
    assert evaluation.proofs[0].adapter == "toy"


def test_command_proof_adapter_uses_no_shell_and_checks_source():
    adapter = CommandProofAdapter(
        "python",
        (sys.executable, "-m", "py_compile", "{file}"),
        lambda _: "answer = 42\n",
        suffix=".py",
    )
    result = adapter.check(seed_conjecture())
    assert result.verified


def test_callable_generator_and_critic_are_injectable():
    generator = CallableConjectureGenerator(
        lambda seed, concepts, n: [
            Conjecture("generated", f"Use {concepts[0].name}: {seed.statement}")
        ][:n]
    )
    critic = CallableResearchCritic("external", lambda _: ())
    graph = ConceptGraph([Concept("analogy", "transfer a mechanism")])
    space = ResearchStateSpace(
        graph,
        generator=generator,
        evaluator=ResearchEvaluator((critic,)),
    )

    neighbors = space.get_neighbors(space.add(seed_conjecture()), n=1)

    assert neighbors[0].features["name"] == "generated"


def test_research_space_generates_open_ended_neighbors_and_engine_runs():
    graph = ConceptGraph([
        Concept("monogamy", "shared capacity cannot serve transverse views", ("rank",)),
        Concept("sheaf obstruction", "local views have no compatible global section", ("topology",)),
        Concept("pebbling", "space time tradeoffs on directed graphs", ("complexity",)),
    ])
    space = ResearchStateSpace(graph)
    seed = space.add(seed_conjecture())
    neighbors = space.get_neighbors(seed, n=4)
    assert neighbors
    assert all(state.metadata["conjecture"].parent_id == seed.id for state in neighbors)
    result = CuriosityEngine(space, strategy="balanced").explore(seed, budget=3, neighbors_per_step=3)
    assert result.stats.states_scored > 0
    assert space.size() > 1
