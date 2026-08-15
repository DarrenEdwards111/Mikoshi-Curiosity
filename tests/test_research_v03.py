import json
import sys

from mikoshi_curiosity import (
    AssumptionAuditCritic,
    CallableResearchCritic,
    CallableTextProvider,
    Concept,
    Conjecture,
    Critique,
    FiniteModelFinder,
    FiniteModelProofAdapter,
    FiniteModelSpec,
    LeanRepairAdapter,
    LLMConjectureGenerator,
    LLMLeanRepairer,
    OllamaProvider,
    ProofResult,
    ResearchArchive,
    ResearchEvaluator,
)


def candidate():
    return Conjecture(
        "Fresh measure", "Every model satisfies the proposed invariant.",
        definitions=("The invariant is executable.",),
        assumptions=("The domain is finite.",),
        proof_sketch=("Search small models.",),
    )


def test_llm_generator_parses_typed_json_and_tracks_parent():
    provider = CallableTextProvider(lambda _: json.dumps({"conjectures": [{
        "name": "Generated lemma", "statement": "A implies B.",
        "definitions": ["A is finite."], "assumptions": ["A holds."],
        "proof_sketch": ["Count A."], "tags": ["counting"],
    }]}))
    seed = candidate()
    result = LLMConjectureGenerator(provider).generate(
        seed, [Concept("direct sum", "independent costs add")], 2
    )
    assert len(result) == 1
    assert result[0].parent_id == seed.id
    assert result[0].generation == seed.generation + 1
    assert "llm-generated" in result[0].provenance


def test_llm_generator_skips_malformed_siblings():
    provider = CallableTextProvider(lambda _: json.dumps({"conjectures": [
        {"name": "missing statement"},
        {"name": "bad typed list", "statement": "bad", "definitions": "not-a-list"},
        {"name": "valid", "statement": "A implies A.", "proof_sketch": ["identity"]},
    ]}))
    result = LLMConjectureGenerator(provider).generate(candidate(), (), 5)
    assert [item.name for item in result] == ["valid"]


def test_ollama_provider_uses_json_mode():
    seen = {}
    def transport(url, headers, payload, timeout):
        seen.update(payload)
        return {"response": '{"conjectures": []}'}
    provider = OllamaProvider(model="test", transport=transport)
    assert provider.complete("prompt").startswith("{")
    assert seen == {"model": "test", "prompt": "prompt", "stream": False, "format": "json"}


def test_archive_persists_evidence_and_retrieves_failures(tmp_path):
    rejected = candidate()
    evaluation = ResearchEvaluator((
        CallableResearchCritic("attack", lambda _: (Critique("attack", "error", "counterexample found"),)),
    )).evaluate(rejected)
    with ResearchArchive(tmp_path / "research.db") as archive:
        archive.save(rejected, evaluation)
        assert archive.count() == 1
        assert archive.count("rejected") == 1
        failures = archive.similar_failures("proposed invariant on finite models")
        assert failures[0]["name"] == rejected.name
        assert failures[0]["reason"] == "counterexample found"


def test_assumption_audit_rejects_configured_load_bearing_premise():
    critic = AssumptionAuditCritic({"cross-frame monogamy": "must be derived"})
    conjecture = Conjecture("circular", "claim", assumptions=("Assume cross-frame monogamy.",))
    issues = critic.review(conjecture)
    assert issues[0].fatal
    assert issues[0].message == "must be derived"


def test_finite_model_finder_returns_concrete_counterexample():
    spec = FiniteModelSpec({"x": (0, 1, 2)}, lambda model: model["x"] < 2)
    result = FiniteModelFinder().search(spec)
    assert result.status == "counterexample"
    assert result.counterexample == {"x": 2}


def test_finite_exhaustion_is_not_misreported_as_global_proof():
    adapter = FiniteModelProofAdapter(
        lambda _: FiniteModelSpec({"x": (0, 1)}, lambda model: model["x"] <= 1)
    )
    result = adapter.check(candidate())
    assert result.status == "bounded"
    assert not result.verified
    assert "not a proof" in result.output


def test_finite_model_adapter_falsifies_candidate():
    adapter = FiniteModelProofAdapter(
        lambda _: FiniteModelSpec({"x": (False, True)}, lambda model: not model["x"])
    )
    evaluation = ResearchEvaluator((), (adapter,)).evaluate(candidate())
    assert evaluation.verdict == "falsified"
    assert "counterexample" in evaluation.proofs[0].output


def test_iterative_repair_uses_diagnostics_until_checker_accepts():
    attempts = []
    def repair(conjecture, source, diagnostics, attempt):
        attempts.append((diagnostics, attempt))
        return "answer = 42\n"
    adapter = LeanRepairAdapter(
        (sys.executable, "-m", "py_compile", "{file}"),
        lambda _: "answer =\n", repair, max_repairs=2,
    )
    result = adapter.check(candidate())
    assert result.verified
    assert len(attempts) == 1
    assert "after 1 repairs" in result.output
    assert result.artifact == "answer = 42\n"


def test_llm_repairer_removes_markdown_fence():
    repairer = LLMLeanRepairer(CallableTextProvider(lambda _: "```lean\ntheorem ok : True := by trivial\n```"))
    result = repairer(candidate(), "bad", "error", 1)
    assert result.startswith("theorem ok")
    assert "```" not in result
