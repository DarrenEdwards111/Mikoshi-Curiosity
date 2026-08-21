"""Research Lab tools for persistent cognitive projects."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import urllib.request
from typing import Dict

from mikoshi_curiosity.cognitive import CognitiveStore, ProposedAction
from mikoshi_curiosity.engine import CuriosityEngine
from mikoshi_curiosity.llm import (
    AnthropicProvider, CodexCLIProvider, LLMConjectureGenerator, OllamaProvider, OpenAIProvider,
)
from mikoshi_curiosity.model_finder import FiniteModelFinder, FiniteModelSpec
from mikoshi_curiosity.research import (
    CircularityCritic, CommandProofAdapter, CompletenessCritic, Concept, ConceptGraph, Conjecture,
    KnownFailureCritic, ResearchEvaluator, ResearchStateSpace,
)


class ResearchLabToolRegistry:
    """Bind the existing Research Lab engines to one persistent cognitive project."""

    def __init__(self, store: CognitiveStore, project_id: str, *, exploration_budget: int = 8):
        self.store = store
        self.project_id = project_id
        self.exploration_budget = exploration_budget
        self.provider = self._provider_from_environment()

    @staticmethod
    def _provider_from_environment():
        selected = os.getenv("MIKOSHI_MODEL_PROVIDER", "").lower()
        if selected == "openai" and os.getenv("OPENAI_API_KEY"):
            return OpenAIProvider(os.environ["OPENAI_API_KEY"], os.getenv("OPENAI_MODEL", "gpt-5-mini"))
        if selected == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
            return AnthropicProvider(os.environ["ANTHROPIC_API_KEY"], os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"))
        if selected == "ollama" and os.getenv("OLLAMA_MODEL"):
            return OllamaProvider(os.environ["OLLAMA_MODEL"], os.getenv("OLLAMA_URL", "http://localhost:11434"))
        if selected == "codex" and shutil.which(os.getenv("CODEX_EXECUTABLE", "codex")):
            return CodexCLIProvider(os.getenv("CODEX_MODEL", "gpt-5.6-sol"), os.getenv("CODEX_EXECUTABLE", "codex"))
        return None

    def capabilities(self):
        lean_command = os.getenv("MIKOSHI_LEAN_COMMAND", "lean {file}")
        lean_executable = shlex.split(lean_command)[0] if lean_command else ""
        return {
            "generate_with_model": {"available": self.provider is not None},
            "delegate_specialist": {"available": self.provider is not None},
            "verify_with_lean": {"available": bool(lean_executable and shutil.which(lean_executable)),
                                 "command": lean_command},
            "find_countermodel": {"available": True, "format": "finite domains + forbidden assignments"},
            "run_simulation": {"available": True, "format": "structured scenarios"},
            "search_literature": {"available": True, "format": "explicit http(s) source URLs"},
        }

    def tools(self) -> Dict[str, object]:
        return {
            "investigate": self.investigate,
            "evaluate_idea": self.evaluate_idea,
            "form_plan": self.form_plan,
            "approve_plan": self.approve_plan,
            "decompose_plan": self.decompose_plan,
            "reflect": self.reflect,
            "execute_task": self.execute_task,
            "verify_with_lean": self.verify_with_lean,
            "find_countermodel": self.find_countermodel,
            "run_simulation": self.run_simulation,
            "search_literature": self.search_literature,
            "delegate_specialist": self.delegate_specialist,
        }

    def investigate(self, action: ProposedAction, context):
        target = self.store.get(action.target_id)
        graph = ConceptGraph([
            Concept("mechanism", "Explain the causal mechanism and its boundary conditions"),
            Concept("counterexample", "Search for a case where the proposed route fails"),
            Concept("measurement", "Define observable success and falsification criteria"),
            Concept("implementation", "Respect cost, time, dependencies and operational constraints"),
        ])
        evaluator = ResearchEvaluator((CompletenessCritic(), CircularityCritic(), KnownFailureCritic()))
        generator = LLMConjectureGenerator(self.provider) if self.provider is not None else None
        space = ResearchStateSpace(graph, generator=generator, evaluator=evaluator)
        seed = space.add(Conjecture(
            name=target.title,
            statement=target.body or target.title,
            definitions=("The project outcome must be observable and testable.",),
            assumptions=tuple(context.get("assumptions", ())),
            proof_sketch=("Generate competing mechanisms.", "Attack assumptions.", "Rank decisive tests."),
            tags=("cognitive-project",),
        ))
        result = CuriosityEngine(space, strategy="balanced").explore(
            seed, budget=self.exploration_budget, neighbors_per_step=4
        )
        discoveries = result.top(4)
        for discovery in discoveries:
            conjecture = discovery.state.metadata["conjecture"]
            evaluation = discovery.state.metadata["evaluation"]
            self.store.add(
                self.project_id, "idea", conjecture.name, conjecture.statement,
                status="candidate", confidence=max(0.0, min(1.0, evaluation.score)),
                priority=discovery.score, parent_id=target.id,
                metadata={
                    "assumptions": list(conjecture.assumptions),
                    "proof_sketch": list(conjecture.proof_sketch),
                    "verdict": evaluation.verdict,
                    "critiques": [critique.message for critique in evaluation.critiques],
                },
            )
        self.store.update(target.id, status="investigated")
        return json.dumps({
            "engine": "Mikoshi Research Lab", "candidates": len(discoveries),
            "states_scored": result.stats.states_scored,
        }, sort_keys=True)

    def evaluate_idea(self, action: ProposedAction, _context):
        idea = self.store.get(action.target_id)
        candidate = Conjecture(
            name=idea.title, statement=idea.body,
            assumptions=tuple(idea.metadata.get("assumptions", ())),
            proof_sketch=tuple(idea.metadata.get("proof_sketch", ())),
        )
        evaluation = ResearchEvaluator(
            (CompletenessCritic(), CircularityCritic(), KnownFailureCritic())
        ).evaluate(candidate)
        status = "rejected" if evaluation.verdict in {"rejected", "falsified"} else "evaluated"
        self.store.update(idea.id, status=status, confidence=max(0.0, min(1.0, evaluation.score)))
        self.store.add(
            self.project_id, "evidence", f"Evaluation: {idea.title}",
            "\n".join(critique.message for critique in evaluation.critiques) or "No built-in critic rejected the idea.",
            status="observed", confidence=0.7, parent_id=idea.id,
            metadata={"verdict": evaluation.verdict, "score": evaluation.score},
        )
        return json.dumps({"verdict": evaluation.verdict, "score": evaluation.score}, sort_keys=True)

    def form_plan(self, action: ProposedAction, context):
        goal = self.store.get(action.target_id)
        plan = self.store.add(
            self.project_id, "plan", f"Plan: {goal.title}",
            "Define success; compare candidate mechanisms; run the cheapest decisive test; review evidence; re-plan.",
            status="approved" if context.get("auto_approve_plan") else "proposed",
            confidence=0.6, priority=goal.priority, parent_id=goal.id,
            metadata={"prediction": context.get("prediction", "The selected test will reduce decision uncertainty."),
                      "success_criteria": context.get("success_criteria", "A result distinguishes leading candidates.")},
        )
        return json.dumps({"plan_id": plan.id, "status": plan.status}, sort_keys=True)

    def decompose_plan(self, action: ProposedAction, _context):
        plan = self.store.get(action.target_id)
        task = self.store.add(
            self.project_id, "task", f"First decisive test for {plan.title}",
            plan.metadata.get("success_criteria", "Collect evidence that distinguishes the leading options."),
            status="ready", confidence=plan.confidence, priority=plan.priority,
            parent_id=plan.id, metadata={"requires_approval": True},
        )
        self.store.update(plan.id, status="active")
        return json.dumps({"task_id": task.id, "requires_approval": True}, sort_keys=True)

    def approve_plan(self, action: ProposedAction, _context):
        plan = self.store.update(action.target_id, status="approved")
        return json.dumps({"plan_id": plan.id, "status": plan.status}, sort_keys=True)

    def execute_task(self, action: ProposedAction, context):
        task = self.store.get(action.target_id)
        result = context.get("task_result")
        if not result:
            raise ValueError("approved task requires a task_result or an external executor")
        self.store.update(task.id, status="completed")
        return str(result)

    def verify_with_lean(self, action: ProposedAction, _context):
        task = self.store.get(action.target_id)
        source = str(task.metadata.get("lean_source", ""))
        if not source:
            raise ValueError("Lean verification task requires lean_source")
        command = shlex.split(os.getenv("MIKOSHI_LEAN_COMMAND", "lean {file}"))
        if not command or not shutil.which(command[0]):
            raise RuntimeError(f"Lean executable is unavailable: {command[0] if command else 'unset'}")
        conjecture = Conjecture(task.title, task.body or task.title)
        result = CommandProofAdapter(
            "lean", command, lambda _: source, suffix=".lean", timeout=120.0
        ).check(conjecture)
        self.store.update(task.id, status="completed" if result.verified else "failed")
        self.store.add(self.project_id, "evidence", f"Lean: {task.title}", result.output,
                       status=result.status, confidence=1.0 if result.verified else 0.9,
                       parent_id=task.id, metadata={"adapter": result.adapter, "artifact": result.artifact})
        return json.dumps({"adapter": result.adapter, "status": result.status}, sort_keys=True)

    def find_countermodel(self, action: ProposedAction, _context):
        task = self.store.get(action.target_id)
        domains = task.metadata.get("domains")
        forbidden = task.metadata.get("forbidden_assignments", [])
        if not isinstance(domains, dict) or not domains:
            raise ValueError("countermodel task requires non-empty domains")
        if not isinstance(forbidden, list):
            raise ValueError("forbidden_assignments must be a list")
        def holds(assignment):
            return not any(all(assignment.get(key) == value for key, value in row.items())
                           for row in forbidden if isinstance(row, dict))
        result = FiniteModelFinder(int(task.metadata.get("max_models", 100000))).search(
            FiniteModelSpec(domains, holds, task.body)
        )
        status = "falsified" if result.status == "counterexample" else "bounded"
        body = json.dumps({"status": result.status, "checked": result.checked,
                           "counterexample": result.counterexample, "reason": result.reason}, sort_keys=True)
        self.store.update(task.id, status="completed")
        self.store.add(self.project_id, "evidence", f"Finite model: {task.title}", body,
                       status=status, confidence=0.95, parent_id=task.id)
        return body

    def run_simulation(self, action: ProposedAction, _context):
        task = self.store.get(action.target_id)
        scenarios = task.metadata.get("scenarios")
        if not isinstance(scenarios, list) or not scenarios:
            raise ValueError("simulation task requires scenarios")
        rows, expected = [], 0.0
        for scenario in scenarios:
            probability = float(scenario.get("probability", 0.0))
            value = float(scenario.get("value", 0.0))
            cost = float(scenario.get("cost", 0.0))
            utility = value - cost
            expected += probability * utility
            rows.append({"name": str(scenario.get("name", "scenario")), "utility": utility,
                         "weighted_utility": probability * utility})
        body = json.dumps({"expected_utility": expected, "scenarios": rows}, sort_keys=True)
        self.store.update(task.id, status="completed")
        self.store.add(self.project_id, "evidence", f"Simulation: {task.title}", body,
                       status="observed", confidence=0.7, parent_id=task.id)
        return body

    def search_literature(self, action: ProposedAction, _context):
        task = self.store.get(action.target_id)
        urls = task.metadata.get("urls")
        if not isinstance(urls, list) or not urls:
            raise ValueError("literature task requires explicit source urls")
        sources = []
        for url in urls[:10]:
            if not isinstance(url, str) or not re.match(r"^https?://", url):
                raise ValueError("literature source must be an http(s) URL")
            request = urllib.request.Request(url, headers={"User-Agent": "MikoshiResearchLab/0.10"})
            with urllib.request.urlopen(request, timeout=20.0) as response:
                text = response.read(250000).decode("utf-8", errors="replace")
            title = re.search(r"<title[^>]*>(.*?)</title>", text, re.I | re.S)
            sources.append({"url": url, "title": re.sub(r"\s+", " ", title.group(1)).strip() if title else "",
                            "bytes_reviewed": len(text.encode("utf-8"))})
        body = json.dumps({"sources": sources, "count": len(sources)}, sort_keys=True)
        self.store.update(task.id, status="completed")
        self.store.add(self.project_id, "evidence", f"Literature: {task.title}", body,
                       status="observed", confidence=0.65, parent_id=task.id,
                       metadata={"provenance": [source["url"] for source in sources]})
        return body

    def delegate_specialist(self, action: ProposedAction, context):
        if self.provider is None:
            raise RuntimeError("no specialist model provider is configured")
        task = self.store.get(action.target_id)
        role = str(task.metadata.get("specialist_role", "research specialist"))
        response = self.provider.complete(
            f"Act as a {role}. Analyze this task, state assumptions, alternatives, evidence needed, "
            f"failure modes and a recommended next test. Do not claim unavailable evidence.\n"
            f"Task: {task.title}\n{task.body}\nContext: {json.dumps(context, sort_keys=True)}"
        )
        self.store.update(task.id, status="completed")
        self.store.add(self.project_id, "evidence", f"Specialist: {task.title}", response,
                       status="advisory", confidence=0.55, parent_id=task.id,
                       metadata={"role": role, "requires_independent_verification": True})
        return response

    def reflect(self, _action: ProposedAction, _context):
        reflection = self.store.add(
            self.project_id, "reflection", "Metacognitive review",
            "No immediate action was available. Review contradictions, stale assumptions and missing evidence.",
            status="open", priority=0.6,
        )
        return json.dumps({"reflection_id": reflection.id}, sort_keys=True)
