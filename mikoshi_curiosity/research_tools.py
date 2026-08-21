"""Research Lab tools for persistent cognitive projects."""

from __future__ import annotations

import json
from typing import Dict

from mikoshi_curiosity.cognitive import CognitiveStore, ProposedAction
from mikoshi_curiosity.engine import CuriosityEngine
from mikoshi_curiosity.research import (
    CircularityCritic, CompletenessCritic, Concept, ConceptGraph, Conjecture,
    KnownFailureCritic, ResearchEvaluator, ResearchStateSpace,
)


class ResearchLabToolRegistry:
    """Bind the existing Research Lab engines to one persistent cognitive project."""

    def __init__(self, store: CognitiveStore, project_id: str, *, exploration_budget: int = 8):
        self.store = store
        self.project_id = project_id
        self.exploration_budget = exploration_budget

    def tools(self) -> Dict[str, object]:
        return {
            "investigate": self.investigate,
            "evaluate_idea": self.evaluate_idea,
            "form_plan": self.form_plan,
            "approve_plan": self.approve_plan,
            "decompose_plan": self.decompose_plan,
            "reflect": self.reflect,
            "execute_task": self.execute_task,
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
        space = ResearchStateSpace(graph, evaluator=evaluator)
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

    def reflect(self, _action: ProposedAction, _context):
        reflection = self.store.add(
            self.project_id, "reflection", "Metacognitive review",
            "No immediate action was available. Review contradictions, stale assumptions and missing evidence.",
            status="open", priority=0.6,
        )
        return json.dumps({"reflection_id": reflection.id}, sort_keys=True)
