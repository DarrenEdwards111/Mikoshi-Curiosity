"""Persistent cognitive project runtime for Mikoshi Research OS.

This module deliberately separates cognition from conversation.  Projects retain goals,
beliefs, ideas, plans, evidence, decisions and reflections in SQLite.  A deliberation cycle
selects a useful next action, executes it through an injected tool, observes the result and
updates durable state.  No LLM provider is required; models are optional cognitive tools.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence


def _id(prefix: str) -> str:
    return f"{prefix}:{uuid.uuid4().hex}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class CognitiveRecord:
    id: str
    project_id: str
    kind: str
    title: str
    body: str
    status: str
    confidence: float
    priority: float
    parent_id: str
    metadata: Mapping[str, Any]
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ProposedAction:
    kind: str
    title: str
    rationale: str
    target_id: str = ""
    requires_approval: bool = False
    payload: Mapping[str, Any] = None

    def __post_init__(self):
        if self.payload is None:
            object.__setattr__(self, "payload", {})


@dataclass(frozen=True)
class CycleResult:
    cycle_id: str
    project_id: str
    action: ProposedAction
    status: str
    observation: str = ""


class CognitiveStore:
    """Transactional SQLite store for the evolving state of research projects."""

    VALID_KINDS = {
        "goal", "belief", "idea", "plan", "task", "evidence", "experiment",
        "decision", "reflection", "question",
    }

    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript("""
        CREATE TABLE IF NOT EXISTS cognitive_projects (
          id TEXT PRIMARY KEY, name TEXT NOT NULL, purpose TEXT NOT NULL,
          status TEXT NOT NULL DEFAULT 'active', created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS cognitive_records (
          id TEXT PRIMARY KEY, project_id TEXT NOT NULL, kind TEXT NOT NULL,
          title TEXT NOT NULL, body TEXT NOT NULL, status TEXT NOT NULL,
          confidence REAL NOT NULL, priority REAL NOT NULL, parent_id TEXT NOT NULL,
          metadata_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
          FOREIGN KEY(project_id) REFERENCES cognitive_projects(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS cognitive_cycles (
          id TEXT PRIMARY KEY, project_id TEXT NOT NULL, action_json TEXT NOT NULL,
          status TEXT NOT NULL, observation TEXT NOT NULL, created_at TEXT NOT NULL,
          completed_at TEXT NOT NULL DEFAULT '',
          FOREIGN KEY(project_id) REFERENCES cognitive_projects(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_cognitive_records_project_kind
          ON cognitive_records(project_id, kind, status);
        """)
        self.connection.commit()

    def create_project(self, name: str, purpose: str) -> str:
        project_id, now = _id("project"), _now()
        with self.connection:
            self.connection.execute(
                "INSERT INTO cognitive_projects VALUES(?,?,?,?,?,?)",
                (project_id, name, purpose, "active", now, now),
            )
        return project_id

    def delete_project(self, project_id: str) -> bool:
        """Delete one exact project and its cascaded cognitive history."""
        with self.connection:
            result = self.connection.execute(
                "DELETE FROM cognitive_projects WHERE id=?", (project_id,)
            )
        return result.rowcount > 0

    def add(self, project_id: str, kind: str, title: str, body: str = "", *,
            status: str = "open", confidence: float = 0.5, priority: float = 0.5,
            parent_id: str = "", metadata: Optional[Mapping[str, Any]] = None) -> CognitiveRecord:
        if kind not in self.VALID_KINDS:
            raise ValueError(f"unknown cognitive record kind: {kind}")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")
        record_id, now = _id(kind), _now()
        with self.connection:
            self.connection.execute(
                "INSERT INTO cognitive_records VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (record_id, project_id, kind, title, body, status, confidence, priority,
                 parent_id, json.dumps(dict(metadata or {}), sort_keys=True), now, now),
            )
        return self.get(record_id)

    def get(self, record_id: str) -> CognitiveRecord:
        row = self.connection.execute(
            "SELECT * FROM cognitive_records WHERE id=?", (record_id,)
        ).fetchone()
        if row is None:
            raise KeyError(record_id)
        return self._record(row)

    def list(self, project_id: str, *, kind: str = "", status: str = "") -> List[CognitiveRecord]:
        query, params = "SELECT * FROM cognitive_records WHERE project_id=?", [project_id]
        if kind:
            query, params = query + " AND kind=?", params + [kind]
        if status:
            query, params = query + " AND status=?", params + [status]
        query += " ORDER BY priority DESC, created_at ASC"
        return [self._record(row) for row in self.connection.execute(query, params)]

    def update(self, record_id: str, **changes: Any) -> CognitiveRecord:
        allowed = {"title", "body", "status", "confidence", "priority", "parent_id", "metadata"}
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"unsupported fields: {sorted(unknown)}")
        if "confidence" in changes and not 0.0 <= float(changes["confidence"]) <= 1.0:
            raise ValueError("confidence must be between zero and one")
        assignments, values = [], []
        for key, value in changes.items():
            column = "metadata_json" if key == "metadata" else key
            assignments.append(f"{column}=?")
            values.append(json.dumps(dict(value), sort_keys=True) if key == "metadata" else value)
        assignments.append("updated_at=?")
        values.extend((_now(), record_id))
        with self.connection:
            self.connection.execute(
                f"UPDATE cognitive_records SET {', '.join(assignments)} WHERE id=?", values
            )
        return self.get(record_id)

    def snapshot(self, project_id: str) -> Dict[str, List[CognitiveRecord]]:
        records = self.list(project_id)
        return {kind: [record for record in records if record.kind == kind]
                for kind in sorted(self.VALID_KINDS)}

    def save_cycle(self, project_id: str, action: ProposedAction, status: str,
                   observation: str = "", cycle_id: str = "") -> str:
        cycle_id = cycle_id or _id("cycle")
        now = _now()
        completed = now if status in {"completed", "failed", "blocked", "awaiting_approval"} else ""
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO cognitive_cycles VALUES(?,?,?,?,?,?,?)",
                (cycle_id, project_id, json.dumps(asdict(action), sort_keys=True), status,
                 observation, now, completed),
            )
        return cycle_id

    @staticmethod
    def _record(row: sqlite3.Row) -> CognitiveRecord:
        values = dict(row)
        values["metadata"] = json.loads(values.pop("metadata_json"))
        return CognitiveRecord(**values)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class ExecutivePlanner:
    """Transparent default policy for choosing the next useful cognitive action."""

    def choose(self, snapshot: Mapping[str, Sequence[CognitiveRecord]]) -> ProposedAction:
        goals = [x for x in snapshot["goal"] if x.status == "open"]
        questions = [x for x in snapshot["question"] if x.status == "open"]
        ideas = [x for x in snapshot["idea"] if x.status in {"open", "candidate"}]
        plans = [x for x in snapshot["plan"] if x.status == "approved"]
        tasks = [x for x in snapshot["task"] if x.status == "ready"]
        if tasks:
            target = tasks[0]
            return ProposedAction("execute_task", target.title,
                                  "Highest-priority approved task is ready.", target.id,
                                  bool(target.metadata.get("requires_approval")), target.metadata)
        if questions:
            target = questions[0]
            return ProposedAction("investigate", target.title,
                                  "An unresolved question blocks reliable planning.", target.id)
        if ideas:
            target = ideas[0]
            return ProposedAction("evaluate_idea", target.title,
                                  "Candidate idea needs evidence and adversarial evaluation.", target.id)
        if goals and not plans:
            target = goals[0]
            return ProposedAction("form_plan", target.title,
                                  "Active goal has no approved plan.", target.id)
        if plans:
            target = plans[0]
            return ProposedAction("decompose_plan", target.title,
                                  "Approved plan needs an executable next task.", target.id)
        return ProposedAction("reflect", "Review project state",
                              "No actionable item exists; consolidate learning and identify gaps.")


class CognitiveRuntime:
    """Runs bounded, auditable deliberation cycles over persistent project state."""

    def __init__(self, store: CognitiveStore, planner: Optional[ExecutivePlanner] = None,
                 tools: Optional[Mapping[str, Callable[[ProposedAction, Mapping[str, Any]], Any]]] = None):
        self.store = store
        self.planner = planner or ExecutivePlanner()
        self.tools = dict(tools or {})

    def initialise(self, name: str, purpose: str) -> str:
        project_id = self.store.create_project(name, purpose)
        self.store.add(project_id, "goal", name, purpose, confidence=1.0, priority=1.0)
        self.store.add(project_id, "question", "What evidence would distinguish success from failure?",
                       "Define observable acceptance and falsification criteria.", priority=0.9)
        return project_id

    def deliberate(self, project_id: str, context: Optional[Mapping[str, Any]] = None,
                   *, approved: bool = False) -> CycleResult:
        action = self.planner.choose(self.store.snapshot(project_id))
        cycle_id = self.store.save_cycle(project_id, action, "selected")
        if action.requires_approval and not approved:
            self.store.save_cycle(project_id, action, "awaiting_approval", cycle_id=cycle_id)
            return CycleResult(cycle_id, project_id, action, "awaiting_approval")
        tool = self.tools.get(action.kind)
        if tool is None:
            observation = f"No tool registered for {action.kind}; action retained for planning."
            self.store.save_cycle(project_id, action, "blocked", observation, cycle_id)
            return CycleResult(cycle_id, project_id, action, "blocked", observation)
        try:
            result = tool(action, dict(context or {}))
            observation = result if isinstance(result, str) else json.dumps(result, sort_keys=True)
            self._learn(project_id, action, observation)
            self.store.save_cycle(project_id, action, "completed", observation, cycle_id)
            return CycleResult(cycle_id, project_id, action, "completed", observation)
        except Exception as error:
            observation = f"{type(error).__name__}: {error}"
            self.store.add(project_id, "reflection", f"Failure during {action.title}", observation,
                           status="open", priority=0.8, parent_id=action.target_id)
            self.store.save_cycle(project_id, action, "failed", observation, cycle_id)
            return CycleResult(cycle_id, project_id, action, "failed", observation)

    def _learn(self, project_id: str, action: ProposedAction, observation: str) -> None:
        self.store.add(project_id, "evidence", f"Outcome: {action.title}", observation,
                       status="observed", confidence=0.7, parent_id=action.target_id,
                       metadata={"action_kind": action.kind})
        self.store.add(project_id, "reflection", f"Review: {action.title}",
                       "Compare the observed outcome with the expected result and revise the plan.",
                       status="open", priority=0.6, parent_id=action.target_id)
