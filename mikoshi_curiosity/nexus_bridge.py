"""Small JSON bridge exposing the Cognitive Runtime to Mikoshi Nexus.

The bridge keeps Python cognition authoritative while allowing the Node portal to own users,
authentication and project access. Requests arrive as one JSON object on stdin; one JSON object is
written to stdout so the process is safe to call from an HTTP adapter.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from mikoshi_curiosity.cognitive import CognitiveRuntime, CognitiveStore
from mikoshi_curiosity.research_tools import ResearchLabToolRegistry


def _serialise_snapshot(snapshot):
    return {kind: [asdict(record) for record in records] for kind, records in snapshot.items()}


def handle(store: CognitiveStore, request):
    action = request.get("action")
    if action == "create_project":
        project_id = CognitiveRuntime(store).initialise(request["name"], request["purpose"])
        return {"project_id": project_id, "snapshot": _serialise_snapshot(store.snapshot(project_id))}
    if action == "snapshot":
        return {"project_id": request["project_id"],
                "snapshot": _serialise_snapshot(store.snapshot(request["project_id"]))}
    if action == "delete_project":
        return {"deleted": store.delete_project(request["project_id"])}
    if action == "capabilities":
        registry = ResearchLabToolRegistry(store, request["project_id"])
        return {"capabilities": registry.capabilities()}
    if action == "add_record":
        record = store.add(
            request["project_id"], request["kind"], request["title"], request.get("body", ""),
            status=request.get("status", "open"), confidence=float(request.get("confidence", 0.5)),
            priority=float(request.get("priority", 0.5)), parent_id=request.get("parent_id", ""),
            metadata=request.get("metadata", {}),
        )
        return {"record": asdict(record)}
    if action == "update_record":
        existing = store.get(request["record_id"])
        if existing.project_id != request["project_id"]:
            raise ValueError("record does not belong to project")
        record = store.update(request["record_id"], **request.get("changes", {}))
        return {"record": asdict(record)}
    if action == "deliberate":
        tools = ResearchLabToolRegistry(store, request["project_id"]).tools()
        result = CognitiveRuntime(store, tools=tools).deliberate(
            request["project_id"], request.get("context", {}), approved=bool(request.get("approved"))
        )
        return {"cycle": asdict(result),
                "snapshot": _serialise_snapshot(store.snapshot(request["project_id"]))}
    if action == "run_program":
        tools = ResearchLabToolRegistry(store, request["project_id"]).tools()
        cycles = CognitiveRuntime(store, tools=tools).run_program(
            request["project_id"], request.get("context", {}),
            max_cycles=int(request.get("max_cycles", 12)), approved=bool(request.get("approved")),
        )
        return {"cycles": [asdict(cycle) for cycle in cycles],
                "snapshot": _serialise_snapshot(store.snapshot(request["project_id"]))}
    raise ValueError(f"unsupported bridge action: {action}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    args = parser.parse_args(argv)
    try:
        request = json.load(sys.stdin)
        with CognitiveStore(args.db) as store:
            response = handle(store, request)
        json.dump({"ok": True, **response}, sys.stdout)
    except (KeyError, ValueError, json.JSONDecodeError) as error:
        json.dump({"ok": False, "error": str(error)}, sys.stdout)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
