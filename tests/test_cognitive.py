from mikoshi_curiosity.cognitive import CognitiveRuntime, CognitiveStore, ExecutivePlanner
from mikoshi_curiosity.nexus_bridge import handle


def test_project_state_persists_across_store_reopen(tmp_path):
    path = tmp_path / "mind.db"
    with CognitiveStore(path) as store:
        project = store.create_project("Reduce trial dropout", "Find a testable intervention")
        belief = store.add(project, "belief", "Reminder timing matters", confidence=0.4)
    with CognitiveStore(path) as store:
        assert store.get(belief.id).title == "Reminder timing matters"
        assert store.snapshot(project)["belief"][0].confidence == 0.4


def test_runtime_initialises_goal_and_success_question():
    with CognitiveStore() as store:
        runtime = CognitiveRuntime(store)
        project = runtime.initialise("Improve formulation", "Increase stability without toxicity")
        snapshot = store.snapshot(project)
        assert snapshot["goal"][0].body == "Increase stability without toxicity"
        assert snapshot["question"]


def test_deliberation_prefers_blocking_question_and_learns_from_result():
    with CognitiveStore() as store:
        runtime = CognitiveRuntime(
            store, ExecutivePlanner(),
            {"investigate": lambda action, context: f"Evidence from {context['source']}"},
        )
        project = runtime.initialise("Select target", "Choose the best supported target")
        result = runtime.deliberate(project, {"source": "replication study"})
        assert result.status == "completed"
        assert result.action.kind == "investigate"
        assert store.snapshot(project)["evidence"][0].body == "Evidence from replication study"
        assert store.snapshot(project)["reflection"]


def test_external_action_waits_for_human_approval():
    with CognitiveStore() as store:
        project = store.create_project("Pilot", "Run a safe pilot")
        store.add(project, "task", "Contact participant", status="ready", priority=1.0,
                  metadata={"requires_approval": True})
        called = []
        runtime = CognitiveRuntime(store, tools={"execute_task": lambda *_: called.append(True)})
        result = runtime.deliberate(project)
        assert result.status == "awaiting_approval"
        assert called == []


def test_failed_action_becomes_reflection_instead_of_being_forgotten():
    with CognitiveStore() as store:
        project = store.create_project("Audit", "Test a mechanism")
        store.add(project, "idea", "Mechanism A", status="candidate", priority=1.0)
        runtime = CognitiveRuntime(
            store, tools={"evaluate_idea": lambda *_: (_ for _ in ()).throw(ValueError("bad data"))}
        )
        result = runtime.deliberate(project)
        assert result.status == "failed"
        assert "bad data" in store.snapshot(project)["reflection"][0].body


def test_nexus_bridge_exposes_projects_without_bypassing_project_ownership():
    with CognitiveStore() as store:
        created = handle(store, {
            "action": "create_project", "name": "Nexus project", "purpose": "Plan through Nexus"
        })
        project = created["project_id"]
        added = handle(store, {
            "action": "add_record", "project_id": project, "kind": "belief",
            "title": "A testable belief", "confidence": 0.3,
        })
        snapshot = handle(store, {"action": "snapshot", "project_id": project})
        assert snapshot["snapshot"]["belief"][0]["title"] == "A testable belief"
        other = store.create_project("Other", "Must remain isolated")
        try:
            handle(store, {
                "action": "update_record", "project_id": other,
                "record_id": added["record"]["id"], "changes": {"confidence": 1.0},
            })
            assert False, "cross-project update should fail"
        except ValueError as error:
            assert "does not belong" in str(error)


def test_nexus_deliberation_runs_existing_research_lab_and_persists_ideas():
    with CognitiveStore() as store:
        created = handle(store, {
            "action": "create_project", "name": "Find intervention", "purpose": "Reduce failures"
        })
        result = handle(store, {
            "action": "deliberate", "project_id": created["project_id"], "context": {}
        })
        assert result["cycle"]["status"] == "completed"
        assert result["cycle"]["action"]["kind"] == "investigate"
        assert result["snapshot"]["idea"]
        assert result["snapshot"]["question"][0]["status"] == "investigated"
