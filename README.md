# Mikoshi Curiosity

<p align="center">
  <img src="https://raw.githubusercontent.com/DarrenEdwards111/Mikoshi-Curiosity/main/curiosity-logo.jpg" alt="Mikoshi Curiosity" width="400">
</p>

> **Explore any state space. Find what you didn't know you were looking for.**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-green.svg)](https://python.org)

---

**Recommendation engines predict what you'll like. Curiosity engines find what you don't know you'd like.**

Mikoshi Curiosity is a domain-agnostic exploration engine that uses intrinsic motivation, state memory, and diversity pressure to discover interesting things in *any* state space — datasets, text corpora, graphs, parameter spaces, or APIs.

Inspired by [Go-Explore](https://arxiv.org/abs/1901.10995) and intrinsic motivation research.

## Quick Start

```bash
pip install mikoshi-curiosity
# With extras:
pip install mikoshi-curiosity[data]  # pandas support
pip install mikoshi-curiosity[all]   # pandas + matplotlib
```

## Examples

### 🔍 Find Anomalies in a Dataset

```python
import pandas as pd
from mikoshi_curiosity import CuriosityEngine
from mikoshi_curiosity.contexts.dataset import DatasetSpace

df = pd.read_csv("sales_data.csv")
space = DatasetSpace(df)
engine = CuriosityEngine(space, strategy="novelty")

seed = space.get_state("0")
result = engine.explore(seed, budget=200)

print(result.summary())
for d in result.top(5):
    print(f"  [{d.score:.2f}] Row {d.state.id}: {d.reason}")
    print(f"         {d.state.features}")
```

### 📚 Explore a Text Corpus

```python
from mikoshi_curiosity import CuriosityEngine
from mikoshi_curiosity.contexts.text import TextSpace

docs = [
    {"id": "paper1", "text": "Deep learning for image recognition..."},
    {"id": "paper2", "text": "Quantum entanglement in biological systems..."},
    # ... hundreds of papers
]

space = TextSpace(docs)
engine = CuriosityEngine(space, strategy="diversity")
result = engine.explore(space.get_state("paper1"), budget=100)

# Find bridging documents that connect different topics
for d in result.top(10):
    print(f"  {d.state.id}: {d.reason}")
```

### 🎛️ Explore a Parameter Space

```python
from mikoshi_curiosity import CuriosityEngine
from mikoshi_curiosity.contexts.numeric import NumericSpace

def simulate(params):
    """Your simulation / evaluation function."""
    return complex_score(params["gravity"], params["friction"], params["elasticity"])

space = NumericSpace(
    dimensions={"gravity": (0, 20), "friction": (0, 1), "elasticity": (0.1, 5)},
    eval_fn=simulate,
)
engine = CuriosityEngine(space, strategy="balanced")
result = engine.explore(space.get_random(5), budget=500)

# Find interesting parameter combinations, phase transitions, sweet spots
for d in result.top(10):
    print(f"  Score: {d.score:.3f} | Params: {d.state.features}")
```

### 🕸️ Explore a Network

```python
from mikoshi_curiosity import CuriosityEngine
from mikoshi_curiosity.contexts.graph import GraphSpace

space = GraphSpace(
    nodes=["alice", "bob", "carol", ...],
    edges=[("alice", "bob"), ("bob", "carol"), ...],
)
engine = CuriosityEngine(space, strategy="novelty")
result = engine.explore(space.get_state("alice"), budget=100)

# Discover bridge nodes, structural holes, unexpected clusters
```

## Strategies

| Strategy | What it optimises | Best for |
|---|---|---|
| `novelty` | Distance from seen states | Finding outliers, anomalies |
| `surprise` | Prediction error | Finding rule-breakers |
| `diversity` | Distance from current discoveries | Broad coverage |
| `serendipity` | Novelty × relevance to profile | Personalised exploration |
| `balanced` | Weighted combination of all | General-purpose |

## API Reference

## 🧪 Research Lab: Generate, Attack, and Certify Conjectures

The research extension turns Curiosity from a closed parameter explorer into
an open-ended conjecture laboratory.  It provides:

- `Conjecture` — typed theorem intermediate representation;
- `ConceptGraph` — fuzzy semantic retrieval and explicit analogy edges;
- `ResearchStateSpace` — dynamic generation beyond a pre-enumerated grid;
- generator and mutator protocols for LLM or local proposal engines;
- critic plugins for circularity, completeness, and known counterexamples;
- proof-adapter plugins for Lean, SMT, test suites, or any local checker;
- an auditable `CandidateEvaluation` stored on every generated state.

```python
from mikoshi_curiosity import (
    Concept, ConceptGraph, Conjecture, CuriosityEngine, ResearchStateSpace,
)

graph = ConceptGraph([
    Concept("monogamy", "one resource cannot serve two independent views"),
    Concept("direct sum", "independent tasks require additive cost"),
    Concept("pebbling", "time-space tradeoffs on DAGs"),
])

space = ResearchStateSpace(graph)
seed = space.add(Conjecture(
    name="Anti-sharing lemma",
    statement="Independent residuals require fresh decision-sensitive cost.",
    definitions=("Residual debt is the number of unresolved row pairs.",),
    assumptions=("Residual rows are injective.",),
    proof_sketch=("Define the charge.", "Prove composition.", "Integrate."),
))

result = CuriosityEngine(space, strategy="balanced").explore(seed, budget=50)
for discovery in result.top(5):
    print(discovery.state.metadata["evaluation"])
```

`CommandProofAdapter` executes an argument vector without a shell and replaces
`{file}` with a temporary rendered candidate.  This supports Lean/SMT adapters
without introducing an API dependency into the core package.  An LLM-backed
generator can implement the `ConjectureGenerator.generate` protocol and be
injected into `ResearchStateSpace`. `CallableConjectureGenerator` and
`CallableResearchCritic` provide lightweight adapters for model APIs, local
inference servers, retrieval pipelines, and organization-specific evaluators.

See [`examples/research_lab.py`](examples/research_lab.py) for a complete
offline example.

### Research Lab v0.3

Version 0.3 adds the executable end-to-end components:

- `LLMConjectureGenerator` with Codex CLI, dependency-free Ollama, OpenAI, and
  Anthropic providers and strict typed-JSON validation;
- `ResearchArchive`, a persistent SQLite record of candidates, critiques,
  proof results, and nearby historical failures;
- `FiniteModelFinder` and `FiniteModelProofAdapter` for exhaustive bounded
  counterexample search (finite exhaustion is explicitly **not** called a
  global proof);
- `LeanRepairAdapter` plus `LLMLeanRepairer` for compile, diagnose, repair,
  and retry loops with configurable budgets;
- optional archive integration in `ResearchStateSpace`, allowing generated
  states and failed routes to survive across runs.

Run the live local-model INDEX-to-SAT workload with:

```bash
CODEX_MODEL=gpt-5.6-sol python examples/index_sat_research_v03.py
```

For the local fallback use `RESEARCH_PROVIDER=ollama` and optionally set
`OLLAMA_MODEL` and `OLLAMA_URL`.

The model proposes candidates; deterministic critics and model finders try to
break them; Lean or another command checker is the trust boundary. A survivor
is still a conjecture until an attached proof adapter verifies it.

### Fanout-neutral debt experiment

`mikoshi_curiosity.debt` implements residual-row debt as the logarithmic gap
between the number of rows and the equivalence classes induced by observed
boundary signatures. It separates two facts:

- **valid local inequality:** adding one Boolean gate can at most double the
  class count, and duplicating its output through fanout adds no distinctions;
- **invalid terminal-load inference:** a single decision bit does not, from
  correctness alone, distinguish every residual row.

Run the exhaustive small-model audit with:

```bash
python examples/debt_decomposition_experiment.py
```

### SAT residual-query orientations

`mikoshi_curiosity.sat_queries` turns every selector of a direct-sum INDEX row
into a concrete variable-free CNF.  The full family of exact SAT answers is
the row itself, so SAT correctness forces an injective external orientation
vector without assuming a circuit lower bound.  The accompanying finite-model
audit also finds the crucial boundary: one fixed query does not distinguish
the row once it contains more than one bit.

```bash
python examples/sat_orientation_experiment.py
```

This is a family-of-executions theorem.  It does not assert that one SAT run
materializes every answer, nor that a general circuit cannot amortize or share
work across the family.

Version 0.6 also includes a genuinely local split construction. Alice emits
only unit clauses fixing the INDEX data variables; Bob emits only unit clauses
fixing a one-hot selector; fixed public clauses enforce `selector -> data`.
Neither private encoder computes the selected answer. The bounded audit proves
satisfiability equals that answer. It also records the quantitative limitation:
`N` forced orientation bits require `3N` clauses, so this embedding alone gives
constant debt density rather than superpolynomial amplification.

### Amplification obstruction audit

`mikoshi_curiosity.amplification` constructs the explicit shared DAG for the
slice predicate: `N` parallel `data_j AND selector_j` gates followed by an OR
tree.  It uses exactly `2N - 1` bounded-fanin gates and is exhaustively checked
on every row and legal selector through the configured finite bound.  Thus the
split INDEX family has a matching linear general-circuit upper bound and cannot
be the compression-resistant family needed for a P-vs-NP proof.

```bash
PYTHONPATH=. python3 examples/amplification_audit.py
```

### Succinct-tableau repeated-recovery audit

`mikoshi_curiosity.tableau` tests the proposal that a narrow reusable
transition computation can force many semantic recoveries. A ripple-counter
gadget visits `2^w` states with `O(w)` reusable hardware and a `w`-bit
frontier, but an ordinary acyclic circuit or Cook--Levin CNF must unroll all
steps, costing `Omega(w * steps)`. Exponential recovery therefore either uses
exponential time/tableau size or leaves the ordinary SAT/P-time setting.

```bash
PYTHONPATH=. python3 examples/tableau_recovery_experiment.py
```

### Gödel solver-capture audit

`mikoshi_curiosity.solver_capture` evaluates the strongest finite-testable
candidate from the Gödel-tower research run: rational rank of the semantic
prefix/continuation relation. The measure is invariant under renamed or
duplicated gates. Exact finite arithmetic confirms that a genuine independent
Kronecker product squares rank and supplies the desired load recurrence, while
copying or re-encoding the old relation does not. Product capture is therefore
the explicit missing theorem; it cannot be inferred from one-bit correctness.

```bash
PYTHONPATH=. python3 examples/solver_capture_experiment.py
```

### Cognitive Project Runtime

`CognitiveRuntime` turns Curiosity's research tools into a persistent project agent rather than a
prompt/response wrapper. It stores goals, beliefs, ideas, plans, tasks, evidence, experiments,
decisions, questions and reflections in SQLite, selects the next useful action from that state,
requires approval for external actions, and records outcomes for later cycles.

```python
from mikoshi_curiosity import CognitiveRuntime, CognitiveStore

store = CognitiveStore("research-os.db")
runtime = CognitiveRuntime(store, tools={
    "investigate": lambda action, context: search_evidence(action.title),
    "evaluate_idea": lambda action, context: challenge_idea(action.target_id),
})

project = runtime.initialise(
    "Improve trial retention",
    "Find and validate an intervention that reduces avoidable dropout",
)
result = runtime.deliberate(project)
print(result.action, result.observation)
```

The executive policy is intentionally transparent and replaceable. Model-backed planners may be
injected later, while durable state, bounded cycles and human approval remain enforced outside the
model.

Research Lab tasks may route to `verify_with_lean`, `find_countermodel`, `run_simulation`,
`search_literature`, or `delegate_specialist` through their metadata. Model-backed generation and
delegation are opt-in: set `MIKOSHI_MODEL_PROVIDER` to `openai`, `anthropic`, `ollama`, or `codex`
and provide that provider's normal model/credential variables. Lean verification uses
`MIKOSHI_LEAN_COMMAND` (default: `lean {file}`). Nexus exposes live availability so missing tools
are visible rather than silently simulated.

### Core

- **`State`** — A point in exploration space (id, features, embedding, metadata)
- **`StateSpace`** — Abstract space to explore (subclass for your domain)
- **`CuriosityEngine`** — The exploration engine
- **`ExplorationMemory`** — Go-Explore style state archive
- **`PredictionModel`** — Online model for surprise detection
- **`Discovery`** — A single interesting finding with score and reason
- **`ExplorationResult`** — Container with discoveries, stats, and memory

### Built-in Contexts

| Context | Module | Input |
|---|---|---|
| Tabular data | `contexts.dataset.DatasetSpace` | DataFrame or CSV path |
| Text corpus | `contexts.text.TextSpace` | List of `{id, text}` dicts |
| Graph/network | `contexts.graph.GraphSpace` | Nodes + edges |
| Numeric/params | `contexts.numeric.NumericSpace` | Dimension bounds + eval function |
| External API | `contexts.api.APISpace` | Fetch function |

### Custom State Space

```python
from mikoshi_curiosity import StateSpace, State

class MySpace(StateSpace):
    def get_neighbors(self, state, n=10):
        ...  # Return nearby states

    def get_random(self, n=10):
        ...  # Return random states

    def get_state(self, id):
        ...  # Lookup by id

    def embed(self, state):
        ...  # Return numpy vector

    def size(self):
        ...  # Approximate space size
```

## Visualization

```python
from mikoshi_curiosity.viz import plot_exploration, plot_discovery_scores

fig = plot_exploration(result)   # 2D PCA projection with discoveries highlighted
fig = plot_discovery_scores(result)  # Score breakdown bar chart
```

Requires `pip install mikoshi-curiosity[viz]`.

## Design Philosophy

- **Zero external API dependencies** — core uses only NumPy
- **Domain-agnostic** — works with any state space you define
- **Go-Explore inspired** — maintains an archive of interesting states and explores from frontiers
- **Multiple signals** — novelty, surprise, diversity, serendipity, diminishing returns
- **Online learning** — prediction model updates as you explore
- **Resumable** — ExplorationResult contains full memory state

---

Built by **Mikoshi Ltd**
