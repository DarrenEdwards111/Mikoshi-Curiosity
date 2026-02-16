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
