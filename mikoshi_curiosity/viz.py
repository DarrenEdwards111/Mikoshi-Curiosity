"""Visualization helpers for exploration results."""

from __future__ import annotations

from typing import Optional

import numpy as np

from mikoshi_curiosity.results import ExplorationResult


def plot_exploration(result: ExplorationResult, title: str = "Exploration Map"):
    """2D PCA projection of explored space with discoveries highlighted."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    if result.memory is None or len(result.memory) == 0:
        raise ValueError("No exploration data to plot")

    entries = result.memory.get_archive()
    embs = [e.state.embedding for e in entries if e.state.embedding is not None]
    if len(embs) < 2:
        raise ValueError("Need at least 2 states with embeddings")

    X = np.array(embs)
    # Simple PCA to 2D
    mean = X.mean(axis=0)
    Xc = X - mean
    cov = Xc.T @ Xc / len(Xc)
    eigvals, eigvecs = np.linalg.eigh(cov)
    top2 = eigvecs[:, -2:][:, ::-1]
    coords = Xc @ top2

    scores = np.array([e.score for e in entries if e.state.embedding is not None])

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=scores, cmap="viridis", alpha=0.6, s=20)
    plt.colorbar(sc, ax=ax, label="Score")

    # Highlight discoveries
    disc_ids = {d.state.id for d in result.discoveries[:10]}
    for i, entry in enumerate(e for e in entries if e.state.embedding is not None):
        if entry.state.id in disc_ids:
            ax.scatter(coords[i, 0], coords[i, 1], c="red", s=100, marker="*", zorder=5)

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    return fig


def plot_discovery_scores(result: ExplorationResult, top_n: int = 10):
    """Bar chart of top discoveries with score breakdown."""
    import matplotlib.pyplot as plt

    discoveries = result.top(top_n)
    if not discoveries:
        raise ValueError("No discoveries to plot")

    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [d.state.id for d in discoveries]
    strategies = set()
    for d in discoveries:
        strategies.update(d.strategy_scores.keys())
    strategies = sorted(strategies)

    x = np.arange(len(labels))
    width = 0.8 / max(len(strategies), 1)

    for i, strat in enumerate(strategies):
        vals = [d.strategy_scores.get(strat, 0) for d in discoveries]
        ax.bar(x + i * width, vals, width, label=strat)

    ax.set_xticks(x + width * len(strategies) / 2)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Discovery Score Breakdown")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_coverage(result: ExplorationResult):
    """Coverage over exploration steps (requires memory)."""
    import matplotlib.pyplot as plt

    if result.memory is None:
        raise ValueError("No memory data")

    entries = result.memory.get_archive()
    # Sort by visit count as proxy for order
    counts = list(range(1, len(entries) + 1))
    cumulative = np.arange(1, len(entries) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(counts, cumulative, linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("States Explored")
    ax.set_title("Exploration Coverage")
    ax.grid(True, alpha=0.3)
    return fig
