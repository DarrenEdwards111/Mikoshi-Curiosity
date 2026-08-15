"""Mikoshi Curiosity — domain-agnostic exploration engine."""

__version__ = "0.2.0"

from mikoshi_curiosity.space import State, StateSpace
from mikoshi_curiosity.engine import CuriosityEngine
from mikoshi_curiosity.memory import ExplorationMemory, MemoryEntry
from mikoshi_curiosity.prediction import PredictionModel
from mikoshi_curiosity.results import Discovery, ExplorationResult, ExplorationStats
from mikoshi_curiosity.research import (
    AssumptionChallengeMutator,
    CallableConjectureGenerator,
    CallableProofAdapter,
    CallableResearchCritic,
    CandidateEvaluation,
    CircularityCritic,
    CommandProofAdapter,
    CompletenessCritic,
    Concept,
    ConceptGraph,
    Conjecture,
    ConjectureGenerator,
    ConjectureMutator,
    Critique,
    FuzzyAnalogyMutator,
    HashEmbedder,
    KnownFailureCritic,
    ProofAdapter,
    ProofResult,
    ResearchCritic,
    ResearchEvaluator,
    ResearchStateSpace,
    TemplateGenerator,
)
from mikoshi_curiosity.scoring import (
    novelty_score,
    surprise_score,
    diversity_score,
    serendipity_score,
    diminishing_returns,
    balanced_score,
)

__all__ = [
    "State",
    "StateSpace",
    "CuriosityEngine",
    "ExplorationMemory",
    "MemoryEntry",
    "PredictionModel",
    "Discovery",
    "ExplorationResult",
    "ExplorationStats",
    "AssumptionChallengeMutator",
    "CallableConjectureGenerator",
    "CallableProofAdapter",
    "CallableResearchCritic",
    "CandidateEvaluation",
    "CircularityCritic",
    "CommandProofAdapter",
    "CompletenessCritic",
    "Concept",
    "ConceptGraph",
    "Conjecture",
    "ConjectureGenerator",
    "ConjectureMutator",
    "Critique",
    "FuzzyAnalogyMutator",
    "HashEmbedder",
    "KnownFailureCritic",
    "ProofAdapter",
    "ProofResult",
    "ResearchCritic",
    "ResearchEvaluator",
    "ResearchStateSpace",
    "TemplateGenerator",
    "novelty_score",
    "surprise_score",
    "diversity_score",
    "serendipity_score",
    "diminishing_returns",
    "balanced_score",
]
