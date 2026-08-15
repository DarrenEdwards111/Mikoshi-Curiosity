"""Mikoshi Curiosity — domain-agnostic exploration engine."""

__version__ = "0.3.0"

from mikoshi_curiosity.space import State, StateSpace
from mikoshi_curiosity.engine import CuriosityEngine
from mikoshi_curiosity.memory import ExplorationMemory, MemoryEntry
from mikoshi_curiosity.prediction import PredictionModel
from mikoshi_curiosity.results import Discovery, ExplorationResult, ExplorationStats
from mikoshi_curiosity.archive import ResearchArchive
from mikoshi_curiosity.lean_repair import LeanRepairAdapter
from mikoshi_curiosity.llm import (
    AnthropicProvider,
    CallableTextProvider,
    LLMConjectureGenerator,
    LLMLeanRepairer,
    OllamaProvider,
    OpenAIProvider,
    TextProvider,
)
from mikoshi_curiosity.model_finder import (
    FiniteModelFinder,
    FiniteModelProofAdapter,
    FiniteModelSpec,
    ModelSearchResult,
)
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
    "ResearchArchive",
    "LeanRepairAdapter",
    "AnthropicProvider",
    "CallableTextProvider",
    "LLMConjectureGenerator",
    "LLMLeanRepairer",
    "OllamaProvider",
    "OpenAIProvider",
    "TextProvider",
    "FiniteModelFinder",
    "FiniteModelProofAdapter",
    "FiniteModelSpec",
    "ModelSearchResult",
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
