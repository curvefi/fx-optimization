"""Lean optimizer execution and result-artifact API."""

from .contract import Candidate, CandidateResult
from .engine import ClientFactory, EvaluatorClient, OptimizerEngine
from .results import (
    ArtifactPaths,
    ResultWriter,
    ResultBundle,
    read_results,
    write_results,
)
from .run import RunConfig, run_config

__all__ = [
    "ArtifactPaths",
    "Candidate",
    "CandidateResult",
    "ClientFactory",
    "EvaluatorClient",
    "OptimizerEngine",
    "ResultBundle",
    "ResultWriter",
    "RunConfig",
    "read_results",
    "write_results",
    "run_config",
]
