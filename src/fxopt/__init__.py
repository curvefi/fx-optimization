"""Cartesian-grid execution and result-artifact API."""

from .contract import Candidate, CandidateResult
from .engine import ClientFactory, EvaluatorClient, EvaluatorSession
from .results import (
    ArtifactPaths,
    GridResultWriter,
    read_result_columns,
)
from .run import RunConfig, run_config

__all__ = [
    "ArtifactPaths",
    "Candidate",
    "CandidateResult",
    "ClientFactory",
    "EvaluatorClient",
    "GridResultWriter",
    "EvaluatorSession",
    "RunConfig",
    "read_result_columns",
    "run_config",
]
