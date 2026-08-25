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
from .optimize import optimize_config
from .shiftclick import save_shiftclick_plot, shiftclick_figure, trace_candidate

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
    "optimize_config",
    "save_shiftclick_plot",
    "shiftclick_figure",
    "trace_candidate",
]
