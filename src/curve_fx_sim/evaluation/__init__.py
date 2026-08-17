"""Evaluator identity, harness client adapter, and selection normalization."""

from .client import HarnessClient, ScenarioHarnessClient, SubprocessHarnessClient
from .identity import VerifiedEvaluator, inspect_binary_identity, validate_evaluator_identity
from .selection import (
    ReplayPlan,
    SelectionKind,
    SelectionRef,
    load_attested_evaluation_table,
    normalize_selection,
)

__all__ = [
    "VerifiedEvaluator",
    "inspect_binary_identity",
    "validate_evaluator_identity",
    "HarnessClient",
    "ScenarioHarnessClient",
    "SubprocessHarnessClient",
    "SelectionRef",
    "load_attested_evaluation_table",
    "ReplayPlan",
    "normalize_selection",
]
