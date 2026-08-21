"""Evaluator artifacts, identity, and selection normalization."""

from .builder import (
    BuildSpec,
    EvaluatorArtifact,
    EvaluatorBuildError,
    build_evaluator,
    build_receipt,
    load_evaluator_artifact,
)
from .identity import VerifiedEvaluator, inspect_binary_identity, validate_evaluator_identity
from .grouping import (
    CompiledEvaluation,
    EvaluationGroupingError,
    LocalSessionGroupBinding,
    ObservationGroup,
    PortableCandidate,
    SessionGroup,
    SessionGroupKey,
    bind_local_session_group,
    group_evaluations,
)
from .plans import ObservationKey
from .session import (
    LocalSessionMaterialization,
    LocalSessionTransportReceipt,
    SessionMaterializationError,
)
from .selected import SelectedEvaluator, materialize_selected_evaluator
from .selection import (
    ReplayPlan,
    SelectionKind,
    SelectionRef,
    load_attested_evaluation_table,
    normalize_selection,
)

__all__ = [
    "BuildSpec",
    "EvaluatorArtifact",
    "EvaluatorBuildError",
    "build_evaluator",
    "build_receipt",
    "load_evaluator_artifact",
    "VerifiedEvaluator",
    "inspect_binary_identity",
    "validate_evaluator_identity",
    "SelectedEvaluator",
    "materialize_selected_evaluator",
    "LocalSessionMaterialization",
    "LocalSessionTransportReceipt",
    "SessionMaterializationError",
    "ObservationKey",
    "CompiledEvaluation",
    "EvaluationGroupingError",
    "ObservationGroup",
    "PortableCandidate",
    "SessionGroupKey",
    "SessionGroup",
    "LocalSessionGroupBinding",
    "group_evaluations",
    "bind_local_session_group",
    "SelectionRef",
    "load_attested_evaluation_table",
    "ReplayPlan",
    "normalize_selection",
]
