"""Compile named optimizer proposals through the selected schema."""

from __future__ import annotations

from typing import Mapping, Sequence

from ..evaluation.plans import CandidateCompiler, CandidatePlan, ScenarioKey
from ..specs.scenario import ScenarioClosure
from .profiles import NamedProfile
from .search import SearchLayout


def compile_named_request(
    profile: NamedProfile | SearchLayout,
    vector: Sequence[int | float],
    compiler: CandidateCompiler,
    *,
    open_session: dict[str, object] | Mapping[str, object],
    scenario: ScenarioClosure | ScenarioKey,
) -> CandidatePlan:
    """Compile one optimizer vector through the canonical named-proposal seam."""
    proposal = profile.to_proposal(vector)
    return compiler.compile(proposal, open_session=open_session, scenario=scenario)


__all__ = ["compile_named_request"]
