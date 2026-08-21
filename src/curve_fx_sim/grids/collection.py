"""Exact Cartesian grid coverage error."""

from __future__ import annotations

class GridCoverageError(ValueError):
    """A grid result does not exactly cover its Cartesian plan."""


__all__ = ["GridCoverageError"]
