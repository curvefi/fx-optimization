"""Named optimizer geometry backed by the selected evaluator schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .lattice import LatticeSpec
from .search import SearchLayout


@dataclass(frozen=True, slots=True)
class NamedProfile:
    """Optimizer identity and the selected schema's canonical search layout."""

    optimizer_name: str
    layout: SearchLayout

    def __post_init__(self) -> None:
        if not self.optimizer_name:
            raise ValueError("optimizer_name must be non-empty")

    @property
    def geometry_sha256(self) -> str:
        return self.layout.sha256

    @property
    def default_vector(self) -> tuple[int | float, ...]:
        return self.layout.default_vector

    @property
    def initial_vector(self) -> tuple[int | float, ...]:
        return self.default_vector

    @property
    def lattice(self) -> LatticeSpec:
        return self.layout.create_lattice_spec()

    def to_proposal(self, vector: Sequence[int | float]) -> dict[str, int | float]:
        return self.layout.to_proposal(vector)

    @property
    def geometry_receipt(self) -> dict[str, object]:
        return {
            "optimizer_name": self.optimizer_name,
            "schema_sha256": self.layout.schema_sha256,
            "layout_sha256": self.layout.sha256,
            "dimensions": [
                {
                    "name": item.name,
                    "default": str(item.default),
                    "minimum": str(item.minimum),
                    "maximum": str(item.maximum),
                    "step": str(item.step),
                    "transform": item.transform,
                }
                for item in self.layout.dimensions
            ],
        }


__all__ = ["NamedProfile"]
