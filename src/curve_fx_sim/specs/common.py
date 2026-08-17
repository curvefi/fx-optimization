"""Common helpers, exact Decimal canonicalization, and strict path containment for specs."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence


class SpecError(ValueError):
    """Raised when a specification is invalid or malformed."""


class PathContainmentError(ValueError):
    """Raised when a path escapes its containment root."""


def canonical_decimal(value: Any, *, label: str = "value") -> Decimal:
    """Coerce a numeric value into an exact canonical, finite Decimal."""
    if isinstance(value, bool):
        raise SpecError(f"{label} cannot be a boolean")
    if isinstance(value, Decimal):
        dec = value
    elif isinstance(value, (int, str)):
        try:
            dec = Decimal(str(value))
        except InvalidOperation as exc:
            raise SpecError(f"{label} is not a valid decimal: {value!r}") from exc
    elif isinstance(value, float):
        try:
            dec = Decimal(str(value))
        except InvalidOperation as exc:
            raise SpecError(f"{label} is not a valid decimal: {value!r}") from exc
    else:
        raise SpecError(f"{label} must be numeric, got {type(value).__name__}")

    if not dec.is_finite():
        raise SpecError(f"{label} must be a finite decimal, got {value!r}")
    return dec


def format_exact_decimal(value: Decimal) -> str:
    """Format a finite Decimal to a deterministic plain string without exponent notation."""
    if not value.is_finite():
        raise ValueError(f"expected finite Decimal, got {value!r}")
    sign, digits, exp = value.as_tuple()
    if exp >= 0:
        s = "".join(str(d) for d in digits) + "0" * exp
        if not s:
            s = "0"
        return f"-{s}" if sign else s
    digits_str = "".join(str(d) for d in digits)
    total_len = len(digits_str)
    needed = abs(exp)
    if total_len <= needed:
        prefix = "0." + "0" * (needed - total_len) + digits_str
    else:
        split_point = total_len - needed
        prefix = digits_str[:split_point] + "." + digits_str[split_point:]
    return f"-{prefix}" if sign else prefix


def canonical_primitive(value: Any) -> Any:
    """Normalize primitives and containers into deterministic JSON-safe structures without precision loss."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise SpecError(f"cannot canonicalize non-finite Decimal: {value!r}")
        if value == value.to_integral():
            return int(value)
        return format_exact_decimal(value)
    if isinstance(value, float):
        dec = Decimal(str(value))
        if not dec.is_finite():
            raise SpecError(f"cannot canonicalize non-finite float: {value!r}")
        return value
    if isinstance(value, Path):
        return str(value.as_posix())
    if isinstance(value, Mapping):
        return {str(k): canonical_primitive(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [canonical_primitive(v) for v in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return canonical_primitive(value.to_dict())
    if hasattr(value, "__dict__"):
        return canonical_primitive(vars(value))
    return str(value)


def canonical_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively convert a mapping to a sorted-key canonical dictionary."""
    if not isinstance(value, Mapping):
        raise TypeError(f"expected mapping, got {type(value).__name__}")
    return {str(k): canonical_primitive(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic canonical UTF-8 JSON bytes for hashing and attestation."""
    normalized = canonical_primitive(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _find_repo_root(candidate: Path) -> Path:
    current = candidate.resolve()
    if current.is_file():
        current = current.parent
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").is_file() or (parent / ".git").exists():
            return parent
    return current


def repository_root(path: str | os.PathLike[str] | None = None) -> Path:
    """Return the repository root used for relative serialized paths."""
    if path is not None:
        candidate = Path(path)
        return _find_repo_root(candidate)
    return _find_repo_root(Path.cwd())


def assert_contained_path(
    path: str | os.PathLike[str],
    root: str | os.PathLike[str],
    *,
    allow_symlinks: bool = False,
) -> Path:
    """Ensure *path* is safely contained inside *root* without escaping or traversal."""
    root_resolved = Path(root).resolve()
    target = Path(path)

    # Check for relative traversal in raw path
    parts = target.parts
    if ".." in parts:
        raise PathContainmentError(f"path traversal '..' is prohibited: {target}")

    target_resolved = (root_resolved / target).resolve() if not target.is_absolute() else target.resolve()

    if not allow_symlinks and (target.is_symlink() or target_resolved.is_symlink()):
        raise PathContainmentError(f"refusing symlink: {target}")

    try:
        target_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise PathContainmentError(
            f"path {target_resolved} escapes root containment directory {root_resolved}"
        ) from exc

    return target_resolved


def repository_relative(path: str | os.PathLike[str], root: Path | None = None) -> Path:
    """Normalize a path to a repository-relative Path without '..' escapes, raising PathContainmentError if outside root."""
    repo = root.resolve() if root is not None else repository_root()
    target = Path(path)

    resolved = assert_contained_path(target, repo, allow_symlinks=True)
    rel = resolved.relative_to(repo)
    return Path(rel.as_posix())


def serializable(value: Any) -> Any:
    """Convert spec and configuration values into deterministic serializable primitives without float loss."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise SpecError(f"cannot serialize non-finite Decimal: {value!r}")
        if value == value.to_integral():
            return int(value)
        return format_exact_decimal(value)
    if isinstance(value, float):
        dec = Decimal(str(value))
        if not dec.is_finite():
            raise SpecError(f"cannot serialize non-finite float: {value!r}")
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return serializable(value.to_dict())
    return str(value)


__all__ = [
    "SpecError",
    "PathContainmentError",
    "canonical_decimal",
    "format_exact_decimal",
    "canonical_primitive",
    "canonical_dict",
    "canonical_json_bytes",
    "repository_root",
    "repository_relative",
    "assert_contained_path",
    "serializable",
]
