"""Tests for strict path containment and traversal rejection."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from curve_fx_sim.specs.common import (
    PathContainmentError,
    assert_contained_path,
    repository_relative,
)


def test_assert_contained_path_valid(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    child = root / "data" / "sub" / "file.csv"
    child.parent.mkdir(parents=True)
    child.touch()

    # Relative path
    resolved = assert_contained_path(Path("data/sub/file.csv"), root)
    assert resolved == child.resolve()

    # Absolute contained path
    resolved2 = assert_contained_path(child, root)
    assert resolved2 == child.resolve()


def test_assert_contained_path_rejects_traversal(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()

    with pytest.raises(PathContainmentError, match="path traversal"):
        assert_contained_path(Path("../secret.txt"), root)

    with pytest.raises(PathContainmentError, match="path traversal"):
        assert_contained_path(Path("data/../../secret.txt"), root)


def test_assert_contained_path_rejects_external_absolute(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.touch()

    with pytest.raises(PathContainmentError, match="escapes root containment"):
        assert_contained_path(outside, root)

    with pytest.raises(PathContainmentError, match="escapes root containment"):
        repository_relative(outside, root)


def test_assert_contained_path_rejects_symlinks_by_default(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "target.txt"
    target.write_text("hello", encoding="utf-8")

    link = root / "link.txt"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("Symlinks not supported on this platform/filesystem")

    with pytest.raises(PathContainmentError, match="refusing symlink"):
        assert_contained_path(link, root, allow_symlinks=False)

    # Allowed when explicit
    resolved = assert_contained_path(link, root, allow_symlinks=True)
    assert resolved == target.resolve()
