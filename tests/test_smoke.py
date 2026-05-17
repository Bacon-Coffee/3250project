"""Phase 0 smoke test: verify package layout is importable and toolchain is wired up."""

from __future__ import annotations

import src


def test_package_version_exposed() -> None:
    assert src.__version__ == "0.1.0"


def test_core_subpackage_importable() -> None:
    import src.core  # noqa: F401


def test_algorithms_subpackage_importable() -> None:
    import src.algorithms  # noqa: F401


def test_workloads_subpackage_importable() -> None:
    import src.workloads  # noqa: F401


def test_visualization_subpackage_importable() -> None:
    import src.visualization  # noqa: F401


def test_experiments_subpackage_importable() -> None:
    import src.experiments  # noqa: F401
