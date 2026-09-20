from __future__ import annotations

import pytest

from backend.slam.intrinsics import build_intrinsics


def test_approximate_intrinsics_are_centered_and_disclosed() -> None:
    intrinsics = build_intrinsics(width=640, height=360)

    assert intrinsics.fx == pytest.approx(576.0)
    assert intrinsics.fy == pytest.approx(576.0)
    assert intrinsics.cx == pytest.approx(320.0)
    assert intrinsics.cy == pytest.approx(180.0)
    assert intrinsics.approximate is True


def test_provided_intrinsics_are_preserved_and_must_be_complete() -> None:
    intrinsics = build_intrinsics(
        width=640, height=360, fx=710.0, fy=705.0, cx=318.0, cy=181.0
    )
    assert intrinsics.matrix.tolist() == [
        [710.0, 0.0, 318.0],
        [0.0, 705.0, 181.0],
        [0.0, 0.0, 1.0],
    ]
    assert intrinsics.approximate is False

    with pytest.raises(ValueError, match="all four"):
        build_intrinsics(width=640, height=360, fx=700.0)

