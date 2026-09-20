from __future__ import annotations

from backend.slam.types import CameraIntrinsics


def build_intrinsics(
    *,
    width: int,
    height: int,
    fx: float | None = None,
    fy: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
) -> CameraIntrinsics:
    supplied = (fx, fy, cx, cy)
    if all(value is None for value in supplied):
        focal = 0.9 * max(width, height)
        return CameraIntrinsics(
            fx=focal,
            fy=focal,
            cx=width / 2,
            cy=height / 2,
            width=width,
            height=height,
            approximate=True,
        )
    if any(value is None for value in supplied):
        raise ValueError("Provide all four values: fx, fy, cx, and cy")
    assert fx is not None and fy is not None and cx is not None and cy is not None
    return CameraIntrinsics(
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),
        width=width,
        height=height,
        approximate=False,
    )
