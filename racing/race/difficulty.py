"""Version one deterministic AI pace governor; physical limits are unchanged."""

from math import isfinite

from racing.core.models import CarControl
from racing.core.physics import validate_control


def govern_ai(
    control: CarControl, speed_mps: float, max_speed_mps: float, difficulty: str
) -> CarControl:
    validate_control(control)
    if difficulty not in ("easy", "medium", "hard"):
        raise ValueError("difficulty: unsupported tier")
    if type(speed_mps) not in (int, float) or not isfinite(speed_mps) or speed_mps < 0:
        raise ValueError("speed_mps: finite nonnegative speed required")
    if (
        type(max_speed_mps) not in (int, float)
        or not isfinite(max_speed_mps)
        or max_speed_mps <= 0
    ):
        raise ValueError("max_speed_mps: finite positive speed required")
    if difficulty == "hard":
        return control
    target = (0.60 if difficulty == "easy" else 0.80) * max_speed_mps
    if speed_mps < target:
        return control
    longitudinal = min(
        control.longitudinal, max(-1.0, min(0.0, -(speed_mps - target) / 2.0))
    )
    return CarControl(control.steering, longitudinal)
