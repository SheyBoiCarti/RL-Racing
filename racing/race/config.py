"""Gameplay limits independent of model observation compatibility."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class RaceConfig:
    target_laps: int = 3
    countdown_seconds: float = 3.0
    race_max_steps: int = 7200
    no_progress_steps: int = 120
    ghost: bool = True
    progress_threshold_m: float = 0.01

    def __post_init__(self):
        for name in ("target_laps", "race_max_steps", "no_progress_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name}: positive integer required")
        for name in ("countdown_seconds", "progress_threshold_m"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not isfinite(value) or value < 0:
                raise ValueError(f"{name}: finite nonnegative number required")
            object.__setattr__(self, name, float(value) if value else 0.0)
        if self.ghost is not True:
            raise ValueError("ghost: only ghost racing is supported")
