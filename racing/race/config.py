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


@dataclass(frozen=True, slots=True)
class RaceRules:
    version: str = "race-recovery-v1"
    difficulty: str = "medium"
    recovery_hold_ticks: int = 20
    pace_adapter: str = "pace-governor-v1"

    def __post_init__(self):
        if self.version != "race-recovery-v1":
            raise ValueError("version: unsupported race rules")
        if self.difficulty not in ("easy", "medium", "hard"):
            raise ValueError("difficulty: unsupported tier")
        if type(self.recovery_hold_ticks) is not int or self.recovery_hold_ticks != 20:
            raise ValueError("recovery_hold_ticks: integer 20 required")
        if self.pace_adapter != "pace-governor-v1":
            raise ValueError("pace_adapter: unsupported adapter")
