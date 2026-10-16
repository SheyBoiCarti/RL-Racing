"""Validated, immutable RL configuration."""

from dataclasses import dataclass, field, fields
from math import isfinite

from racing.core.config import PhysicsConfig, _read

RAY_ANGLES = (-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0)


@dataclass(frozen=True, slots=True)
class RLConfig:
    observation_version: str = "obs-rays7-v1"
    action_version: str = "action-v1"
    reward_version: str = "reward-v1"
    ray_angles_deg: tuple[float, ...] = RAY_ANGLES
    ray_range_m: float = 40.0
    target_laps: int = 1
    max_steps: int = 1200
    no_progress_steps: int = 120
    progress_threshold_m: float = 0.01
    default_track_id: str = "demo"
    default_spawn_id: str = "solo"
    allowed_split: str = "demo"
    track_sampling: str = "fixed"
    split_manifest: str | None = None
    steering_penalty: float = 0.01
    progress_coefficient: float = 2.0
    step_cost: float = 0.002
    off_track_penalty: float = -10.0
    lap_bonus: float = 50.0
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)

    def __post_init__(self):
        if self.track_sampling not in ("fixed", "uniform_train"):
            raise ValueError("track_sampling: expected fixed or uniform_train")
        if self.track_sampling == "uniform_train" and self.allowed_split != "train":
            raise ValueError("uniform_train requires allowed_split=train")
        for name, expected in (
            ("observation_version", "obs-rays7-v1"),
            ("action_version", "action-v1"),
            ("reward_version", "reward-v1"),
        ):
            if getattr(self, name) != expected:
                raise ValueError(f"{name}: unsupported version")
        if (
            not isinstance(self.ray_angles_deg, (tuple, list))
            or any(type(angle) not in (int, float) for angle in self.ray_angles_deg)
            or tuple(self.ray_angles_deg) != RAY_ANGLES
            or self.ray_range_m != 40
        ):
            raise ValueError("ray settings: incompatible with obs-rays7-v1")
        object.__setattr__(self, "ray_angles_deg", RAY_ANGLES)
        for name in ("target_laps", "max_steps", "no_progress_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name}: positive integer required")
        for name in (
            "ray_range_m",
            "progress_threshold_m",
            "steering_penalty",
            "progress_coefficient",
            "step_cost",
            "off_track_penalty",
            "lap_bonus",
        ):
            value = getattr(self, name)
            if type(value) not in (float, int) or not isfinite(value):
                raise ValueError(f"{name}: finite number required")
            if name == "off_track_penalty":
                if value > 0:
                    raise ValueError(f"{name}: nonpositive number required")
            elif value < 0 or (name == "progress_coefficient" and value == 0):
                raise ValueError(f"{name}: invalid reward/config range")
            object.__setattr__(self, name, float(value))
        for name in ("default_track_id", "default_spawn_id", "allowed_split"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name}: nonempty string required")
        if not isinstance(self.physics, PhysicsConfig):
            raise ValueError("physics: PhysicsConfig required")
        object.__setattr__(
            self,
            "physics",
            PhysicsConfig(
                **{
                    item.name: float(getattr(self.physics, item.name))
                    for item in fields(PhysicsConfig)
                }
            ),
        )


def load_rl(path="configs/rl.yaml"):
    filename, data = _read(path, RLConfig)
    try:
        if "physics" in data:
            data["physics"] = PhysicsConfig(**data["physics"])
        return RLConfig(**data)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{filename}: {error}") from error
