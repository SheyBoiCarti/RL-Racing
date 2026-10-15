"""Privileged geometric follower: track knowledge is unavailable to learned policies."""

import math
from dataclasses import dataclass, fields

import numpy as np

from racing.core.config import PhysicsConfig, _read
from racing.core.geometry import wrap
from racing.core.models import CarControl


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    lookahead_base_m: float = 4.0
    lookahead_speed_s: float = 0.5
    lookahead_min_m: float = 4.0
    lookahead_max_m: float = 15.0
    curvature_horizon_m: float = 20.0
    max_speed_mps: float = 16.0
    lateral_acceleration_mps2: float = 4.0
    curvature_floor: float = 0.001
    speed_gain_divisor: float = 5.0

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{field.name}: expected finite positive number")
            object.__setattr__(self, field.name, float(value))
        if not self.lookahead_min_m <= self.lookahead_base_m <= self.lookahead_max_m:
            raise ValueError("lookahead bounds must contain base")


def load_baselines(path="configs/baselines.yaml") -> BaselineConfig:
    filename, data = _read(path, BaselineConfig)
    try:
        return BaselineConfig(**data)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{filename}: {error}") from error


class ScriptedAgent:
    def __init__(self, config=None, physics=None):
        self.config = config or load_baselines()
        self.physics = physics or PhysicsConfig()

    def reset(self, seed: int) -> None:
        pass

    def max_curvature(self, track, s_m):
        # A vertex turn is assigned to its preceding polyline segment.
        # Inspect every upcoming vertex, including the wrap at the finish.
        horizon = self.config.curvature_horizon_m
        offsets = (track.cumulative[:-1] - s_m) % track.length_m
        indices = np.flatnonzero(offsets <= horizon)
        maximum = 0.0
        for index in indices:
            before = track.tangents[(index - 1) % len(track.points)]
            after = track.tangents[index]
            turn = wrap(
                math.atan2(after[1], after[0]) - math.atan2(before[1], before[0])
            )
            maximum = max(
                maximum,
                abs(turn) / float(track.lengths[(index - 1) % len(track.points)]),
            )
        return maximum

    def act(self, observation, snapshot, track) -> CarControl:
        car = next((c for c in snapshot.cars if c.car_id == "agent"), snapshot.cars[0])
        state, config = car.state, self.config
        lookahead = float(
            np.clip(
                config.lookahead_base_m + config.lookahead_speed_s * state.speed_mps,
                config.lookahead_min_m,
                config.lookahead_max_m,
            )
        )
        target, _ = track.point_at(car.projection.s_m + lookahead)
        error = wrap(
            math.atan2(target[1] - state.y_m, target[0] - state.x_m) - state.heading_rad
        )
        steering = (
            math.atan2(2 * self.physics.wheelbase_m * math.sin(error), lookahead)
            / self.physics.max_steer_rad
        )
        curvature = self.max_curvature(track, car.projection.s_m)
        target_speed = min(
            config.max_speed_mps,
            math.sqrt(
                config.lateral_acceleration_mps2
                / max(curvature, config.curvature_floor)
            ),
        )
        longitudinal = (target_speed - state.speed_mps) / config.speed_gain_divisor
        return CarControl(
            float(np.clip(steering, -1, 1)), float(np.clip(longitudinal, -1, 1))
        )
