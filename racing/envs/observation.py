"""obs-rays7-v1 built exclusively from published world state."""

from math import pi

import numpy as np

from racing.core.geometry import wrap
from racing.envs.config import RLConfig
from racing.envs.sensors import cast_rays


class ObservationBuilder:
    def __init__(self, config=None):
        self.config = config or RLConfig()

    def build(self, snapshot, track, car_id) -> np.ndarray:
        car = next(car for car in snapshot.cars if car.car_id == car_id)
        state, projection, config = car.state, car.projection, self.config
        values = (
            state.x_m,
            state.y_m,
            state.heading_rad,
            state.speed_mps,
            state.steering_rad,
            projection.tangent_heading,
            projection.lateral_m,
        )
        if not np.isfinite(values).all():
            raise ValueError("observation: finite state required")
        rays = cast_rays(track, state, config.ray_angles_deg, config.ray_range_m)
        observation = np.asarray(
            [
                *(rays / config.ray_range_m),
                2 * state.speed_mps / config.physics.max_speed_mps - 1,
                wrap(projection.tangent_heading - state.heading_rad) / pi,
                np.clip(projection.lateral_m / (track.width_m / 2), -1, 1),
                state.steering_rad / config.physics.max_steer_rad,
            ],
            dtype=np.float32,
        )
        if not np.isfinite(observation).all() or np.any(np.abs(observation) > 1):
            raise ValueError("observation: finite values within schema bounds required")
        return observation
