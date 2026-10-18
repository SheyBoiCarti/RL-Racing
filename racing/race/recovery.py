"""Immutable race recovery state and pure safe-anchor restoration."""

from dataclasses import dataclass, replace
from math import isfinite

from racing.core.models import CarSnapshot, CarStatus


@dataclass(frozen=True, slots=True)
class RecoveryState:
    safe: CarSnapshot
    remaining_ticks: int = 0
    count: int = 0
    held_ticks: int = 0
    last_release_tick: int | None = None


def restored_snapshot(safe, track, physics):
    if not isinstance(safe, CarSnapshot) or safe.status != CarStatus.ACTIVE:
        raise ValueError("recovery: active safe snapshot required")
    values = (
        safe.state.x_m,
        safe.state.y_m,
        safe.state.heading_rad,
        safe.state.speed_mps,
        safe.state.steering_rad,
        safe.projection.tangent_heading,
    )
    if any(type(v) not in (int, float) or not isfinite(v) for v in values):
        raise ValueError("recovery: finite pose required")
    if not track.contains_disk(safe.state.x_m, safe.state.y_m, physics.car_radius_m):
        raise ValueError("recovery: unsafe disk")
    return replace(
        safe,
        state=replace(
            safe.state,
            heading_rad=safe.projection.tangent_heading,
            speed_mps=0.0,
            steering_rad=0.0,
        ),
        signed_delta_m=0.0,
        rewardable_delta_m=0.0,
        projection_discontinuity=False,
    )
