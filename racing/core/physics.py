"""The sole pure six-step kinematic update."""

from math import cos, isfinite, sin, tan

from racing.core.geometry import wrap
from racing.core.models import CarControl, CarState


def validate_control(control):
    if not isinstance(control, CarControl) or any(
        type(v) not in (int, float) or not isfinite(v)
        for v in (control.steering, control.longitudinal)
    ):
        raise ValueError("control: finite steering and longitudinal required")


def integrate_car(state, control, physics):
    validate_control(control)

    def clip(v, lo, hi):
        return min(hi, max(lo, v))

    command = clip(control.steering, -1.0, 1.0)
    u = clip(control.longitudinal, -1.0, 1.0)
    steering = state.steering_rad + clip(
        command * physics.max_steer_rad - state.steering_rad,
        -physics.steering_slew_radps * physics.dt_s,
        physics.steering_slew_radps * physics.dt_s,
    )
    acceleration = (
        u * (physics.acceleration_mps2 if u >= 0 else physics.braking_mps2)
        - physics.rolling_drag_mps2
    )
    speed = clip(
        state.speed_mps + acceleration * physics.dt_s, 0.0, physics.max_speed_mps
    )
    heading = wrap(
        state.heading_rad + speed / physics.wheelbase_m * tan(steering) * physics.dt_s
    )
    return CarState(
        state.x_m + speed * cos(heading) * physics.dt_s,
        state.y_m + speed * sin(heading) * physics.dt_s,
        heading,
        speed,
        steering,
    )
