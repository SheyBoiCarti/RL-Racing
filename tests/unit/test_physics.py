import pytest

from racing.core.config import PhysicsConfig
from racing.core.models import CarControl, CarState
from racing.core.physics import integrate_car


def test_six_step_order():
    p = PhysicsConfig()
    rest = CarState(0.0, 0.0, 0.0, 0.0, 0.0)
    assert integrate_car(rest, CarControl(0.0, 0.0), p) == rest
    first = integrate_car(rest, CarControl(0.0, 1.0), p)
    assert first.speed_mps == pytest.approx(0.2875)
    assert first.x_m == pytest.approx(0.014375)
    assert integrate_car(rest, CarControl(1.0, 1.0), p).steering_rad == 0.125
    state = rest
    for _ in range(1000):
        state = integrate_car(state, CarControl(1.0, 1.0), p)
        assert state.steering_rad <= 0.55 and 0 <= state.speed_mps <= 22
    for _ in range(100):
        state = integrate_car(state, CarControl(0.0, -1.0), p)
    assert state.speed_mps == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_invalid_control(value):
    with pytest.raises(ValueError):
        integrate_car(
            CarState(0.0, 0.0, 0.0, 0.0, 0.0), CarControl(value, 0.0), PhysicsConfig()
        )
