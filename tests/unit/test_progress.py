from dataclasses import replace
from math import sqrt
from types import SimpleNamespace

import numpy as np
import pytest

from racing.core.config import PhysicsConfig
from racing.core.models import CarSnapshot, CarState, CarStatus, Projection
from racing.core.progress import advance, continuous, gate_crossing


def gate(x):
    return tuple(np.array(v, dtype=float) for v in ((x, 0), (1, 0), (x, -6), (x, 6)))


@pytest.mark.parametrize(
    "positions,expected",
    [
        ([-0.01, -0.0000005, 0.01], [False, True]),
        ([-0.01, 0, 0.01], [True, False]),
        ([-1e-8, 1e-8], [True]),
        ([0, 0], [False]),
        ([0.01, -0.01], [False]),
    ],
)
def test_exact_crossing(positions, expected):
    assert [
        gate_crossing((a, 0), (b, 0), gate(0)) is not None
        for a, b in zip(positions, positions[1:])
    ] == expected
    assert gate_crossing((-0.1, 6.001), (0.1, 6.001), gate(0)) is None


def fixture():
    track = SimpleNamespace(
        length_m=100.0,
        cumulative=np.array([0.0, 100.0]),
        tangents=np.array([[1.0, 0.0]]),
        gates=(gate(0), gate(25), gate(50), gate(75)),
    )
    state = CarState(-0.1, 0.0, 0.0, 0.0, 0.0)
    projection = Projection(0, 0.0, 99.9, 0.0, 0.0, 0.0)
    car = CarSnapshot(
        "human",
        state,
        CarStatus.ACTIVE,
        projection,
        False,
        0.0,
        0.0,
        0.0,
        0.0,
        0.1,
        0,
        False,
        1,
        None,
        (),
    )
    return track, car


def move(car, track, old_x, new_x, old_s, new_s, tick, y=0):
    car = replace(
        car,
        state=replace(car.state, x_m=old_x, y_m=y),
        projection=replace(car.projection, s_m=old_s),
    )
    return advance(
        car,
        replace(car.state, x_m=new_x),
        replace(car.projection, s_m=new_s),
        track,
        tick,
        PhysicsConfig(),
    )


def test_ordered_lap_and_high_water():
    track, car = fixture()
    # Ignore pre-arm gates and do not reward retracing accepted travel.
    car, events = move(car, track, 24.9, 25.1, 24.9, 25.1, 1)
    assert not events and car.next_checkpoint == 1
    car, _ = move(car, track, 25.1, 24.9, 25.1, 24.9, 2)
    car, _ = move(car, track, 24.9, 25.1, 24.9, 25.1, 3)
    assert car.rewardable_delta_m == pytest.approx(0)
    car, events = move(car, track, -0.1, 0.1, 99.9, 0.1, 4)
    assert [e.kind.value for e in events] == ["lap_started"]
    assert car.next_checkpoint == 1
    # Missing checkpoint prevents a finish.
    car, events = move(car, track, 49.9, 50.1, 49.9, 50.1, 5)
    assert not events and car.next_checkpoint == 1
    car, events = move(car, track, -0.1, 0.1, 99.9, 0.1, 6)
    assert not events and car.completed_laps == 0
    for i, x in enumerate((25, 50, 75), start=7):
        car, events = move(car, track, x - 0.1, x + 0.1, x - 0.1, x + 0.1, i)
        assert [e.kind.value for e in events] == ["checkpoint_crossed"]
    car, events = move(car, track, -0.1, 0.1, 99.9, 0.1, 10)
    assert [e.kind.value for e in events] == ["lap_completed"]
    assert car.completed_lap_times_s == pytest.approx((0.3,))
    assert car.completed_laps == 1 and car.next_checkpoint == 1
    car, events = move(car, track, 0.1, -0.1, 0.1, 99.9, 11)
    assert not events
    car, events = move(car, track, -2, 2, 98, 2, 12)
    assert not events and car.projection_discontinuity
    assert car.signed_delta_m == car.rewardable_delta_m == 0


def test_inner_corner_switch_and_teleport():
    track = SimpleNamespace(
        length_m=100.0,
        cumulative=np.array([0.0, 10.0, 100.0]),
        tangents=np.array([[1.0, 0.0], [1 / sqrt(2), 1 / sqrt(2)]]),
    )
    previous = Projection(0, 0.792, 7.92, 5.0, 0.0, 5.0)
    distance = abs(7.06) / sqrt(2)
    candidate = Projection(1, 0.147, 10 + 2.94 / sqrt(2), distance, 0.785, distance)
    assert candidate.s_m - previous.s_m == pytest.approx(4.158893936688449)
    assert continuous(track, previous, candidate, 0.02, PhysicsConfig())
    assert not continuous(track, previous, candidate, 1.100002, PhysicsConfig())


def test_outside_and_discontinuous_do_not_cross():
    track, car = fixture()
    for old_x, new_x, y in ((-0.1, 0.1, 6.001), (-2, 2, 0)):
        result, events = move(car, track, old_x, new_x, 99.9, 0.1, 1, y)
        assert not events and not result.finish_armed
