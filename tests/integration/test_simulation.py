import math
import subprocess
import sys
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from racing.core.config import PhysicsConfig, load_game
from racing.core.geometry import wrap
from racing.core.models import CarControl, CarSpawn, CarStatus
from racing.core.progress import continuous
from racing.core.simulation import Simulation
from racing.render.app import Application, AppState
from racing.tracks import Track


@pytest.fixture(scope="module")
def track():
    return Track.from_json("assets/tracks/demo.json")


def simulator(track):
    sim = Simulation(PhysicsConfig())
    sim.reset(track, 7, (CarSpawn("human", "solo"),))
    return sim


def test_atomic_validation_and_reset(track):
    sim = simulator(track)
    original = sim.snapshot()
    for controls in (
        {"bad": CarControl(0.0, 0.0)},
        {"human": CarControl(float("nan"), 0.0)},
        [],
    ):
        with pytest.raises(ValueError):
            sim.step(controls)
        assert sim.snapshot() == original
    sim.step({"human": CarControl(0.0, 1.0)})
    assert sim.reset(track, 7, (CarSpawn("human", "solo"),)) == original
    with pytest.raises(FrozenInstanceError):
        original.cars[0].state.x_m = 999
    with pytest.raises(FrozenInstanceError):
        original.cars[0].completed_laps = 999


def test_deterministic_500_ticks_and_failure_freeze(track):
    a, b = simulator(track), simulator(track)
    failure_count = 0
    for _ in range(500):
        controls = {"human": CarControl(0.2, 1.0)}
        first, second = a.step(controls), b.step(controls)
        assert first == second
        failure_count += sum(e.kind.value == "off_track" for e in first.events)
        if first.snapshot.cars[0].status == CarStatus.DISQUALIFIED:
            pose = first.snapshot.cars[0].state
    assert failure_count == 1
    assert a.snapshot().cars[0].state == pose


def test_full_simulator_lap(track):
    sim = simulator(track)
    kinds = []
    # Deterministic test driver: pure pursuit consumes published projection.
    for _ in range(1800):
        car = sim.snapshot().cars[0]
        target, _ = track.point_at(car.projection.s_m + 7)
        angle = wrap(
            math.atan2(target[1] - car.state.y_m, target[0] - car.state.x_m)
            - car.state.heading_rad
        )
        distance = math.hypot(target[0] - car.state.x_m, target[1] - car.state.y_m)
        steer = (
            math.atan2(2 * sim.config.wheelbase_m * math.sin(angle), distance)
            / sim.config.max_steer_rad
        )
        throttle = (
            0.25 * (10 - car.state.speed_mps)
            + sim.config.rolling_drag_mps2 / sim.config.acceleration_mps2
        )
        result = sim.step({"human": CarControl(steer, throttle)})
        kinds.extend(e.kind.value for e in result.events)
        assert result.snapshot.cars[0].status == CarStatus.ACTIVE
        assert not result.snapshot.cars[0].projection_discontinuity
        if result.snapshot.cars[0].completed_laps:
            break
    assert kinds == ["lap_started"] + ["checkpoint_crossed"] * (
        len(track.gates) - 1
    ) + ["lap_completed"]
    car = sim.snapshot().cars[0]
    assert car.completed_laps == 1 and len(car.completed_lap_times_s) == 1
    assert car.completed_lap_times_s[0] > 50
    # Core remains active after one lap; application owns one-lap termination.
    continuation = sim.step({})
    assert continuation.snapshot.cars[0].status == CarStatus.ACTIVE
    assert continuation.snapshot.cars[0].completed_laps == 1
    app = Application(load_game(), sim.config, track)
    app.simulation = sim
    app.action("resume")
    app.state = AppState.DRIVING
    app.frame(0.05, CarControl(0.0, 0.0))
    assert app.state == AppState.ATTEMPT_FINISHED
    assert app.best_lap_s == car.completed_lap_times_s[0]
    finished = sim.retire_car("human", CarStatus.FINISHED)
    assert sim.retire_car("human", CarStatus.FINISHED) == finished
    assert sim.step({}).snapshot.cars == finished.cars
    app.action("reset")
    assert app.best_lap_s == car.completed_lap_times_s[0]
    assert app.simulation.snapshot().tick == 0


def test_corner_switch_on_full_valid_track():
    points = np.array(
        [
            [-60, 0],
            [-10, 0],
            [0, 0],
            [10, 10],
            [40, 40],
            [80, 40],
            [100, 0],
            [80, -60],
            [-60, -60],
            [-80, -30],
        ],
        dtype=float,
    )
    length = float(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1).sum())
    track = Track(
        {
            "track_id": "corner",
            "name": "Independent corner fixture",
            "width_m": 12.0,
            "centerline": points.tolist(),
            "checkpoints": [{"index": i, "s_m": i * length / 4} for i in range(4)],
            "spawns": [
                {"spawn_id": sid, "s_m": length - 5, "lateral_m": lateral}
                for sid, lateral in (
                    ("solo", 0.0),
                    ("grid_left", -1.5),
                    ("grid_right", 1.5),
                )
            ],
        }
    )
    previous = track.project(-2.08, 5.0)
    candidate = track.project(-2.06, 5.0, previous.s_m)
    assert candidate.s_m - previous.s_m == pytest.approx(4.158893936688449)
    assert track.contains_disk(-2.08, 5.0, 0.8)
    assert track.contains_disk(-2.06, 5.0, 0.8)
    assert continuous(track, previous, candidate, 0.02, PhysicsConfig())


def test_core_import_without_pygame():
    script = "import sys; sys.modules['pygame']=None; from racing.core.simulation import Simulation; from racing.tracks import Track"
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True)
