import math
from dataclasses import replace

import numpy as np
import pytest

from racing.agents.random import RandomAgent
from racing.agents.scripted import BaselineConfig, ScriptedAgent, load_baselines
from racing.core.config import PhysicsConfig
from racing.core.models import CarSpawn, CarStatus
from racing.core.simulation import Simulation


def test_random_agent_independent_streams(world, demo_track):
    agent = RandomAgent()
    agent.reset(9000)
    rng = np.random.default_rng(9000)
    for _ in range(8):
        expected = rng.uniform(-1, 1, 2)
        np.random.seed(7)
        actual = agent.act(None, world, demo_track)
        assert (actual.steering, actual.longitudinal) == pytest.approx(expected)
    agent.reset(9000)
    assert agent.act(None, world, demo_track).steering == pytest.approx(
        np.random.default_rng(9000).uniform(-1, 1)
    )


def test_scripted_straight_and_curvature(world, demo_track):
    agent = ScriptedAgent()
    control = agent.act(None, world, demo_track)
    assert -1 <= control.steering <= 1
    assert control.longitudinal == 1
    assert agent.max_curvature(demo_track, world.cars[0].projection.s_m) > 0
    assert load_baselines() == BaselineConfig()
    for kwargs in (
        {"lookahead_min_m": -1},
        {"max_speed_mps": float("nan")},
        {"lookahead_max_m": 1},
    ):
        with pytest.raises(ValueError):
            BaselineConfig(**kwargs)


def test_scripted_uses_published_projection(world, demo_track, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("controller reprojected")

    monkeypatch.setattr(demo_track, "project", forbidden)
    agent = ScriptedAgent()
    first = agent.act(None, world, demo_track)
    car = world.cars[0]
    changed = replace(
        world,
        cars=(
            replace(
                car, state=replace(car.state, heading_rad=car.state.heading_rad + 0.2)
            ),
        ),
    )
    assert agent.act(None, changed, demo_track).steering < first.steering


def test_scripted_real_simulator_full_lap(demo_track):
    sim = Simulation(PhysicsConfig())
    snapshot = sim.reset(demo_track, 1, (CarSpawn("agent", "solo"),))
    agent = ScriptedAgent()
    for _ in range(1200):
        control = agent.act(None, snapshot, demo_track)
        snapshot = sim.step({"agent": control}).snapshot
        car = snapshot.cars[0]
        assert car.status == CarStatus.ACTIVE
        if car.completed_laps:
            break
    assert car.completed_laps == 1
    assert math.isfinite(car.completed_lap_times_s[0])


def test_scripted_curvature_and_commands_have_independent_numeric_fixture(world):
    from types import SimpleNamespace

    # A 90-degree turn over a 10 m incoming segment gives curvature pi/20.
    # At s=5, the vertex at 10 m lies in the 20 m lookahead horizon.
    track = SimpleNamespace(
        length_m=100.0,
        points=np.zeros((4, 2)),
        tangents=np.array(((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))),
        cumulative=np.array((0.0, 10.0, 40.0, 70.0, 100.0)),
        lengths=np.array((10.0, 30.0, 30.0, 30.0)),
        point_at=lambda s: (np.array((5.0, 5.0)), np.array((0.0, 1.0))),
    )
    car = world.cars[0]
    car = replace(
        car,
        projection=replace(car.projection, s_m=5.0),
        state=replace(car.state, x_m=0.0, y_m=0.0, heading_rad=0.0, speed_mps=2.0),
    )
    agent = ScriptedAgent()
    assert agent.max_curvature(track, 5.0) == pytest.approx(math.pi / 20.0)
    control = agent.act(None, replace(world, cars=(car,)), track)
    # lookahead=5 m, error=pi/4; pursuit angle divided by 0.55.
    assert control.steering == pytest.approx(1.0)  # angle .634.. exceeds max .55
    # target speed=sqrt(4/(pi/20))=5.046265044; (target-2)/5.
    assert control.longitudinal == pytest.approx(0.609253008808064)
