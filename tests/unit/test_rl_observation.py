from dataclasses import replace
from math import pi

import numpy as np
import pytest

from racing.core.models import CarSpawn
from racing.core.simulation import Simulation
from racing.envs.config import RLConfig
from racing.envs.observation import ObservationBuilder
from racing.tracks.loader import Track


def test_spawn_and_published_projection(monkeypatch):
    config = RLConfig()
    track = Track.from_json("assets/tracks/demo.json")
    world = Simulation(config.physics).reset(track, 1, [CarSpawn("agent", "solo")])
    builder = ObservationBuilder(config)
    observation = builder.build(world, track, "agent")
    assert observation.shape == (11,)
    assert observation.dtype == np.float32
    assert observation[7] == -1 and observation[10] == 0
    assert observation[8] == pytest.approx(0)
    car = world.cars[0]
    car = replace(
        car,
        projection=replace(
            car.projection, tangent_heading=car.state.heading_rad + pi / 2, lateral_m=3
        ),
    )
    monkeypatch.setattr(
        track, "project", lambda *a: pytest.fail("must use published projection")
    )
    changed = builder.build(replace(world, cars=(car,)), track, "agent")
    assert changed[8] == pytest.approx(0.5)
    assert changed[9] == pytest.approx(0.5)
    assert np.isfinite(changed).all()
    wrapped_car = replace(
        car,
        projection=replace(
            car.projection, tangent_heading=car.state.heading_rad - 3 * pi / 2
        ),
    )
    wrapped = builder.build(replace(world, cars=(wrapped_car,)), track, "agent")
    assert wrapped[8] == pytest.approx(0.5)


def test_invalid_state_is_not_hidden():
    config = RLConfig()
    track = Track.from_json("assets/tracks/demo.json")
    world = Simulation(config.physics).reset(track, 1, [CarSpawn("agent", "solo")])
    car = world.cars[0]
    world = replace(
        world, cars=(replace(car, state=replace(car.state, speed_mps=float("nan"))),)
    )
    with pytest.raises(ValueError, match="finite"):
        ObservationBuilder(config).build(world, track, "agent")


def test_observation_order_preserves_ambiguous_core_projection(world, monkeypatch):
    # Two equal-distance segments at a square corner have different tangents.
    # The published core choice must survive arbitrary observation call order.
    # Start midway along the top edge so finite checkpoint gates are valid.
    points = np.array(
        ((50.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0), (0.0, 0.0))
    )
    track = Track(
        {
            "track_id": "square",
            "name": "Ambiguous corner",
            "width_m": 12.0,
            "centerline": points.tolist(),
            "checkpoints": [{"index": i, "s_m": i * 100.0} for i in range(4)],
            "spawns": [
                {"spawn_id": sid, "s_m": 395.0, "lateral_m": lateral}
                for sid, lateral in [
                    ("solo", 0.0),
                    ("grid_left", -1.5),
                    ("grid_right", 1.5),
                ]
            ],
        }
    )
    # At (3,3), both outgoing and incoming segments lie 3 m away.
    projection = track.project(3.0, 3.0)
    assert projection.segment_index == 3
    car = world.cars[0]
    car = replace(
        car,
        projection=projection,
        state=replace(car.state, x_m=3.0, y_m=3.0, heading_rad=0.0),
    )
    snapshot = replace(world, track_id="square", cars=(car,))
    builder = ObservationBuilder()
    before = builder.build(snapshot, track, car.car_id)
    monkeypatch.setattr(
        track, "project", lambda *a, **kw: pytest.fail("adapter reprojected")
    )
    # Alternate with a different pose, then revisit exactly the same snapshot.
    builder.build(
        replace(
            snapshot, cars=(replace(car, state=replace(car.state, heading_rad=0.2)),)
        ),
        track,
        car.car_id,
    )
    after = builder.build(snapshot, track, car.car_id)
    np.testing.assert_array_equal(before, after)
    assert after[8] == pytest.approx(-0.5)
    assert after[9] == pytest.approx(0.5)
    assert snapshot.cars[0].projection is projection
