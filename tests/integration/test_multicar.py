from dataclasses import replace

import pytest

from racing.core.config import PhysicsConfig
from racing.core.models import CarControl, CarSpawn, CarStatus
from racing.core.simulation import Simulation


def make(track, spawns):
    simulation = Simulation(PhysicsConfig())
    simulation.reset(track, 7, spawns)
    return simulation


def test_two_car_controls_validate_before_any_motion(demo_track):
    sim = make(
        demo_track, (CarSpawn("human", "grid_left"), CarSpawn("ai", "grid_right"))
    )
    original = sim.snapshot()
    assert tuple(c.car_id for c in original.cars) == ("ai", "human")
    for controls in (
        {"human": CarControl(0, 1), "ai": CarControl(0, float("nan"))},
        {"human": CarControl(0, 1), "unknown": CarControl(0, 0)},
        [],
    ):
        with pytest.raises(ValueError):
            sim.step(controls)
        assert sim.snapshot() is original
    result = sim.step({"human": CarControl(0, 1), "ai": CarControl(0, 0)})
    assert result.snapshot.cars[0].state.speed_mps == 0
    assert result.snapshot.cars[1].state.speed_mps == pytest.approx(0.2875)


def test_order_ids_and_ghost_opponent_preserve_single_car_trajectory(demo_track):
    single = make(demo_track, (CarSpawn("solo", "grid_left"),))
    first = make(demo_track, (CarSpawn("z", "grid_left"), CarSpawn("a", "grid_right")))
    second = make(demo_track, (CarSpawn("z", "grid_right"), CarSpawn("a", "grid_left")))
    for _ in range(180):
        control = CarControl(0.12, 0.7)
        s = single.step({"solo": control}).snapshot.cars[0]
        a = first.step({"a": CarControl(-0.4, 1), "z": control}).snapshot.cars[1]
        b = second.step({"z": CarControl(-0.4, 1), "a": control}).snapshot.cars[0]
        assert replace(a, car_id="solo") == s
        assert replace(b, car_id="solo") == s


def test_retirement_freezes_only_selected_car(demo_track):
    sim = make(
        demo_track, (CarSpawn("human", "grid_left"), CarSpawn("ai", "grid_right"))
    )
    retired = sim.retire_car("ai", CarStatus.TIMED_OUT)
    assert sim.retire_car("ai", CarStatus.TIMED_OUT) is retired
    for _ in range(5):
        sim.step({"human": CarControl(0, 1)})
    assert sim.snapshot().cars[0] == retired.cars[0]
    assert sim.snapshot().cars[1].state.speed_mps > 0
    with pytest.raises(ValueError):
        sim.retire_car("ai", CarStatus.FINISHED)


def test_duplicate_spawn_ids_and_empty_roster_are_rejected_atomically(demo_track):
    sim = make(demo_track, (CarSpawn("human", "solo"),))
    original = sim.snapshot()
    for spawns in ((), (CarSpawn("a", "solo"), CarSpawn("a", "solo"))):
        with pytest.raises(ValueError):
            sim.reset(demo_track, 0, spawns)
        assert sim.snapshot() is original


def test_three_lap_race_keeps_driving_without_environment_reset(demo_track):
    from racing.agents.scripted import ScriptedAgent
    from racing.race.session import RaceSession

    class PerCarDriver:
        def __init__(self, car_id):
            self.car_id = car_id
            self.agent = ScriptedAgent()
            self.resets = 0
            self.saw_second_lap = False

        def reset(self, seed):
            self.resets += 1

        def act(self, observation, snapshot, track):
            car = next(c for c in snapshot.cars if c.car_id == self.car_id)
            self.saw_second_lap |= car.completed_laps >= 1
            return self.agent.act(observation, replace(snapshot, cars=(car,)), track)

    drivers = {car_id: PerCarDriver(car_id) for car_id in ("ai", "human")}
    session = RaceSession(demo_track, controllers=drivers)
    session.start()
    for _ in range(60):
        session.countdown_tick()
    while session.state == "RUNNING":
        session.step()
    assert session.winner in ("human", "ai", "draw")
    assert max(c.completed_laps for c in session.snapshot().cars) == 3
    assert all(
        driver.saw_second_lap and driver.resets == 1 for driver in drivers.values()
    )
    assert len(session.results["ai"].lap_times_s) >= 2
