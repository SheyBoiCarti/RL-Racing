from dataclasses import replace

import pytest

from racing.core.config import PhysicsConfig
from racing.core.models import CarControl, CarSpawn, CarStatus
from racing.core.simulation import Simulation


def simulation(track):
    sim = Simulation(PhysicsConfig())
    sim.reset(track, 0, (CarSpawn("human", "solo"),))
    return sim


@pytest.mark.parametrize("collection", [set, frozenset, list, tuple])
def test_hold_freezes_car_but_advances_world(demo_track, collection):
    sim = simulation(demo_track)
    before = sim.snapshot()
    result = sim.step({"human": CarControl(0.0, 0.0)}, held_ids=collection(["human"]))
    assert result.snapshot.tick == before.tick + 1
    assert result.snapshot.cars == before.cars
    assert result.events == ()


@pytest.mark.parametrize(
    "held,controls",
    [
        ("human", {"human": CarControl(0, 0)}),
        (b"human", {"human": CarControl(0, 0)}),
        ([1], {"human": CarControl(0, 0)}),
        ([""], {"human": CarControl(0, 0)}),
        (["unknown"], {"human": CarControl(0, 0)}),
        (["human"], {}),
        (["human"], {"human": CarControl(0, 1)}),
        (iter(["human"]), {"human": CarControl(0, 0)}),
    ],
)
def test_invalid_hold_is_atomic(demo_track, held, controls):
    sim = simulation(demo_track)
    before = sim.snapshot()
    with pytest.raises(ValueError):
        sim.step(controls, held_ids=held)
    assert sim.snapshot() == before


def test_terminal_hold_is_atomic(demo_track):
    sim = simulation(demo_track)
    before = sim.retire_car("human", CarStatus.TIMED_OUT)
    with pytest.raises(ValueError):
        sim.step({"human": CarControl(0, 0)}, held_ids=["human"])
    assert sim.snapshot() == before


def test_invalid_second_control_with_hold_is_atomic(demo_track):
    sim = Simulation(PhysicsConfig())
    before = sim.reset(
        demo_track, 0, (CarSpawn("human", "grid_left"), CarSpawn("ai", "grid_right"))
    )
    with pytest.raises(ValueError):
        sim.step(
            {"human": CarControl(0, 0), "ai": CarControl(0, float("nan"))},
            held_ids=["human"],
        )
    assert sim.snapshot() == before


def test_restoration_retains_history(demo_track):
    from racing.race.recovery import restored_snapshot

    safe = simulation(demo_track).snapshot().cars[0]
    safe = replace(
        safe,
        completed_laps=1,
        completed_lap_times_s=(30.0,),
        lap_start_tick=600,
        net_travel_m=610.0,
        high_water_m=620.0,
    )
    restored = restored_snapshot(safe, demo_track, PhysicsConfig())
    assert restored == replace(
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


@pytest.mark.parametrize(
    "kwargs",
    [{"recovery_hold_ticks": v} for v in (True, 0, 19, 21, 20.0)]
    + [{"version": "unknown"}, {"pace_adapter": "unknown"}, {"difficulty": "unknown"}],
)
def test_rules_reject_invalid_parameters(kwargs):
    from racing.race.config import RaceRules

    with pytest.raises(ValueError):
        RaceRules(**kwargs)


def crash(sim, monkeypatch):
    safe = sim.snapshot().cars[0]
    with monkeypatch.context() as patch:
        patch.setattr(
            "racing.core.simulation.integrate_car",
            lambda *args: replace(safe.state, x_m=1e6, y_m=1e6),
        )
        result = sim.step({"human": CarControl(0, 1)})
    assert result.snapshot.cars[0].status == CarStatus.DISQUALIFIED
    assert result.events[0].kind.value == "off_track"
    return safe


def test_real_offtrack_can_recover_without_advancing_time(demo_track, monkeypatch):
    sim = simulation(demo_track)
    safe = crash(sim, monkeypatch)
    before = sim.snapshot()
    recovered = sim.recover_car(safe, expected_tick=before.tick)
    assert recovered.tick == before.tick
    assert recovered.time_s == before.time_s
    assert recovered.cars[0] == replace(
        safe,
        state=replace(
            safe.state,
            heading_rad=safe.projection.tangent_heading,
            speed_mps=0.0,
            steering_rad=0.0,
        ),
    )


@pytest.mark.parametrize(
    "invalid", ["tick", "id", "history", "nan", "outside", "status"]
)
def test_recovery_rejection_is_atomic(demo_track, monkeypatch, invalid):
    sim = simulation(demo_track)
    safe = crash(sim, monkeypatch)
    before = sim.snapshot()
    tick = before.tick
    if invalid == "tick":
        tick -= 1
    elif invalid == "id":
        safe = replace(safe, car_id="unknown")
    elif invalid == "history":
        safe = replace(safe, completed_laps=1)
    elif invalid == "nan":
        safe = replace(safe, state=replace(safe.state, heading_rad=float("nan")))
    elif invalid == "outside":
        safe = replace(safe, state=replace(safe.state, x_m=1e6))
    else:
        safe = replace(safe, status=CarStatus.FINISHED)
    with pytest.raises(ValueError):
        sim.recover_car(safe, expected_tick=tick)
    assert sim.snapshot() == before


def test_recovery_requires_disqualified_car(demo_track):
    sim = simulation(demo_track)
    before = sim.snapshot()
    with pytest.raises(ValueError):
        sim.recover_car(before.cars[0], expected_tick=0)
    assert sim.snapshot() == before


def crash_race(race, monkeypatch, ids=("human", "ai")):
    original = {c.state: c.car_id for c in race.snapshot().cars}
    from racing.core.physics import integrate_car

    with monkeypatch.context() as patch:
        patch.setattr(
            "racing.core.simulation.integrate_car",
            lambda state, control, physics: (
                replace(state, x_m=state.x_m + 100000.0)
                if original[state] in ids
                else integrate_car(state, control, physics)
            ),
        )
        return race.step({key: CarControl(0, 0) for key in ("human", "ai")})


def test_recovery_exact_hold(recovery_race, monkeypatch):
    race = recovery_race
    crash_race(race, monkeypatch)
    trigger = race.snapshot().tick
    safe_pose = race.snapshot().cars
    assert all(r.remaining_ticks == 20 for r in race.recovery_by_car.values())
    for remaining in range(19, -1, -1):
        race.step({key: CarControl(0, 0) for key in ("human", "ai")})
        assert race.snapshot().cars == safe_pose
        assert all(
            r.remaining_ticks == remaining for r in race.recovery_by_car.values()
        )
        assert all(reason == "running" for reason in race.reason_by_car.values())
        assert race.no_progress_by_car == {"ai": 0, "human": 0}
    assert race.snapshot().tick == trigger + 20
    assert all(
        r.held_ticks == 20 and r.count == 1 and r.last_release_tick == 21
        for r in race.recovery_by_car.values()
    )
    assert race.results["human"].recovery_count == 1
    assert race.results["human"].recovery_held_ticks == 20
    race.step({key: CarControl(0, 1) for key in ("human", "ai")})
    assert all(car.state.speed_mps > 0 for car in race.snapshot().cars)


def test_recovery_pause_restart_and_atomic_controls(recovery_race, monkeypatch):
    race = recovery_race
    crash_race(race, monkeypatch, ids=("human",))
    before = race.snapshot()
    recovery = dict(race.recovery_by_car)
    with pytest.raises(ValueError):
        race.step({"human": CarControl(0, 1), "ai": CarControl(0, 1)})
    assert race.snapshot() == before and race.recovery_by_car == recovery
    race.pause()
    race.step()
    assert race.snapshot() == before and race.recovery_by_car == recovery
    race.resume()
    race.step({"human": CarControl(0, 0), "ai": CarControl(0, 1)})
    assert race.snapshot().cars[0].state.speed_mps > 0
    assert race.snapshot().cars[1].state == before.cars[1].state
    race.restart()
    assert race.snapshot().tick == 0
    assert all(
        r.count == r.held_ticks == r.remaining_ticks == 0
        and r.last_release_tick is None
        for r in race.recovery_by_car.values()
    )


def test_global_timeout_during_hold(demo_track, monkeypatch):
    from racing.race.config import RaceConfig, RaceRules
    from racing.race.session import RaceSession

    race = RaceSession(
        demo_track,
        rules=RaceRules(),
        config=RaceConfig(countdown_seconds=0, race_max_steps=2, no_progress_steps=1),
    )
    race.start()
    crash_race(race, monkeypatch)
    assert race.state == "RUNNING"
    race.step({key: CarControl(0, 0) for key in ("human", "ai")})
    assert race.state == "RESULTS"
    assert all(r.reason == "race_max_steps" for r in race.results.values())


def test_no_progress_restarts_after_hold(demo_track, monkeypatch):
    from racing.race.config import RaceConfig, RaceRules
    from racing.race.session import RaceSession

    race = RaceSession(
        demo_track,
        rules=RaceRules(),
        config=RaceConfig(countdown_seconds=0, no_progress_steps=2),
    )
    race.start()
    crash_race(race, monkeypatch)
    for _ in range(21):
        race.step({key: CarControl(0, 0) for key in ("human", "ai")})
    assert race.state == "RUNNING"
    assert race.no_progress_by_car == {"ai": 1, "human": 1}
    race.step({key: CarControl(0, 0) for key in ("human", "ai")})
    assert all(r.reason == "no_progress" for r in race.results.values())


def test_opponent_finish_during_recovery(recovery_race, monkeypatch):
    race = recovery_race
    world = race.snapshot()
    ai, human = world.cars
    race.simulation._world = replace(world, cars=(replace(ai, completed_laps=3), human))
    crash_race(race, monkeypatch, ids=("human",))
    assert race.winner == "ai" and race.state == "RESULTS"
    assert race.recovery_by_car["human"].remaining_ticks == 20
    assert race.results["human"].reason == "running"


def test_bad_recovery_emits_tick_only_and_diagnostic(recovery_race, monkeypatch):
    race = recovery_race
    events = []
    race.add_listener(lambda s, event, payload: events.append(event))

    def broken(*args, **kwargs):
        raise ValueError("invalid anchor")

    monkeypatch.setattr(race.simulation, "recover_car", broken)
    crash_race(race, monkeypatch)
    assert events == ["tick"]
    assert race.snapshot().tick == 1
    assert race.state == "RESULTS" and race.abort_reason == "recovery_error"
    assert "invalid anchor" in race.error_message
    assert all(r.reason == "recovery_error" for r in race.results.values())
    race.restart()
    assert race.error_message == ""


def test_recovery_error_abort_requires_rules(demo_track, recovery_race):
    from racing.race.config import RaceConfig
    from racing.race.session import RaceSession

    legacy = RaceSession(demo_track, config=RaceConfig(countdown_seconds=0))
    legacy.start()
    with pytest.raises(ValueError):
        legacy.abort("recovery_error")
    recovery_race.abort("recovery_error")
    assert recovery_race.abort_reason == "recovery_error"


@pytest.mark.parametrize(
    "history",
    [
        {},
        {
            "finish_armed": True,
            "next_checkpoint": 1,
            "net_travel_m": 4.0,
            "high_water_m": 4.0,
        },
        {
            "finish_armed": True,
            "next_checkpoint": 2,
            "net_travel_m": 100.0,
            "high_water_m": 100.0,
        },
        {
            "finish_armed": True,
            "next_checkpoint": 1,
            "completed_laps": 1,
            "completed_lap_times_s": (30.0,),
            "lap_start_tick": 600,
            "net_travel_m": 610.0,
            "high_water_m": 620.0,
        },
    ],
)
def test_recovery_preserves_lap_history(recovery_race, monkeypatch, history):
    race = recovery_race
    world = race.snapshot()
    cars = tuple(replace(car, **history) for car in world.cars)
    race.simulation._world = replace(world, cars=cars)
    race.recovery_by_car = {
        car.car_id: replace(race.recovery_by_car[car.car_id], safe=car) for car in cars
    }
    crash_race(race, monkeypatch)
    for safe, after in zip(cars, race.snapshot().cars, strict=True):
        assert after == replace(
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


def test_held_live_controllers_are_not_called(recovery_race, monkeypatch):
    from tests.unit.test_race_session import Driver

    race = recovery_race
    human, ai = Driver(), Driver()
    race.controllers = {"human": human, "ai": ai}
    crash_race(race, monkeypatch, ids=("human",))
    race.step()
    assert human.calls == []
    assert len(ai.calls) == 1
    assert race.last_controls["human"] == CarControl(0, 0)


def test_physics_driven_multilap_recovery_includes_held_time(demo_track):
    from racing.race.config import RaceConfig, RaceRules
    from racing.race.session import RaceSession
    from scripts.verify_race import ScriptedCar

    race = RaceSession(
        demo_track,
        rules=RaceRules(difficulty="hard"),
        config=RaceConfig(countdown_seconds=0, target_laps=2),
        controllers={key: ScriptedCar(key) for key in ("human", "ai")},
    )
    race.start()
    # Arm a real lap before the crash so its duration must include recovery.
    driver = race.controllers["human"]
    for _ in range(200):
        before = race.snapshot()
        race.step(
            {"human": driver.act(None, before, demo_track), "ai": CarControl(0, 0)}
        )
        if race.snapshot().cars[1].finish_armed:
            break
    lap_start = race.snapshot().cars[1].lap_start_tick
    assert lap_start is not None
    finish_ticks = []
    race.add_listener(
        lambda session, event, payload: (
            finish_ticks.extend(
                e.tick
                for e in payload["result"].events
                if e.car_id == "human" and e.kind.value == "lap_completed"
            )
            if event == "tick"
            else None
        )
    )
    # Real off-track driving with a committed safe anchor; no private world changes.
    for _ in range(200):
        race.step({"human": CarControl(1, 1), "ai": CarControl(0, 0)})
        if race.recovery_by_car["human"].count:
            break
    assert race.recovery_by_car["human"].count == 1
    trigger = race.snapshot().tick
    safe = race.snapshot()
    while any(r.remaining_ticks for r in race.recovery_by_car.values()):
        race.step({key: CarControl(0, 0) for key in ("human", "ai")})
    assert race.snapshot().tick == trigger + 20
    assert race.snapshot().cars == safe.cars
    while race.state == "RUNNING":
        race.step()
    assert race.winner in ("human", "ai", "draw")
    winner = "human" if race.winner == "draw" else race.winner
    assert race.results[winner].completed_laps == 2
    assert race.results["human"].completed_laps == 2
    assert race.results["human"].lap_times_s[0] == pytest.approx(
        (finish_ticks[0] - lap_start) * race.physics.dt_s
    )
    assert finish_ticks[0] > trigger + 20
    assert race.results["human"].recovery_held_ticks >= 20
