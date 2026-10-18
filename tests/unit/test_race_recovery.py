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
