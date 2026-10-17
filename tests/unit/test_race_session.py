from copy import copy
from dataclasses import FrozenInstanceError, replace

import pytest

from racing.core.config import PhysicsConfig
from racing.core.models import CarControl, CarStatus


def race(track, **kwargs):
    from racing.race.config import RaceConfig
    from racing.race.session import RaceSession

    config = kwargs.pop("config", RaceConfig(countdown_seconds=0))
    session = RaceSession(track, config=config, **kwargs)
    session.start()
    return session


def controls(ai=CarControl(0, 0), human=CarControl(0, 0)):
    return {"ai": ai, "human": human}


def test_config_validation_and_grid_swap(demo_track):
    from racing.race.config import RaceConfig
    from racing.race.session import RaceSession

    for values in (
        {"target_laps": True},
        {"countdown_seconds": float("nan")},
        {"race_max_steps": 0},
        {"no_progress_steps": 1.5},
        {"ghost": False},
        {"progress_threshold_m": -1},
    ):
        with pytest.raises(ValueError):
            RaceConfig(**values)
    config = RaceConfig()
    with pytest.raises(FrozenInstanceError):
        config.target_laps = 2
    normal = RaceSession(demo_track)
    swapped = RaceSession(demo_track, lane_swap=True)
    assert normal.snapshot().cars[0].state == swapped.snapshot().cars[1].state
    assert [(s.car_id, s.spawn_id) for s in normal.spawns] == [
        ("ai", "grid_right"),
        ("human", "grid_left"),
    ]
    bad_track = copy(demo_track)
    bad_track.spawns = {
        **demo_track.spawns,
        "grid_right": demo_track.spawns["grid_left"],
    }
    with pytest.raises(ValueError, match="grid"):
        RaceSession(bad_track)


class Driver:
    def __init__(self, action=CarControl(0, 1), error=False):
        self.action, self.error = action, error
        self.calls, self.resets = [], []

    def reset(self, seed):
        self.resets.append(seed)

    def act(self, observation, snapshot, track):
        self.calls.append((observation.copy(), snapshot))
        if self.error:
            raise RuntimeError("prediction failed")
        return self.action


def test_countdown_shared_snapshots_pause_and_controller_capture(demo_track):
    from racing.race.session import RaceSession

    ai, human = Driver(), Driver()
    session = RaceSession(demo_track, controllers={"ai": ai, "human": human}, seed=7)
    assert session.state == "MENU"
    session.start()
    for _ in range(59):
        session.countdown_tick()
        session.step()
        assert session.snapshot().tick == 0
        assert not ai.calls and not human.calls
    session.countdown_tick()
    assert session.state == "RUNNING"
    before = session.snapshot()
    session.step()
    assert ai.calls[0][1] is before and human.calls[0][1] is before
    assert session.snapshot().tick == 1
    session.pause()
    session.step()
    assert len(ai.calls) == len(human.calls) == 1
    session.resume()
    session.step()
    assert len(ai.calls) == len(human.calls) == 2


def test_explicit_controls_are_full_atomic_and_clipped(demo_track):
    session = race(demo_track)
    before = session.snapshot()
    for bad in (
        {"ai": CarControl(0, 0)},
        {**controls(), "bad": CarControl(0, 0)},
        controls(ai=CarControl(float("inf"), 0)),
        [],
    ):
        with pytest.raises(ValueError):
            session.step(bad)
        assert session.snapshot() is before
        assert session.state == "RUNNING" and session.last_controls == {}
        assert session.no_progress_by_car == {"ai": 0, "human": 0}
    session.step(controls(ai=CarControl(8, -5)))
    assert session.last_controls["ai"] == CarControl(1.0, -1.0)


@pytest.mark.parametrize(
    "action,error", [(CarControl(0, 0), True), (CarControl(float("nan"), 0), False)]
)
def test_inference_failure_aborts_without_applying_partial_tick(
    demo_track, action, error
):
    session = race(
        demo_track, controllers={"ai": Driver(action, error), "human": Driver()}
    )
    before = session.snapshot()
    session.step()
    assert session.snapshot().tick == before.tick == 0
    assert session.state == "RESULTS" and session.winner == "no_winner"
    assert session.abort_reason == "inference_error" and session.last_controls == {}
    assert all(c.status == CarStatus.ABORTED for c in session.snapshot().cars)
    assert [c.state for c in before.cars] == [c.state for c in session.snapshot().cars]


@pytest.mark.parametrize(
    "limit,stall,reason", [(3, 10, "race_max_steps"), (10, 3, "no_progress")]
)
def test_exact_timeout_thresholds(demo_track, limit, stall, reason):
    from racing.race.config import RaceConfig

    session = race(
        demo_track,
        config=RaceConfig(
            countdown_seconds=0, race_max_steps=limit, no_progress_steps=stall
        ),
    )
    session.step(controls())
    session.step(controls())
    assert session.state == "RUNNING"
    session.step(controls())
    assert session.snapshot().tick == 3 and session.state == "RESULTS"
    assert session.winner == "no_winner"
    assert all(
        r.status == CarStatus.TIMED_OUT and r.reason == reason
        for r in session.results.values()
    )


def prepare_finish(session, ids):
    # Fixture at the reducer boundary: no synthetic event ordering determines a winner.
    world = session.snapshot()
    session.simulation._world = replace(
        world,
        cars=tuple(
            replace(c, completed_laps=3, completed_lap_times_s=(1.0, 1.1, 1.2))
            if c.car_id in ids
            else c
            for c in world.cars
        ),
    )


def test_whole_tick_finish_draw_and_retirement(demo_track):
    session = race(demo_track)
    prepare_finish(session, {"human", "ai"})
    session.step(controls())
    assert session.winner == "draw" and session.state == "RESULTS"
    assert session.finish_tick_by_car == {"ai": 1, "human": 1}
    assert all(c.status == CarStatus.FINISHED for c in session.snapshot().cars)
    assert session.results["ai"].finish_time_s == PhysicsConfig().dt_s
    assert session.results["ai"].best_lap_s == 1.0


def test_disqualified_survivor_must_finish_and_off_track_precedes_finish(demo_track):
    session = race(demo_track)
    world = session.snapshot()
    ai, human = world.cars
    ai = replace(ai, state=replace(ai.state, x_m=1e6), completed_laps=3)
    session.simulation._world = replace(world, cars=(ai, human))
    session.step(controls())
    assert session.state == "RUNNING" and session.winner == "no_winner"
    assert session.results["ai"].reason == "off_track"
    assert session.finish_tick_by_car["ai"] is None
    prepare_finish(session, {"human"})
    session.step({"human": CarControl(0, 0)})
    assert session.winner == "human"


def test_all_disqualified_no_winner(demo_track):
    session = race(demo_track)
    world = session.snapshot()
    session.simulation._world = replace(
        world,
        cars=tuple(replace(c, state=replace(c.state, x_m=1e6)) for c in world.cars),
    )
    session.step(controls())
    assert session.state == "RESULTS" and session.winner == "no_winner"


def test_listeners_restart_and_paused_abort(demo_track):
    from racing.race.config import RaceConfig
    from racing.race.session import RaceSession

    driver = Driver()
    session = RaceSession(
        demo_track,
        config=RaceConfig(countdown_seconds=0),
        controllers={"human": driver},
        seed=4,
    )
    events = []
    session.add_listener(
        lambda s, event, payload: events.append(
            (event, s.state, s.snapshot().tick, payload)
        )
    )
    session.start()
    session.step(controls())
    session.pause()
    session.abort("user_exit")
    assert [(e[0], e[1], e[2]) for e in events] == [
        ("start", "RUNNING", 0),
        ("tick", "RUNNING", 1),
        ("pause", "PAUSED", 1),
        ("abort", "RESULTS", 1),
    ]
    session.restart()
    assert session.state == "RUNNING" and session.snapshot().tick == 0
    assert session.last_controls == {} and session.abort_reason is None
    assert session.no_progress_by_car == {"ai": 0, "human": 0}
    assert session.finish_tick_by_car == {"ai": None, "human": None}
    assert driver.resets == [4, 4]


@pytest.mark.parametrize("as_pairs", [False, True])
def test_human_adapter_captures_input_and_reset_clears_it(as_pairs):
    from racing.race.controllers import HumanController

    bindings = {
        "left": ("a",),
        "right": ("d",),
        "accelerate": ("w",),
        "brake": ("s",),
        "brake_override": ("space",),
    }
    driver = HumanController(tuple(bindings.items()) if as_pairs else bindings)
    held = {"w", "d"}
    driver.capture(held)
    held.clear()
    assert driver.act(None, None, None) == CarControl(1, 1)
    driver.reset(0)
    assert driver.act(None, None, None) == CarControl(0, 0)


def test_zero_progress_does_not_reset_stall_at_zero_threshold(demo_track):
    from racing.race.config import RaceConfig

    session = race(
        demo_track,
        config=RaceConfig(
            countdown_seconds=0, progress_threshold_m=0, no_progress_steps=2
        ),
    )
    session.step(controls())
    session.step(controls())
    assert session.state == "RESULTS"
    assert session.no_progress_by_car == {"ai": 2, "human": 2}


def test_lap_boundary_does_not_reset_no_progress_and_finish_beats_timeout(demo_track):
    from racing.race.config import RaceConfig

    session = race(
        demo_track,
        config=RaceConfig(countdown_seconds=0, no_progress_steps=3, race_max_steps=3),
    )
    session.step(controls())
    world = session.snapshot()
    session.simulation._world = replace(
        world, cars=tuple(replace(c, completed_laps=1) for c in world.cars)
    )
    session.step(controls())
    assert session.no_progress_by_car == {"ai": 2, "human": 2}
    prepare_finish(session, {"human"})
    session.step(controls())
    assert session.winner == "human"
    assert session.results["human"].status == CarStatus.FINISHED
    assert session.results["ai"].reason == "race_max_steps"


def test_pause_and_abort_reject_invalid_transitions_without_tick(demo_track):
    session = race(demo_track)
    with pytest.raises(ValueError):
        session.abort("unknown_reason")
    assert session.state == "RUNNING"
    with pytest.raises(ValueError):
        session.resume()
    session.pause()
    with pytest.raises(ValueError):
        session.pause()
    with pytest.raises(ValueError):
        session.step(controls())
    session.abort("restart")
    for operation in (
        session.pause,
        session.resume,
        lambda: session.abort("user_exit"),
    ):
        with pytest.raises(ValueError):
            operation()
    assert session.snapshot().tick == 0


def test_ranking_uses_net_distance_with_ties_and_terminal_status(demo_track):
    session = race(demo_track)
    world = session.snapshot()
    ai, human = world.cars
    session.simulation._world = replace(
        world,
        cars=(replace(ai, net_travel_m=10), replace(human, net_travel_m=10.0000005)),
    )
    assert session.position_by_car == {"ai": 1, "human": 1}
    assert session.gap_m == pytest.approx(-0.0000005)
    session.simulation._world = replace(
        world, cars=(replace(ai, net_travel_m=20, status=CarStatus.DISQUALIFIED), human)
    )
    assert session.position_by_car == {"ai": 2, "human": 1}


def test_grid_rejects_moving_or_misoriented_spawn(demo_track):
    from racing.race.session import RaceSession

    for values in ({"heading_rad": 0.0}, {"speed_mps": 1.0}, {"steering_rad": 0.1}):
        track = copy(demo_track)
        track.spawns = {
            **demo_track.spawns,
            "grid_left": replace(demo_track.spawns["grid_left"], **values),
        }
        with pytest.raises(ValueError, match="grid"):
            RaceSession(track)
