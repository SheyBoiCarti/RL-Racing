import pytest

from racing.core.models import CarControl


@pytest.mark.parametrize(
    "difficulty,speed,expected",
    [
        ("easy", 13.1, 1.0),
        ("easy", 13.2, 0.0),
        ("easy", 14.2, -0.5),
        ("easy", 15.2, -1.0),
        ("medium", 17.5, 1.0),
        ("medium", 17.6, 0.0),
        ("medium", 18.6, -0.5),
        ("medium", 19.6, -1.0),
    ],
)
def test_governor_boundaries(difficulty, speed, expected):
    from racing.race.difficulty import govern_ai

    result = govern_ai(CarControl(0.3, 1.0), speed, 22.0, difficulty)
    assert result.steering == 0.3
    assert result.longitudinal == pytest.approx(expected)


def test_governor_retains_stronger_braking_and_hard_identity():
    from racing.race.difficulty import govern_ai

    control = CarControl(0.3, -0.8)
    assert govern_ai(control, 14.2, 22.0, "easy").longitudinal == -0.8
    assert govern_ai(control, 22.0, 22.0, "hard") is control


@pytest.mark.parametrize(
    "speed,maximum,tier,control",
    [
        (float("nan"), 22.0, "easy", CarControl(0, 1)),
        (1.0, float("inf"), "easy", CarControl(0, 1)),
        (-1.0, 22.0, "easy", CarControl(0, 1)),
        (1.0, 0.0, "easy", CarControl(0, 1)),
        (1.0, 22.0, "unknown", CarControl(0, 1)),
        (1.0, 22.0, "hard", CarControl(float("nan"), 1)),
        (True, 22.0, "easy", CarControl(0, 1)),
    ],
)
def test_governor_rejects_invalid_inputs(speed, maximum, tier, control):
    from racing.race.difficulty import govern_ai

    with pytest.raises(ValueError):
        govern_ai(control, speed, maximum, tier)


def test_live_governor_and_explicit_actions_are_distinct(demo_track):
    from dataclasses import replace

    from racing.race.config import RaceConfig, RaceRules
    from racing.race.session import RaceSession
    from tests.unit.test_race_session import Driver

    ai, human = Driver(CarControl(0.3, 1)), Driver()
    race = RaceSession(
        demo_track,
        config=RaceConfig(countdown_seconds=0),
        rules=RaceRules(difficulty="easy"),
        controllers={"ai": ai, "human": human},
    )
    race.start()
    world = race.snapshot()
    race.simulation._world = replace(
        world,
        cars=tuple(
            replace(c, state=replace(c.state, speed_mps=14.2))
            if c.car_id == "ai"
            else c
            for c in world.cars
        ),
    )
    race.step()
    assert race.last_controls["ai"].longitudinal == pytest.approx(-0.5)
    assert race.last_controls["ai"].steering == 0.3
    race.step({"ai": CarControl(0.3, 1), "human": CarControl(0, 1)})
    assert race.last_controls["ai"] == CarControl(0.3, 1)
