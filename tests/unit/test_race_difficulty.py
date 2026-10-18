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
