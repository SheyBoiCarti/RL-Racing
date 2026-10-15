import pytest

from racing.core.models import CarControl
from racing.envs.config import RLConfig
from racing.envs.reward import RewardCalculator
from racing.envs.termination import TerminationEvaluator


@pytest.mark.parametrize(
    "off,laps,steps,idle,reason,terminated,truncated",
    [
        (False, 0, 1199, 119, "running", False, False),
        (False, 0, 1200, 120, "max_steps", False, True),
        (False, 0, 1, 120, "no_progress", False, True),
        (False, 1, 1200, 120, "success", True, False),
        (True, 1, 1200, 120, "off_track", True, False),
    ],
)
def test_precedence(off, laps, steps, idle, reason, terminated, truncated):
    outcome = TerminationEvaluator.check(off, laps, steps, idle, RLConfig())
    assert (outcome.reason, outcome.terminated, outcome.truncated) == (
        reason,
        terminated,
        truncated,
    )


def test_numeric_reward():
    calculator = RewardCalculator()
    running = TerminationEvaluator.check(False, 0, 1, 0, RLConfig())
    assert calculator.score(0.5, CarControl(0.2, 1), running).total == pytest.approx(
        0.996
    )
    off = TerminationEvaluator.check(True, 0, 1, 0, RLConfig())
    assert calculator.score(0.5, CarControl(0.2, 1), off).total == pytest.approx(-9.004)
    assert calculator.score(0, CarControl(0, 0), running).progress == 0


def test_default_reward_config_matches_shared_schema():
    import json
    from dataclasses import asdict

    from jsonschema import Draft202012Validator

    from racing.core.config import project_path

    common = json.loads(
        project_path("docs/specs/schemas/common.schema.json").read_text()
    )
    schema = common["$defs"]["env_config"]
    values = asdict(RLConfig())
    config = {key: values[key] for key in schema["required"]}
    Draft202012Validator(schema).validate(config)
    assert config["off_track_penalty"] == -10.0


@pytest.mark.parametrize(
    "kwargs", [{"off_track_penalty": 1.0}, {"progress_coefficient": 0.0}]
)
def test_reward_config_rejects_schema_incompatible_signs(kwargs):
    with pytest.raises(ValueError):
        RLConfig(**kwargs)
