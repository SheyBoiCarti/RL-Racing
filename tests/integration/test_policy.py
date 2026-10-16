"""Watch mode must use the exact environment observation and tick cadence."""

import numpy as np
import pytest

from racing.core.config import load_game
from racing.core.models import CarControl
from racing.envs.config import RLConfig
from racing.envs.racing_env import RacingEnv


def test_watch_observations_and_termination_match_environment():
    from racing.render.watch import WatchApplication

    class RecordingController:
        metadata = {"checkpoint_step": 64}

        def __init__(self):
            self.observations = []

        def reset(self, seed):
            pass

        def act(self, observation, snapshot, track):
            self.observations.append(observation.copy())
            return CarControl(.1, .4)

    config = RLConfig(max_steps=8)
    controller = RecordingController()
    app = WatchApplication(load_game(), controller, config=config)
    direct = RacingEnv(config=config)
    observation, _ = direct.reset(seed=9000)
    app.action("reset")
    for _ in range(8):
        app.frame(.05, None)
        assert np.array_equal(controller.observations[-1], observation)
        observation, _, terminated, truncated, _ = direct.step(np.asarray([.1, .4], dtype=np.float32))
    assert truncated and not terminated
    assert app.simulation.snapshot() == direct.snapshot
    assert app.state.value == "ATTEMPT_FAILED"
    tick = app.simulation.snapshot().tick
    app.frame(1., None)
    assert app.simulation.snapshot().tick == tick
    app.env.close()
    direct.close()


def test_policy_refuses_nonfinite_prediction():
    from racing.agents.policy import PolicyController

    class BrokenModel:
        def predict(self, *args, **kwargs):
            return np.array([np.nan, 0.]), None

    controller = PolicyController.__new__(PolicyController)
    controller.model = BrokenModel()
    with pytest.raises(ValueError, match="prediction"):
        controller.act(np.zeros(11, dtype=np.float32))
