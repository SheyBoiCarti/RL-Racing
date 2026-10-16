"""Validated deterministic inference over the shared observation/action contract."""

import numpy as np

from racing.core.models import CarControl
from racing.training.checkpoints import validate_metadata


class PolicyController:
    def __init__(self, path, config=None):
        from stable_baselines3 import PPO

        self.metadata = validate_metadata(path, config)
        self.training_seed = self.metadata["training_seed"]
        self.model = PPO.load(path, device="cpu")
        self.config = {}

    def reset(self, seed):
        pass

    def act(self, observation, snapshot=None, track=None):
        action, _ = self.model.predict(observation, deterministic=True)
        action = np.asarray(action)
        if action.shape != (2,) or action.dtype.kind not in "fiu" or not np.isfinite(action).all():
            raise ValueError("policy prediction: finite numeric shape (2,) required")
        action = np.clip(action, -1, 1)
        return CarControl(float(action[0]), float(action[1]))
