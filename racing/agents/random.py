"""Uniform random baseline with an independent, explicitly seeded stream."""

import numpy as np

from racing.core.models import CarControl


class RandomAgent:
    def __init__(self):
        self._rng = None

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, observation, snapshot, track) -> CarControl:
        if self._rng is None:
            raise RuntimeError("reset(seed) must precede act")
        steering, longitudinal = self._rng.uniform(-1, 1, 2)
        return CarControl(float(steering), float(longitudinal))
