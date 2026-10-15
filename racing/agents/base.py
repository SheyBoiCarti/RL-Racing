from typing import Protocol

import numpy as np

from racing.core.models import CarControl, WorldSnapshot
from racing.tracks import Track


class Agent(Protocol):
    def reset(self, seed: int) -> None: ...
    def act(
        self, observation: np.ndarray, snapshot: WorldSnapshot, track: Track
    ) -> CarControl: ...
