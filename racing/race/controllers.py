"""Race controller adapters; keyboard capture has no display dependency."""

from racing.core.models import CarControl
from racing.core.physics import validate_control
from racing.render.input import held_control

__all__ = ["HumanController"]


class HumanController:
    def __init__(self, bindings=None):
        self.bindings = tuple(
            (key, tuple(values)) for key, values in dict(bindings or {}).items()
        )
        self.reset(0)

    def capture(self, held):
        self._held = frozenset(held)
        self._control = None

    def set_control(self, control):
        validate_control(control)
        self._control = control

    def reset(self, seed):
        self._held = frozenset()
        self._control = CarControl(0.0, 0.0)

    def act(self, observation, snapshot, track):
        if self._control is not None:
            return self._control
        return held_control(self._held, self.bindings)
