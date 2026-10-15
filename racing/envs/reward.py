"""Pure reward terms; progress is the core's high-water delta."""

from dataclasses import dataclass

from racing.envs.config import RLConfig


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    progress: float = 0.0
    steering: float = 0.0
    step_cost: float = 0.0
    terminal: float = 0.0
    total: float = 0.0


class RewardCalculator:
    def __init__(self, config=None):
        self.config = config or RLConfig()

    def score(self, progress_delta, control, outcome) -> RewardBreakdown:
        config = self.config
        progress = float(
            config.progress_coefficient * min(2.0, max(0.0, progress_delta))
        )
        steering = float(-config.steering_penalty * abs(control.steering))
        terminal = float(
            config.off_track_penalty
            if outcome.reason == "off_track"
            else config.lap_bonus
            if outcome.success
            else 0.0
        )
        cost = float(-config.step_cost)
        return RewardBreakdown(
            progress, steering, cost, terminal, progress + steering + cost + terminal
        )
