"""Pure episode precedence."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EpisodeOutcome:
    reason: str = "running"
    terminated: bool = False
    truncated: bool = False

    @property
    def success(self):
        return self.reason == "success"


class TerminationEvaluator:
    @staticmethod
    def check(off_track, laps, steps, no_progress, config) -> EpisodeOutcome:
        if off_track:
            return EpisodeOutcome("off_track", True, False)
        if laps >= config.target_laps:
            return EpisodeOutcome("success", True, False)
        if steps >= config.max_steps:
            return EpisodeOutcome("max_steps", False, True)
        if no_progress >= config.no_progress_steps:
            return EpisodeOutcome("no_progress", False, True)
        return EpisodeOutcome()
