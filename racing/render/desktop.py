"""Desktop session preferences shared across successive application routes."""

from dataclasses import dataclass


@dataclass
class DesktopPreferences:
    camera_mode: str = "overhead"
    difficulty: str = "medium"

    def __post_init__(self):
        if self.camera_mode not in ("overhead", "follow"):
            raise ValueError("camera_mode: overhead or follow required")
        if self.difficulty not in ("easy", "medium", "hard"):
            raise ValueError("difficulty: easy, medium or hard required")
