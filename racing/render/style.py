"""Shared daylight motorsport style and viewport geometry."""

import math
from dataclasses import dataclass

import pygame

PALETTE = {
    "grass": (82, 99, 72), "grass_dark": (70, 87, 61),
    "grass_light": (99, 114, 81), "asphalt": (48, 52, 56),
    "asphalt_dark": (41, 45, 49), "asphalt_light": (58, 62, 66),
    "ivory": (238, 234, 224), "red": (188, 68, 58),
    "accent": (242, 138, 53), "panel": (21, 27, 32),
    "secondary": (174, 184, 189), "headlight": (255, 243, 191),
}


@dataclass(frozen=True, slots=True)
class Camera:
    center: tuple[float, float]
    scale: float
    viewport: tuple[int, int]

    def to_screen(self, point):
        return tuple((point[i] - self.center[i]) * self.scale + self.viewport[i] / 2
                     for i in (0, 1))

    def to_world(self, point):
        return tuple((point[i] - self.viewport[i] / 2) / self.scale + self.center[i]
                     for i in (0, 1))


_fonts = {}


def font(size_px, bold=False):
    # pygame.quit invalidates Font objects; start a new cache after reinitialization.
    if not pygame.font.get_init():
        _fonts.clear()
        pygame.font.init()
    if not _fonts:
        # Fonts hold SDL pointers; discard them before pygame.quit destroys SDL.
        # Register again for each new session because quit consumes callbacks.
        pygame.register_quit(_fonts.clear)
    key = (size_px, bold)
    if key not in _fonts:
        face = pygame.font.match_font("dejavusans,verdana,arial", bold=bold)
        result = pygame.font.Font(face, size_px)
        if face is None:
            result.set_bold(bold)
        _fonts[key] = result
    return _fonts[key]


def layout(viewport, state, watch_mode=False):
    width, height = viewport
    compact = width < 800 or height < 560
    margin = 12 if compact else 24
    if compact:
        rects = {
            "lap": pygame.Rect(margin, margin, 202, 64),
            "timer": pygame.Rect(width - margin - 232, margin, 232, 84),
            "minimap": pygame.Rect(margin, height - 164, 116, 100),
            "speed": pygame.Rect(width - margin - 150, height - 164, 150, 100),
            "hint": pygame.Rect(margin, height - 30, width - 2 * margin, 24),
        }
    else:
        rects = {
            "lap": pygame.Rect(margin, margin, 248, 78),
            "timer": pygame.Rect(width - margin - 300, margin, 300, 112),
            "minimap": pygame.Rect(margin, height - 232, 196, 162),
            "speed": pygame.Rect(width - margin - 204, height - 218, 204, 148),
            "hint": pygame.Rect(margin, height - 32, width - 2 * margin, 24),
        }
    actions = {
        "MENU": ("reset", "quit") if watch_mode else ("reset", "watch", "race", "quit"), "PAUSED": ("resume", "reset", "menu"),
        "ATTEMPT_FAILED": ("reset", "menu"),
        "ATTEMPT_FINISHED": ("reset", "menu"), "DRIVING": (),
    }[state]
    if actions:
        panel_width = min(width - 24, 620)
        panel_height = min(height - 24, 332 if compact else 340)
        panel = pygame.Rect((width - panel_width) // 2, (height - panel_height) // 2,
                            panel_width, panel_height)
        rects["modal"] = panel
        if compact:
            for index, action in enumerate(actions):
                columns = 2 if len(actions) == 4 else 1
                rows = math.ceil(len(actions) / columns)
                button_width = (panel.width - 40 - (columns - 1) * 8) // columns
                rects[action] = pygame.Rect(panel.x + 20 + (index % columns) * (button_width + 8),
                    panel.bottom - 20 - rows * 48 + (index // columns) * 48,
                    button_width, 40)
        else:
            gap = 12
            action_width = (panel.width - 48 - gap * (len(actions) - 1)) // len(actions)
            for index, action in enumerate(actions):
                rects[action] = pygame.Rect(panel.x + 24 + index * (action_width + gap),
                                           panel.bottom - 78, action_width, 50)
    return rects
