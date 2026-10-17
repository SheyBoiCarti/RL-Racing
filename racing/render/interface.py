"""Race readouts, responsive overlays and player actions."""

import math
from dataclasses import dataclass

import pygame

from racing.render.style import PALETTE as P
from racing.render.style import Camera, font, layout


def format_time(seconds):
    if seconds is None:
        return "—"
    hundredths = round(seconds * 100)
    minutes, remainder = divmod(hundredths, 6000)
    whole, fraction = divmod(remainder, 100)
    return f"{minutes:02}:{whole:02}.{fraction:02}"


@dataclass(frozen=True, slots=True)
class RaceReadout:
    lap: str
    phase: str
    time: str
    best: str
    speed: str


def race_readout(snapshot, physics, best_lap):
    car = snapshot.cars[0]
    elapsed = car.completed_lap_times_s[-1] if car.completed_lap_times_s else (
        (snapshot.tick - car.lap_start_tick) * physics.dt_s
        if car.lap_start_tick is not None else None)
    return RaceReadout("01 / 01", "Reach the start line" if car.lap_start_tick is None else "Timed lap",
                       format_time(elapsed), format_time(best_lap),
                       f"{round(car.state.speed_mps * 3.6):03}")


class RaceInterface:
    primary_font_px = 18
    secondary_font_px = 14

    def __init__(self, game, physics):
        self.game, self.physics = game, physics
        self.watch_mode = False

    def _text(self, surface, text, position, size=18, color=None, bold=False):
        image = font(size, bold).render(text, True, color or P["ivory"])
        surface.blit(image, position)
        return image.get_size()

    def _panel(self, surface, rect):
        panel = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(panel, (*P["panel"], 230), panel.get_rect(), border_radius=10)
        surface.blit(panel, rect)

    def _minimap(self, surface, track, snapshot, rect):
        low, high = track.outer.min(axis=0), track.outer.max(axis=0)
        camera = Camera(tuple((low + high) / 2), min((rect.width - 20) / (high[0] - low[0]),
                         (rect.height - 20) / (high[1] - low[1])), rect.size)

        def screen(point):
            x, y = camera.to_screen(point)
            return x + rect.x, y + rect.y

        pygame.draw.polygon(surface, P["secondary"], [screen(p) for p in track.outer])
        pygame.draw.polygon(surface, P["panel"], [screen(p) for p in track.inner])
        _, _, a, b = track.gates[0]
        pygame.draw.line(surface, P["ivory"], screen(a), screen(b), 2)
        state = snapshot.cars[0].state
        x, y = screen((state.x_m, state.y_m))
        c, s = math.cos(state.heading_rad), math.sin(state.heading_rad)
        vertices = [(x + px * c - py * s, y + px * s + py * c)
                    for px, py in ((7, 0), (-5, -4), (-3, 0), (-5, 4))]
        pygame.draw.polygon(surface, P["accent"], vertices)

    def draw(self, surface, track, snapshot, state, best_lap, elapsed_s, pointer=None):
        rects = layout(surface.get_size(), state, self.watch_mode)
        readout = race_readout(snapshot, self.physics, best_lap)
        compact = surface.get_width() < 800 or surface.get_height() < 560
        if state != "MENU":
            for key in ("lap", "timer", "speed", "minimap"):
                self._panel(surface, rects[key])
            r = rects["lap"]
            self._text(surface, f"LAP {readout.lap}", (r.x + 12, r.y + 8), 18, bold=True)
            self._text(surface, readout.phase, (r.x + 12, r.y + 36), 14, P["secondary"])
            r = rects["timer"]
            self._text(surface, "LAP TIME", (r.x + 12, r.y + 6), 14, P["secondary"])
            self._text(surface, readout.time, (r.x + 12, r.y + 24), 26 if compact else 36, bold=True)
            self._text(surface, f"BEST  {readout.best}", (r.x + 12, r.bottom - 24), 14, P["secondary"])
            r = rects["speed"]
            self._text(surface, readout.speed, (r.x + 12, r.y + 2), 40 if compact else 64, bold=True)
            self._text(surface, "km/h", (r.x + 14, r.bottom - 42), 14, P["secondary"])
            bar = pygame.Rect(r.x + 14, r.bottom - 16, r.width - 28, 4)
            pygame.draw.rect(surface, P["asphalt_light"], bar)
            fraction = min(1., max(0., snapshot.cars[0].state.speed_mps / self.physics.max_speed_mps))
            pygame.draw.rect(surface, P["accent"], (bar.x, bar.y, round(bar.width * fraction), bar.height))
            self._minimap(surface, track, snapshot, rects["minimap"])
            bindings = dict(self.game.bindings)
            hint = " · ".join(f"{bindings[action][0].upper()} {label}" for action, label in
                              (("pause", "pause"), ("reset", "restart"), ("debug", "debug")))
            self._text(surface, hint, rects["hint"].topleft, 14, P["ivory"])
        if state == "DRIVING":
            return {}
        shade = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 120))
        surface.blit(shade, (0, 0))
        panel = rects["modal"]
        self._panel(surface, panel)
        titles = {"MENU": "RL RACING", "PAUSED": "Paused", "ATTEMPT_FAILED": "Off track",
                  "ATTEMPT_FINISHED": "Lap complete"}
        self._text(surface, titles[state], (panel.x + 20, panel.y + 16),
                   32 if compact else 46, P["accent"], True)
        messages = {"MENU": [f"{track.name} · {track.length_m:.0f} m"], "PAUSED": ["Your lap is on hold."],
                    "ATTEMPT_FAILED": ["Keep the whole car inside", "the white road edges."]}
        if state == "ATTEMPT_FINISHED":
            self._text(surface, readout.time, (panel.x + 20, panel.y + 66), 32, bold=True)
            self._text(surface, f"SESSION BEST  {readout.best}", (panel.x + 20, panel.y + 112),
                       14, P["secondary"])
        else:
            for index, message in enumerate(messages[state]):
                self._text(surface, message, (panel.x + 20, panel.y + 68 + 24 * index), 18)
        if state == "MENU":
            bindings = dict(self.game.bindings)
            controls = [f"{ '/'.join(bindings['accelerate']).upper()} accelerate · "
                        f"{ '/'.join(bindings['brake']).upper()} brake",
                        f"{bindings['left'][0].upper()} / {bindings['right'][0].upper()} steer · "
                        f"{bindings['brake_override'][0].upper()} brake · C choose track"]
            for index, text in enumerate(controls):
                self._text(surface, text, (panel.x + 20, panel.y + 105 + index * 23), 14, P["secondary"])
        choices = {"MENU": [("reset", "Drive [Enter]"), ("watch", "Watch AI"), ("race", "Race AI"), ("quit", "Quit [Q]")],
                   "PAUSED": [("resume", "Resume  [Enter]"), ("reset", "Restart"), ("menu", "Menu")],
                   "ATTEMPT_FAILED": [("reset", "Restart"), ("menu", "Menu")],
                   "ATTEMPT_FINISHED": [("reset", "Restart"), ("menu", "Menu")]}[state]
        if self.watch_mode and state == "MENU":
            choices = [("reset", "Watch AI  [Enter]"), ("quit", "Quit  [Q]")]
        actions = {}
        for index, (action, label) in enumerate(choices):
            rect = rects[action]
            hovered = pointer is not None and rect.collidepoint(pointer)
            color = P["accent"] if index == 0 else P["asphalt_light"]
            if hovered:
                color = tuple(min(255, channel + 20) for channel in color)
            pygame.draw.rect(surface, color, rect, border_radius=6)
            if index == 0:
                pygame.draw.rect(surface, P["ivory"], rect, 1, border_radius=6)
            image = font(18, index == 0).render(label, True, P["panel"] if index == 0 else P["ivory"])
            surface.blit(image, image.get_rect(center=rect.center))
            actions[action] = rect
        return actions
