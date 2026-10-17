"""Two-car race presentation, compact layouts and an offscreen opponent cue."""

import math

import pygame

from racing.render.interface import format_time
from racing.render.scene import TrackScene
from racing.render.style import PALETTE as P
from racing.render.style import Camera, font

AI_COLOR = (102, 195, 232)


def opponent_indicator(camera, human, ai):
    """Screen-edge position and angle, or None when the opponent is visible."""
    point = camera.to_screen((ai.x_m, ai.y_m))
    width, height = camera.viewport
    if 24 <= point[0] <= width - 24 and 130 <= point[1] <= height - 60:
        return None
    origin = camera.to_screen((human.x_m, human.y_m))
    dx, dy = point[0] - origin[0], point[1] - origin[1]
    if abs(dx) + abs(dy) < 1e-9:
        return None
    limits = []
    if dx:
        limits.append(((width - 28 if dx > 0 else 28) - origin[0]) / dx)
    if dy:
        limits.append(((height - 65 if dy > 0 else 135) - origin[1]) / dy)
    fraction = max(0, min(v for v in limits if v >= 0))
    return (origin[0] + dx * fraction, origin[1] + dy * fraction), math.atan2(dy, dx)


class RaceView:
    def __init__(self, surface, game, physics):
        self.surface, self.game, self.physics = surface, game, physics
        self.scene = TrackScene(physics)
        self.buttons, self.car_labels = {}, {}

    def _text(self, text, pos, size=17, color=None, width=None):
        if width is not None:
            while len(text) > 3 and font(size).size(text)[0] > width:
                text = text[:-4] + "…"
        self.surface.blit(font(size).render(text, True, color or P["ivory"]), pos)

    def _buttons(self, actions):
        self.buttons = {}
        width, height = self.surface.get_size()
        gap = 6
        columns = min(4, len(actions))
        if not columns:
            return
        button_width = (width - 24 - gap * (columns - 1)) // columns
        rows = math.ceil(len(actions) / columns)
        for index, (action, label) in enumerate(actions):
            rect = pygame.Rect(12 + (index % columns) * (button_width + gap),
                               height - 12 - rows * 40 + (index // columns) * 40,
                               button_width, 34)
            pygame.draw.rect(self.surface, P["panel"], rect, border_radius=5)
            pygame.draw.rect(self.surface, P["accent"], rect, 1, border_radius=5)
            rendered = font(13 if width < 800 else 17).render(label, True, P["ivory"])
            self.surface.blit(rendered, rendered.get_rect(center=rect.center))
            self.buttons[action] = rect

    def draw(self, app):
        snapshot = app.display_snapshot
        cars = {car.car_id: car for car in snapshot.cars}
        human, ai = cars["human"], cars["ai"]
        width, height = self.surface.get_size()
        compact = width < 800 or height < 560
        self.physics = app.physics
        self.scene.physics = app.physics
        camera = Camera((human.state.x_m, human.state.y_m), self.game.pixels_per_metre, (width, height))
        self.scene.draw(self.surface, app.track, snapshot, camera, app.debug)
        self.car_labels = {}
        for car_id, car in cars.items():
            point = camera.to_screen((car.state.x_m, car.state.y_m))
            label = "YOU" if car_id == "human" else "AI"
            self.car_labels[car_id] = point
            self._text(label, (point[0] - 12, point[1] - 28), 14,
                       P["accent"] if car_id == "human" else AI_COLOR)
        indicator = opponent_indicator(camera, human.state, ai.state)
        if indicator:
            point, angle = indicator
            vertices = [(point[0] + math.cos(angle + offset) * radius,
                         point[1] + math.sin(angle + offset) * radius)
                        for offset, radius in ((0, 13), (2.5, 9), (-2.5, 9))]
            pygame.draw.polygon(self.surface, AI_COLOR, vertices)
            self._text("AI", (point[0] - 8, point[1] + 12), 14, AI_COLOR)

        pygame.draw.rect(self.surface, P["panel"], (0, 0, width, 112 if compact else 124))
        size, line = (14, 23) if compact else (19, 27)
        lap = min(app.config.target_laps, human.completed_laps + 1)
        lap_time = ((snapshot.tick - human.lap_start_tick) * app.physics.dt_s
                    if human.lap_start_tick is not None else 0)
        self._text(f"YOU  {human.state.speed_mps * 3.6:.0f} km/h   Lap {lap}/{app.config.target_laps}"
                   f"   Lap time {format_time(lap_time)}", (12, 7), size, width=width - 24)
        self._text(f"Race {format_time(snapshot.time_s)}   {app.track.name}", (12, 7 + line), size,
                   width=width - 24)
        labels = []
        for car_id in ("human", "ai"):
            status = cars[car_id].status.value
            label = "DQ" if status == "disqualified" else "timeout" if status == "timed_out" else status
            labels.append(f"{'YOU' if car_id == 'human' else 'AI'} P{app.display_positions[car_id]} {label}")
        self._text("    ".join(labels), (12, 7 + line * 2), size, width=width - 24)
        lap_difference = ai.completed_laps - human.completed_laps
        gap = app.display_gap
        self._text(f"AI {'ahead' if gap >= 0 else 'behind'} {abs(gap):.1f} m   "
                   f"Lap difference {lap_difference:+d}", (12, 7 + line * 3), size,
                   color=AI_COLOR, width=width - 24)

        if app.debug:
            self._debug(app, camera, cars)
        actions = []
        if app.state == "MENU":
            panel = pygame.Rect(12, 130, width - 24, max(110, height - 238))
            pygame.draw.rect(self.surface, P["panel"], panel, border_radius=6)
            self._text("HUMAN vs AI · GHOST RACE", (24, 140), 20)
            self._text(app.model_label, (24, 174), 14, width=width - 48)
            self._text(f"{app.config.target_laps} laps · YOU {'right' if app.lane_swap else 'left'} grid · no contact",
                       (24, 198), 14, color=P["secondary"], width=width - 48)
            actions = [("start", "Race AI"), ("choose_track", "Track [C]"),
                       ("model_prev", "Previous model ["), ("model_next", "Next model ]"),
                       ("swap_lanes", "Swap lanes [L]"), ("quit", "Quit [Q]")]
        elif app.state == "COUNTDOWN":
            number = math.ceil(app.session.countdown_ticks_remaining / 20)
            rendered = font(92, True).render(str(number), True, P["ivory"])
            self.surface.blit(rendered, rendered.get_rect(center=(width / 2, height / 2)))
            actions = [("menu", "Cancel")]
        elif app.state == "PAUSED":
            self._text("PAUSED", (width // 2 - 48, height // 2 - 45), 28)
            actions = [("resume", "Resume [Enter]"), ("restart", "Restart [R]"), ("menu", "Menu [Esc]")]
        elif app.state == "RESULTS":
            self._results(app, compact)
            actions = [("restart", "Race again [R]"), ("replay", "Replay [P]"), ("menu", "Menu [Esc]"), ("quit", "Quit")]
        elif app.state == "REPLAY":
            self._text("REPLAY · PAUSED" if app.replay_paused else "REPLAY · VERIFIED", (12, height - 85), 19,
                       color=AI_COLOR)
            actions = [("resume" if app.replay_paused else "pause", "Resume" if app.replay_paused else "Pause [Esc]"),
                       ("menu", "Menu"), ("quit", "Quit")]
        elif app.state == "RUNNING":
            self._text("WASD / arrows · Space brake · Esc pause · R restart · Tab sensors", (12, height - 26),
                       12 if compact else 16)
        self._buttons(actions)
        if app.error:
            self._text(app.error, (16, height - (104 if app.state == "MENU" else 94)), 13,
                       color=(255, 154, 137), width=width - 32)

    def _results(self, app, compact):
        width, height = self.surface.get_size()
        top = 123 if compact else 147
        pygame.draw.rect(self.surface, P["panel"], (12, top, width - 24, height - top - 60), border_radius=6)
        winner = app.session.winner
        title = {"human": "YOU WIN", "ai": "AI WINS", "draw": "DRAW", "no_winner": "NO WINNER"}[winner]
        self._text(title, (24, top + 7), 21 if compact else 28)
        line = 19 if compact else 25
        y = top + (36 if compact else 46)
        for car_id in ("human", "ai"):
            result = app.display_results[car_id]
            finish = f"{result.finish_time_s:.2f}s (tick {result.finish_tick})" if result.finish_tick is not None else "—"
            best = f"{result.best_lap_s:.2f}s" if result.best_lap_s is not None else "—"
            self._text(f"{car_id.upper()} · {result.status.value} ({result.reason}) · {result.completed_laps} laps",
                       (24, y), 13 if compact else 17, width=width - 48)
            self._text(f"Finish {finish} · Best {best} · Laps " +
                       (", ".join(f"{lap:.2f}s" for lap in result.lap_times_s) or "—"),
                       (24, y + line), 12 if compact else 15, width=width - 48)
            y += line * 2 + 5
        self._text(f"Model: {app.model_label}", (24, y + 4), 12 if compact else 14,
                   color=P["secondary"], width=width - 48)

    def _debug(self, app, camera, cars):
        from racing.envs.config import RAY_ANGLES
        from racing.envs.sensors import cast_rays

        for car_id, car in cars.items():
            color = AI_COLOR if car_id == "ai" else P["accent"]
            rays = cast_rays(app.track, car.state, RAY_ANGLES, 40.)
            origin = camera.to_screen((car.state.x_m, car.state.y_m))
            for angle, distance in zip(RAY_ANGLES, rays, strict=True):
                heading = car.state.heading_rad + math.radians(angle)
                endpoint = camera.to_screen((car.state.x_m + math.cos(heading) * distance,
                                             car.state.y_m + math.sin(heading) * distance))
                pygame.draw.line(self.surface, color, origin, endpoint, 1)
            control = app.display_controls.get(car_id)
            if control:
                self._text(f"{car_id}: steer {control.steering:+.2f} accel {control.longitudinal:+.2f}",
                           (12, self.surface.get_height() - (112 if car_id == "ai" else 94)), 13, color)
