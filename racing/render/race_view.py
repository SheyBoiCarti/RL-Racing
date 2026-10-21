"""Two-car race presentation, compact layouts and an offscreen opponent cue."""

import math

import pygame

from racing.render.interface import format_time
from racing.render.race_camera import overhead_camera, race_world_rect
from racing.render.scene import TrackScene
from racing.render.style import PALETTE as P
from racing.render.style import Camera, font

AI_COLOR = (102, 195, 232)


def opponent_indicator(camera, human, ai, *, world_rect=None):
    """Screen-edge position and angle within the reserved world area."""
    point = camera.to_screen((ai.x_m, ai.y_m))
    width, height = camera.viewport
    bounds = (world_rect.inflate(-32, -32) if world_rect is not None
              else pygame.Rect(24, 130, width - 48, height - 190))
    if bounds.collidepoint(point):
        return None
    origin = camera.to_screen((human.x_m, human.y_m))
    dx, dy = point[0] - origin[0], point[1] - origin[1]
    if abs(dx) + abs(dy) < 1e-9:
        return None
    limits = []
    if dx:
        limits.append(((bounds.right if dx > 0 else bounds.left) - origin[0]) / dx)
    if dy:
        limits.append(((bounds.bottom if dy > 0 else bounds.top) - origin[1]) / dy)
    positive = [value for value in limits if value >= 0]
    fraction = min(positive) if positive else 0
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
            rect = pygame.Rect(
                12 + (index % columns) * (button_width + gap),
                height - 12 - rows * 40 + (index // columns) * 40,
                button_width,
                34,
            )
            pygame.draw.rect(self.surface, P["panel"], rect, border_radius=5)
            pygame.draw.rect(self.surface, P["accent"], rect, 1, border_radius=5)
            rendered = font(13 if width < 800 else 17).render(label, True, P["ivory"])
            self.surface.blit(rendered, rendered.get_rect(center=rect.center))
            self.buttons[action] = rect

    def _actions(self, app):
        camera = (
            "toggle_camera",
            f"Map {'On' if app.preferences.camera_mode == 'overhead' else 'Off'} [V]",
        )
        if app.state == "MENU":
            return [
                ("start", "Start"),
                ("choose_track", "Track [C]"),
                ("swap_lanes", "Swap lanes [L]"),
                camera,
                ("difficulty_next", f"{app.preferences.difficulty.title()} [D]"),
                ("back", "Back [Esc]"),
                ("quit", "Quit [Q]"),
            ]
        if app.state == "COUNTDOWN":
            return [("menu", "Cancel"), camera]
        if app.state == "PAUSED":
            return [
                ("resume", "Resume [Enter]"),
                ("restart", "Restart [R]"),
                ("main_menu", "Main menu"),
                camera,
            ]
        if app.state == "RESULTS":
            return [
                ("restart", "Race again [R]"),
                ("replay", "Replay [P]"),
                ("menu", "Setup [Esc]"),
                ("main_menu", "Main menu"),
                camera,
                ("quit", "Quit"),
            ]
        if app.state == "REPLAY":
            return [
                (
                    "resume" if app.replay_paused else "pause",
                    "Resume" if app.replay_paused else "Pause [Esc]",
                ),
                camera,
                ("main_menu", "Main menu"),
                ("quit", "Quit"),
            ]
        human = next(car for car in app.display_snapshot.cars if car.car_id == "human")
        if app.state == "RUNNING" and human.status.value != "active":
            return [("restart", "Restart [R]"), ("main_menu", "Main menu"), camera]
        return [camera]

    def _draw_minimap(self, app, cars, compact):
        """Draw a north-up circuit map without changing the driving camera."""
        size = (128, 94) if compact else (190, 138)
        self.minimap_rect = pygame.Rect(
            self.world_rect.right - size[0] - 8,
            self.world_rect.top + 8,
            *size,
        )
        self.minimap_camera = overhead_camera(
            app.track,
            self.surface.get_size(),
            self.minimap_rect.inflate(-12, -12),
            margin_m=app.physics.car_radius_m,
        )
        panel = pygame.Surface(self.minimap_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(panel, (*P["panel"], 225), panel.get_rect(), border_radius=6)
        self.surface.blit(panel, self.minimap_rect)
        pygame.draw.rect(self.surface, P["secondary"], self.minimap_rect, 1, border_radius=6)

        def screen(point):
            return self.minimap_camera.to_screen(point)

        old_clip = self.surface.get_clip()
        self.surface.set_clip(self.minimap_rect)
        pygame.draw.polygon(self.surface, P["secondary"], [screen(point) for point in app.track.outer])
        pygame.draw.polygon(self.surface, P["panel"], [screen(point) for point in app.track.inner])
        for car_id, car in cars.items():
            color = P["accent"] if car_id == "human" else AI_COLOR
            point = screen((car.state.x_m, car.state.y_m))
            pygame.draw.circle(self.surface, color, point, 5 if compact else 6)
            heading = car.state.heading_rad
            pygame.draw.line(
                self.surface,
                P["ivory"],
                point,
                (point[0] + math.cos(heading) * 8, point[1] + math.sin(heading) * 8),
                2,
            )
        self.surface.set_clip(old_clip)

    def draw(self, app):
        snapshot = app.display_snapshot
        cars = {car.car_id: car for car in snapshot.cars}
        human, ai = cars["human"], cars["ai"]
        width, height = self.surface.get_size()
        compact = width < 800 or height < 560
        actions = self._actions(app)
        self.world_rect = race_world_rect(
            (width, height),
            state=app.state,
            button_rows=math.ceil(len(actions) / 4),
            debug=app.debug,
        )
        self.physics = self.scene.physics = app.physics
        camera = Camera(
            (human.state.x_m, human.state.y_m),
            self.game.pixels_per_metre,
            (width, height),
        )
        self.camera = camera
        # World drawing is clipped to the reserved area; HUD/buttons own the rest.
        self.surface.fill(P["panel"])
        old_clip = self.surface.get_clip()
        self.surface.set_clip(self.world_rect)
        self.scene.draw(
            self.surface, app.track, snapshot, camera, app.debug and not compact
        )
        self.car_labels = {}
        self.recovery_messages = {}
        label_rects = []
        for car_id, car in cars.items():
            point = camera.to_screen((car.state.x_m, car.state.y_m))
            color = P["accent"] if car_id == "human" else AI_COLOR
            label = "YOU" if car_id == "human" else "AI"
            recovery = app.display_recovery.get(car_id)
            if recovery is not None:
                if recovery.remaining_ticks:
                    message = f"Back on track in {recovery.remaining_ticks * app.physics.dt_s:.1f} s"
                    label += " recovering"
                    self.recovery_messages[car_id] = message
                elif (
                    recovery.last_release_tick is not None
                    and 0 <= snapshot.tick - recovery.last_release_tick < 10
                ):
                    self.recovery_messages[car_id] = "Go!"
            rect = (
                font(14)
                .render(label, True, color)
                .get_rect(topleft=(point[0] - 12, point[1] - 22))
            )
            rect.clamp_ip(self.world_rect)
            if any(rect.colliderect(other) for other in label_rects):
                rect.y += 20
                rect.clamp_ip(self.world_rect)
                if any(rect.colliderect(other) for other in label_rects):
                    rect.y = max(
                        self.world_rect.top, label_rects[0].top - rect.height - 2
                    )
            label_rects.append(rect)
            self.car_labels[car_id] = rect.topleft
            self._text(label, rect.topleft, 14, color)
        indicator = opponent_indicator(camera, human.state, ai.state, world_rect=self.world_rect)
        if indicator:
            point, angle = indicator
            vertices = [
                (
                    point[0] + math.cos(angle + offset) * radius,
                    point[1] + math.sin(angle + offset) * radius,
                )
                for offset, radius in ((0, 13), (2.5, 9), (-2.5, 9))
            ]
            pygame.draw.polygon(self.surface, AI_COLOR, vertices)
        if app.preferences.camera_mode == "overhead":
            self._draw_minimap(app, cars, compact)
        else:
            self.minimap_rect = None
            self.minimap_camera = None
        for index, (car_id, message) in enumerate(self.recovery_messages.items()):
            self._text(
                f"{'YOU' if car_id == 'human' else 'AI'}: {message}",
                (self.world_rect.left + 6, self.world_rect.top + 4 + index * 19),
                14,
            )
        if app.state == "RUNNING" and human.status.value != "active":
            reason = app.display_results["human"].reason.replace("_", " ")
            if human.status.value == "timed_out":
                reason += " timeout"
            self._text(f"YOU: {reason}. Restart or return to main menu.",
                       (self.world_rect.left + 6, self.world_rect.top + 4),
                       13 if compact else 18, width=self.world_rect.width - 12)
        self.surface.set_clip(old_clip)
        line = 19 if compact else 27
        size = 13 if compact else 18
        rules = app.session.rules
        identity = (
            f"{rules.difficulty.title()} · {rules.version}" if rules else "Legacy rules"
        )
        if app.state == "MENU":
            identity = (
                f"{app.preferences.difficulty.title()} · {app.preferences.camera_mode.title()} · "
                f"{app.config.target_laps} laps · "
                + (rules.version if rules else "Legacy rules")
            )
            lines = [
                f"RACE AI · {app.track.name} · {'DEBUG' if app.debug else 'GHOST RACE'}",
                f"Model: {app.model_label}",
                identity,
                "Crashes recover automatically; race time keeps running."
                if rules
                else "Legacy off-track rules",
            ]
        else:
            lap = min(app.config.target_laps, human.completed_laps + 1)
            lap_time = (
                (snapshot.tick - human.lap_start_tick) * app.physics.dt_s
                if human.lap_start_tick is not None
                else 0
            )
            lines = [
                f"YOU {human.state.speed_mps * 3.6:.0f} km/h · Lap {lap}/{app.config.target_laps} · {format_time(lap_time)}",
                f"Race {format_time(snapshot.time_s)} · {app.track.name}",
                "   ".join(
                    f"{'YOU' if key == 'human' else 'AI'} P{app.display_positions[key]} "
                    f"{'recovering' if app.display_recovery.get(key) and app.display_recovery[key].remaining_ticks else cars[key].status.value}"
                    for key in ("human", "ai")
                ),
                identity,
            ]
        for index, text in enumerate(lines):
            self._text(text, (12, 4 + index * line), size, width=width - 24)
        if app.debug and not compact:
            self._debug(app, camera, cars)
        if app.state == "COUNTDOWN":
            rendered = font(64, True).render(
                str(math.ceil(app.session.countdown_ticks_remaining / 20)),
                True,
                P["ivory"],
            )
            self.surface.blit(
                rendered, rendered.get_rect(center=self.world_rect.center)
            )
        elif app.state == "PAUSED":
            self._text(
                "PAUSED",
                (self.world_rect.centerx - 48, self.world_rect.centery - 15),
                26,
            )
        elif app.state == "RESULTS":
            self._results(app, compact)
        elif app.state == "REPLAY":
            self._text(
                "REPLAY · PAUSED" if app.replay_paused else "REPLAY · VERIFIED",
                (12, height - 70),
                14,
                AI_COLOR,
            )
        self._buttons(actions)
        if app.error:
            self._text(
                app.error,
                (12, 80 if compact else 103),
                12,
                color=(255, 154, 137),
                width=width - 24,
            )
        # Hover over the model line to read the complete checkpoint identity.
        if (
            app.state == "MENU"
            and pygame.display.get_init()
            and pygame.Rect(0, 20, width, 20).collidepoint(pygame.mouse.get_pos())
        ):
            chunks = [
                app.model_label[i : i + max(20, width // 8)]
                for i in range(0, len(app.model_label), max(20, width // 8))
            ]
            panel = pygame.Rect(12, 96, width - 24, 18 * len(chunks) + 8)
            pygame.draw.rect(self.surface, P["panel"], panel)
            for index, text in enumerate(chunks):
                self._text(text, (16, 100 + index * 18), 12)

    def _results(self, app, compact):
        panel = self.world_rect
        pygame.draw.rect(self.surface, P["panel"], panel, border_radius=6)
        title = {
            "human": "YOU WIN",
            "ai": "AI WINS",
            "draw": "DRAW",
            "no_winner": "NO WINNER",
        }[app.session.winner]
        self._text(title, (panel.left + 8, panel.top + 4), 18 if compact else 28)
        y = panel.top + (29 if compact else 42)
        for car_id in ("human", "ai"):
            result = app.display_results[car_id]
            self._text(
                f"{car_id.upper()} · {result.reason} · {result.completed_laps} laps · "
                f"{result.recovery_count} recoveries / {result.recovery_held_ticks * app.physics.dt_s:.2f}s held",
                (panel.left + 8, y),
                12 if compact else 17,
                width=panel.width - 16,
            )
            best = f"{result.best_lap_s:.2f}s" if result.best_lap_s is not None else "—"
            finish = (
                f"{result.finish_time_s:.2f}s"
                if result.finish_time_s is not None
                else "—"
            )
            self._text(
                f"Finish {finish} · Best {best} · Laps "
                + (", ".join(f"{lap:.2f}s" for lap in result.lap_times_s) or "—"),
                (panel.left + 8, y + (17 if compact else 24)),
                12 if compact else 16,
                width=panel.width - 16,
            )
            y += 40 if compact else 60

    def _debug(self, app, camera, cars):
        from racing.envs.config import RAY_ANGLES
        from racing.envs.sensors import cast_rays

        for car_id, car in cars.items():
            color = AI_COLOR if car_id == "ai" else P["accent"]
            rays = cast_rays(app.track, car.state, RAY_ANGLES, 40.0)
            origin = camera.to_screen((car.state.x_m, car.state.y_m))
            for angle, distance in zip(RAY_ANGLES, rays, strict=True):
                heading = car.state.heading_rad + math.radians(angle)
                endpoint = camera.to_screen(
                    (
                        car.state.x_m + math.cos(heading) * distance,
                        car.state.y_m + math.sin(heading) * distance,
                    )
                )
                pygame.draw.line(self.surface, color, origin, endpoint, 1)
            control = app.display_controls.get(car_id)
            if control:
                self._text(
                    f"{car_id}: steer {control.steering:+.2f} accel {control.longitudinal:+.2f}",
                    (12, self.surface.get_height() - (112 if car_id == "ai" else 94)),
                    13,
                    color,
                )
