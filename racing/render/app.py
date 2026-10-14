"""Explicit desktop lifecycle and bounded fixed-dt scheduler."""

import math
from enum import Enum

from racing.core.config import load_game, load_physics
from racing.core.models import CarSpawn, CarStatus
from racing.core.simulation import Simulation
from racing.render.input import held_control, key_bindings
from racing.tracks import Track


class AppState(str, Enum):
    MENU = "MENU"
    DRIVING = "DRIVING"
    PAUSED = "PAUSED"
    ATTEMPT_FAILED = "ATTEMPT_FAILED"
    ATTEMPT_FINISHED = "ATTEMPT_FINISHED"


class FixedScheduler:
    def __init__(self, dt_s, limit=5):
        self.dt_s, self.limit = dt_s, limit
        self.accumulator = 0.0
        self.dropped_time_s = 0.0

    def clear(self):
        self.accumulator = 0.0

    def consume(self, elapsed, step):
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("elapsed: finite nonnegative time required")
        self.accumulator += elapsed
        count = 0
        while self.accumulator + 1e-12 >= self.dt_s and count < self.limit:
            self.accumulator = max(0.0, self.accumulator - self.dt_s)
            count += 1
            if step() is False:
                self.clear()
                return count
        if self.accumulator + 1e-12 >= self.dt_s:
            excess = math.floor((self.accumulator + 1e-12) / self.dt_s)
            self.dropped_time_s += excess * self.dt_s
            self.accumulator = max(0.0, self.accumulator - excess * self.dt_s)
        return count


class Application:
    def __init__(self, game, physics, track):
        self.game, self.physics, self.track = game, physics, track
        self.simulation = Simulation(physics)
        self.simulation.reset(track, 0, (CarSpawn("human", "solo"),))
        self.scheduler = FixedScheduler(physics.dt_s, game.catch_up_limit)
        self.state = AppState.MENU
        self.debug = game.debug
        self.best_lap_s = None
        self.running = True

    def action(self, action):
        if action == "quit":
            self.running = False
        elif action == "debug":
            self.debug = not self.debug
        elif action == "reset":
            self.simulation.reset(self.track, 0, (CarSpawn("human", "solo"),))
            self.scheduler.clear()
            self.state = AppState.DRIVING
        elif action == "pause" and self.state == AppState.DRIVING:
            self.scheduler.clear()
            self.state = AppState.PAUSED
        elif action == "resume" and self.state == AppState.PAUSED:
            self.scheduler.clear()
            self.state = AppState.DRIVING
        elif action == "menu" or (action == "pause" and self.state != AppState.DRIVING):
            self.scheduler.clear()
            self.state = AppState.MENU

    def frame(self, elapsed, control):
        if self.state != AppState.DRIVING:
            return 0

        def tick():
            result = self.simulation.step({"human": control})
            car = result.snapshot.cars[0]
            if car.status == CarStatus.DISQUALIFIED:
                self.state = AppState.ATTEMPT_FAILED
                return False
            if car.completed_laps:
                lap = car.completed_lap_times_s[-1]
                self.best_lap_s = (
                    lap if self.best_lap_s is None else min(lap, self.best_lap_s)
                )
                self.simulation.retire_car("human", CarStatus.FINISHED)
                self.state = AppState.ATTEMPT_FINISHED
                return False
            return True

        return self.scheduler.consume(elapsed, tick)

    def run(self):
        import pygame

        from racing.render.renderer import Renderer
        from racing.render.style import layout

        pygame.init()
        try:
            bindings = key_bindings(self.game)
            surface = pygame.display.set_mode(
                (self.game.width_px, self.game.height_px), pygame.RESIZABLE
            )
            pygame.display.set_caption("RL Racing — Drive")
            renderer = Renderer(surface, self.game, self.physics)
            clock = pygame.time.Clock()
            while self.running:
                elapsed = clock.tick(self.game.render_fps) / 1000.0
                transitioned = False
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.action("quit")
                    elif event.type == pygame.VIDEORESIZE:
                        surface = pygame.display.set_mode(
                            (max(480, event.w), max(360, event.h)), pygame.RESIZABLE
                        )
                        renderer.surface = surface
                    elif event.type == pygame.KEYDOWN and not getattr(
                        event, "repeat", False
                    ):
                        for action in ("pause", "reset", "debug"):
                            if event.key in bindings[action]:
                                self.action(action)
                                transitioned |= action != "debug"
                        if event.key == pygame.K_RETURN:
                            self.action(
                                "resume"
                                if self.state == AppState.PAUSED
                                else "reset"
                                if self.state == AppState.MENU
                                else "noop"
                            )
                            transitioned = True
                        if event.key == pygame.K_q and self.state == AppState.MENU:
                            self.action("quit")
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        # Events in one batch may resize or change state before
                        # this click; the last drawn action map can be stale.
                        rects = layout(surface.get_size(), self.state.value)
                        for action in ("reset", "resume", "menu", "quit"):
                            rect = rects.get(action)
                            if rect is None:
                                continue
                            if rect.collidepoint(event.pos):
                                self.action(action)
                                transitioned = True
                                break
                if not self.running:
                    break
                pressed = pygame.key.get_pressed()
                held = {
                    key for keys in bindings.values() for key in keys if pressed[key]
                }
                if not transitioned:
                    self.frame(elapsed, held_control(held, tuple(bindings.items())))
                renderer.pointer = pygame.mouse.get_pos()
                renderer.draw(
                    self.track,
                    self.simulation.snapshot(),
                    self.state.value,
                    self.debug,
                    self.best_lap_s,
                    self.scheduler.dropped_time_s,
                )
                hovered = any(rect.collidepoint(renderer.pointer)
                              for rect in renderer.buttons.values())
                try:
                    pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND if hovered
                                            else pygame.SYSTEM_CURSOR_ARROW)
                except pygame.error:
                    pass  # SDL dummy/headless backends do not supply cursors.
                pygame.display.flip()
        finally:
            pygame.quit()


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Drive the deterministic Phase 1 circuit. Enter starts; Esc pauses."
    )
    parser.add_argument(
        "--mode",
        default="drive",
        choices=("drive",),
        help="only manual drive is supported",
    )
    parser.add_argument(
        "--track", default="demo", help="track ID under assets/tracks (default: demo)"
    )
    parser.add_argument(
        "--config",
        default="configs/game.yaml",
        help="game configuration relative to project root",
    )
    args = parser.parse_args(argv)
    if not args.track or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for c in args.track
    ):
        parser.error("--track must be a track ID, without path separators")
    try:
        Application(
            load_game(args.config),
            load_physics(),
            Track.from_json(f"assets/tracks/{args.track}.json"),
        ).run()
    except (ValueError, OSError) as error:
        parser.exit(1, f"Cannot start game: {error}\n")
