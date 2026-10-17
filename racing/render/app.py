"""Explicit desktop lifecycle and bounded fixed-dt scheduler."""

import math
from enum import Enum

from racing.core.config import load_game, load_physics
from racing.core.models import CarSpawn, CarStatus
from racing.core.simulation import Simulation
from racing.render.input import held_control, key_bindings


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
        self.watch_mode = False
        self.selected_policy = None
        self.selected_mode = None
        self.chooser = None
        self.choosing_track = False
        self.split_manifest = "experiments/splits/v1.json"

    def select_track(self, track):
        self.simulation.reset(track, 0, (CarSpawn("human", "solo"),))
        self.track = track
        self.best_lap_s = None
        self.scheduler.clear()
        self.state = AppState.MENU

    def action(self, action):
        if action == "choose_track" and self.state == AppState.MENU:
            from racing.render.chooser import TrackChooser

            if self.chooser is None:
                self.chooser = TrackChooser(self.split_manifest)
            if self.track.track_id in self.chooser.ids:
                self.chooser.index = self.chooser.ids.index(self.track.track_id)
            self.choosing_track = True
            self.scheduler.clear()
        elif action == "chooser_back":
            self.choosing_track = False
        elif action in ("chooser_prev", "chooser_next") and self.choosing_track:
            self.chooser.move(-1 if action == "chooser_prev" else 1)
        elif action.startswith("track:") and self.choosing_track:
            key = action.split(":", 1)[1]
            if key in self.chooser.ids:
                self.chooser.index = self.chooser.ids.index(key)
        elif action == "select_track" and self.choosing_track:
            try:
                self.select_track(self.chooser.tracks[self.chooser.selected_id])
                self.choosing_track = False
            except (ValueError, OSError) as error:
                self.chooser.error = f"{self.chooser.selected_id}.json: {error}"
        elif action == "race" and self.state == AppState.MENU:
            self.selected_mode = "race"
            self.running = False
        elif action == "watch" and self.state == AppState.MENU:
            from tkinter import Tk, filedialog

            from racing.agents.policy import PolicyController

            dialog = Tk()
            dialog.withdraw()
            try:
                path = filedialog.askopenfilename(title="Choose trained PPO model", filetypes=[("PPO model", "*.zip")])
                if path:
                    self.selected_policy = PolicyController(path)
                    self.running = False
            except (ValueError, OSError, ImportError) as error:
                self.action("choose_track")
                self.chooser.error = f"{path}: {error}"
            finally:
                dialog.destroy()
        elif action == "quit":
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
            pygame.display.set_caption("RL Racing — Watch AI" if self.watch_mode else "RL Racing — Drive")
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
                        if self.choosing_track:
                            actions = {pygame.K_LEFT: "chooser_prev", pygame.K_UP: "chooser_prev",
                                       pygame.K_RIGHT: "chooser_next", pygame.K_DOWN: "chooser_next",
                                       pygame.K_RETURN: "select_track", pygame.K_ESCAPE: "chooser_back"}
                            self.action(actions.get(event.key, "noop"))
                            transitioned = True
                            continue
                        if event.key == pygame.K_c and self.state == AppState.MENU:
                            self.action("choose_track")
                            transitioned = True
                            continue
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
                        if self.choosing_track:
                            for action, rect in self.chooser.draw(surface).items():
                                if rect.collidepoint(event.pos):
                                    self.action(action)
                                    transitioned = True
                                    break
                            continue
                        # Events in one batch may resize or change state before
                        # this click; the last drawn action map can be stale.
                        rects = layout(surface.get_size(), self.state.value, self.watch_mode)
                        for action in ("reset", "watch", "race", "resume", "menu", "quit"):
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
                renderer.interface.watch_mode = self.watch_mode
                renderer.policy_debug = (
                    f"{self.model_label} · steer {self.last_action[0]:+.2f} · accel {self.last_action[1]:+.2f}"
                    if self.watch_mode else None
                )
                renderer.draw(
                    self.track,
                    self.simulation.snapshot(),
                    self.state.value,
                    self.debug,
                    self.best_lap_s,
                    self.scheduler.dropped_time_s,
                )
                if self.choosing_track:
                    renderer.buttons = self.chooser.draw(surface)
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
        description="Drive, watch a saved policy, race the AI, or play a verified action replay."
    )
    parser.add_argument(
        "--mode",
        default="drive",
        choices=("drive", "watch", "race", "replay"),
        help="desktop driving, policy viewing, human versus AI racing, or model-free replay",
    )
    parser.add_argument("--model", help="Watch/race PPO zip with matching metadata sidecar")
    parser.add_argument("--laps", type=int, default=None, help="Race target laps (default 3)")
    parser.add_argument("--replay-output", help="New race JSONL recording path")
    parser.add_argument("--replay", help="Self-contained replay JSONL to play")
    parser.add_argument("--swap-lanes", action="store_true", help="Swap human and AI grid lanes")
    parser.add_argument("--tolerant-replay", action="store_true", help="Compare replay floats at absolute tolerance 1e-6")
    parser.add_argument("--model-root", action="append", help="Race menu model search root (repeatable; default runs)")
    parser.add_argument(
        "--track", default="demo", help="track ID under assets/tracks (default: demo)"
    )
    parser.add_argument("--split-manifest", default="experiments/splits/v1.json",
                        help="frozen public train/validation catalog")
    parser.add_argument(
        "--config",
        default="configs/game.yaml",
        help="game configuration relative to project root",
    )
    args = parser.parse_args(argv)
    if args.mode == "replay" and not args.replay:
        parser.error("--mode replay requires --replay")
    if args.replay and args.mode != "replay":
        parser.error("--replay requires --mode replay")
    if args.tolerant_replay and args.mode != "replay":
        parser.error("--tolerant-replay requires --mode replay")
    if (args.laps is not None or args.replay_output or args.swap_lanes or args.model_root) and args.mode != "race":
        parser.error("--laps/--replay-output/--swap-lanes/--model-root require --mode race")
    if args.laps is not None and args.laps < 1:
        parser.error("--laps must be positive")
    if args.model and args.mode not in ("watch", "race"):
        parser.error("--model requires --mode watch or race")
    if not args.track or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for c in args.track
    ):
        parser.error("--track must be a track ID, without path separators")
    try:
        if args.mode in ("race", "replay"):
            from racing.render.race import run_from_args

            args.laps = args.laps or 3
            run_from_args(args)
            return
        from racing.render.chooser import TrackChooser

        if args.mode == "watch" and not args.model:
            parser.error("--mode watch requires --model")
        if args.model and args.mode != "watch":
            parser.error("--model requires --mode watch")
        chooser = TrackChooser(args.split_manifest)
        track = chooser.tracks.get(args.track, chooser.tracks["demo"])
        if args.track not in chooser.tracks:
            reason = f"; {chooser.error}" if chooser.error else ""
            chooser.error = f"{args.track}.json: track unavailable in demo/train/validation catalog{reason}"
        chooser.index = chooser.ids.index(track.track_id)
        if args.mode == "watch":
            from racing.agents.policy import PolicyController
            from racing.render.watch import WatchApplication

            try:
                app = WatchApplication(load_game(args.config), PolicyController(args.model), track=track)
            except (ValueError, OSError, ImportError) as error:
                app = Application(load_game(args.config), load_physics(), track)
                chooser.error = f"{args.model}: {error}"
        else:
            app = Application(load_game(args.config), load_physics(), track)
        app.chooser = chooser
        app.split_manifest = args.split_manifest
        app.choosing_track = True
        app.run()
        if app.selected_mode == "race":
            from racing.render.race import RaceApplication

            RaceApplication(app.game, app.track).run()
        elif app.selected_policy is not None:
            from racing.render.watch import WatchApplication

            WatchApplication(app.game, app.selected_policy, track=app.track).run()
    except (ValueError, OSError) as error:
        parser.exit(1, f"Cannot start game: {error}\n")
