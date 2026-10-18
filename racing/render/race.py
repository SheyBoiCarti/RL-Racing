"""Desktop race/playback lifecycle; presentation never changes simulation dt."""

from dataclasses import replace
from pathlib import Path

from racing.core.config import load_physics, project_path
from racing.race.config import RaceConfig, RaceRules
from racing.race.controllers import HumanController
from racing.race.replay import ReplayRecorder, load_replay
from racing.race.session import RaceSession
from racing.render.app import FixedScheduler
from racing.render.desktop import DesktopPreferences

_DEFAULT_RULES = object()


class RaceApplication:
    def __init__(
        self,
        game,
        track,
        *,
        controller=None,
        model_path=None,
        physics=None,
        config=None,
        replay_output=None,
        replay_path=None,
        lane_swap=False,
        model_roots=None,
        tolerant=False,
        preferences=None,
        rules=_DEFAULT_RULES,
    ):
        self.preferences = (
            preferences if preferences is not None else DesktopPreferences()
        )
        self._rules_override = rules
        self.exit_to_menu = False
        self.game, self.track = game, track
        self.physics, self.config = physics or load_physics(), config or RaceConfig()
        self.controller, self.model_path = (
            controller,
            Path(model_path) if model_path else None,
        )
        self._loaded_path = self.model_path if controller is not None else None
        self.human = HumanController()
        self.lane_swap, self.tolerant = lane_swap, tolerant
        self.scheduler = FixedScheduler(self.physics.dt_s, game.catch_up_limit)
        self.countdown_scheduler = FixedScheduler(0.05, game.catch_up_limit)
        self.replay_output = (
            Path(replay_output)
            if replay_output
            else project_path("runs/replays/race.jsonl")
        )
        self.replay_path = Path(replay_path) if replay_path else None
        self.recorder = self.playback = None
        self.replay_index = 0
        self.replay_paused = False
        self.running, self.debug, self.error = True, game.debug, ""
        self.state = "MENU"
        self.chooser, self.choosing_track = None, False
        self.split_manifest = "experiments/splits/v1.json"
        self.model_roots = tuple(
            Path(root) for root in (model_roots or (project_path("runs"),))
        )
        self.models = self._find_models()
        if self.controller is None and self.model_path is None and self.models:
            self.model_path = self.models[0]
        if self.replay_path:
            self.open_replay(self.replay_path)
        else:
            self.session = RaceSession(
                track,
                self.physics,
                self.config,
                lane_swap=lane_swap,
                rules=self._new_rules(),
            )

    def _new_rules(self):
        return (
            RaceRules(difficulty=self.preferences.difficulty)
            if self._rules_override is _DEFAULT_RULES
            else self._rules_override
        )

    @property
    def display_recovery(self):
        if self.playback is not None and self.state in ("REPLAY", "RESULTS"):
            return self.playback.frames[self.replay_index].recovery_by_car
        return self.session.recovery_by_car

    def _find_models(self):
        paths = {
            path
            for root in self.model_roots
            if root.is_dir()
            for path in root.rglob("*.zip")
            if path.with_suffix(".metadata.json").is_file()
        }
        if self.model_path:
            paths.add(self.model_path)
        return sorted(
            paths, key=lambda path: (path.name != "best_model.zip", str(path))
        )

    @property
    def model_label(self):
        if self.playback and self.state in ("REPLAY", "RESULTS"):
            return self.playback.header["model_label"] or "Recorded controllers"
        return (
            str(self.model_path)
            if self.model_path
            else "Controller"
            if self.controller
            else "No saved model found"
        )

    @property
    def display_snapshot(self):
        if self.playback is not None and self.state in ("REPLAY", "RESULTS"):
            return self.playback.frames[self.replay_index].snapshot
        return self.session.snapshot()

    @property
    def display_results(self):
        if self.playback is not None and self.state in ("REPLAY", "RESULTS"):
            return self.playback.frames[self.replay_index].results
        return self.session.results

    @property
    def display_positions(self):
        if self.playback is not None and self.state in ("REPLAY", "RESULTS"):
            return self.playback.frames[self.replay_index].position_by_car
        return self.session.position_by_car

    @property
    def display_gap(self):
        if self.playback is not None and self.state in ("REPLAY", "RESULTS"):
            return self.playback.frames[self.replay_index].gap_m
        return self.session.gap_m

    @property
    def display_controls(self):
        if self.playback is not None and self.state in ("REPLAY", "RESULTS"):
            return self.playback.frames[self.replay_index].controls
        return self.session.last_controls

    def _clear_time(self):
        self.scheduler.clear()
        self.countdown_scheduler.clear()

    def _close_race(self, reason):
        if self.session.state in ("RUNNING", "PAUSED"):
            self.session.abort(reason)
        if self.recorder and not self.recorder.closed:
            self.recorder.close()

    def _new_path(self):
        path = self.replay_output
        index = 1
        while path.exists():
            path = self.replay_output.with_name(
                f"{self.replay_output.stem}-{index:03d}{self.replay_output.suffix}"
            )
            index += 1
        return path

    def _start(self):
        try:
            if self.controller is None or self._loaded_path != self.model_path:
                if self.model_path is None:
                    raise ValueError(
                        "Choose a PPO model with its matching metadata sidecar"
                    )
                from racing.agents.policy import PolicyController
                from racing.envs.config import RLConfig

                try:
                    loaded = PolicyController(
                        self.model_path, RLConfig(physics=self.physics)
                    )
                except Exception as error:
                    # Third-party archive decoding also raises AssertionError,
                    # KeyError and BadZipFile for malformed runnable artifacts.
                    raise ValueError(f"{self.model_path}: {error}") from error
                self.controller = loaded
                self._loaded_path = self.model_path
            self.playback = None
            self.replay_index, self.replay_paused = 0, False
            self.session = RaceSession(
                self.track,
                self.physics,
                self.config,
                {"human": self.human, "ai": self.controller},
                lane_swap=self.lane_swap,
                rules=self._new_rules(),
            )
            metadata = getattr(self.controller, "metadata", {})
            self.replay_path = self._new_path()
            self.recorder = ReplayRecorder(
                self.replay_path,
                self.session,
                model_sha256=metadata.get("model_sha256"),
                model_label=self.model_label,
            )
            self.session.start()
            self.state = self.session.state.value
            self.error = ""
        except (ValueError, OSError, ImportError, RuntimeError) as error:
            if self.recorder and not self.recorder.closed:
                self.recorder.close()
            self.state, self.error = "MENU", str(error)
        self._clear_time()

    def open_replay(self, path):
        from racing.tracks import Track

        playback = load_replay(path, tolerant=self.tolerant)
        self.playback = playback
        self.replay_path = Path(path)
        self.session = playback.session
        self.track = Track(playback.header["track"])
        self.physics = self.session.physics
        self.config = self.session.config
        self.lane_swap = self.session.lane_swap
        self.scheduler = FixedScheduler(self.physics.dt_s, self.game.catch_up_limit)
        self.replay_index, self.replay_paused = 0, False
        self.state = "REPLAY"

    def action(self, action):
        if action == "toggle_camera":
            self.preferences.camera_mode = (
                "follow" if self.preferences.camera_mode == "overhead" else "overhead"
            )
        elif (
            action == "difficulty_next"
            and self.state == "MENU"
            and self.playback is None
        ):
            choices = ("easy", "medium", "hard")
            self.preferences.difficulty = choices[
                (choices.index(self.preferences.difficulty) + 1) % 3
            ]
        elif action in ("back", "main_menu"):
            self._close_race("user_exit")
            self._clear_time()
            self.playback = self.recorder = None
            self.exit_to_menu, self.running = True, False
        elif action == "debug":
            self.debug = not self.debug
        elif action == "start" and self.state == "MENU":
            self._start()
        elif action == "restart":
            self._close_race("restart")
            self._start()
        elif action in ("quit", "menu"):
            self._close_race("user_exit")
            self._clear_time()
            self.state = "MENU"
            if action == "quit":
                self.running = False
            else:
                self.playback = None
                self.recorder = None
                self.session = RaceSession(
                    self.track,
                    self.physics,
                    self.config,
                    lane_swap=self.lane_swap,
                    rules=self._new_rules(),
                )
        elif action == "pause":
            if self.state == "RUNNING":
                self.session.pause()
                self.state = "PAUSED"
            elif self.state == "REPLAY":
                self.replay_paused = True
            self._clear_time()
        elif action == "resume":
            if self.state == "PAUSED":
                self.session.resume()
                self.state = "RUNNING"
            elif self.state == "REPLAY":
                self.replay_paused = False
            self._clear_time()
        elif action == "replay" and self.state == "RESULTS" and self.replay_path:
            try:
                self.open_replay(self.replay_path)
                self.error = ""
            except (ValueError, OSError) as error:
                self.error = str(error)
            self._clear_time()
        elif (
            action in ("model_prev", "model_next")
            and self.state == "MENU"
            and self.models
        ):
            index = (
                self.models.index(self.model_path)
                if self.model_path in self.models
                else 0
            )
            self.model_path = self.models[
                (index + (-1 if action == "model_prev" else 1)) % len(self.models)
            ]
            self.error = ""
        elif action == "swap_lanes" and self.state == "MENU":
            self.lane_swap = not self.lane_swap
            self.session = RaceSession(
                self.track,
                self.physics,
                self.config,
                lane_swap=self.lane_swap,
                rules=self._new_rules(),
            )
        elif action == "choose_track" and self.state == "MENU":
            from racing.render.chooser import TrackChooser

            self.chooser = self.chooser or TrackChooser(self.split_manifest)
            self.choosing_track = True
        elif self.choosing_track:
            if action in ("chooser_prev", "chooser_next"):
                self.chooser.move(-1 if action == "chooser_prev" else 1)
            elif action.startswith("track:"):
                track_id = action.split(":", 1)[1]
                if track_id in self.chooser.ids:
                    self.chooser.index = self.chooser.ids.index(track_id)
            elif action == "select_track":
                try:
                    track = self.chooser.tracks[self.chooser.selected_id]
                    session = RaceSession(
                        track,
                        self.physics,
                        self.config,
                        lane_swap=self.lane_swap,
                        rules=self._new_rules(),
                    )
                    self.track, self.session = track, session
                    self.playback = None
                    self.choosing_track = False
                    self.error = ""
                except (ValueError, OSError) as error:
                    self.chooser.error = str(error)
            elif action == "chooser_back":
                self.choosing_track = False

    def frame(self, elapsed, control):
        if self.choosing_track:
            return 0
        if self.state == "COUNTDOWN":

            def countdown():
                self.session.countdown_tick()
                self.state = self.session.state.value
                if self.state == "RUNNING":
                    self._clear_time()
                    return False
                return True

            return self.countdown_scheduler.consume(elapsed, countdown)
        if self.state == "REPLAY" and not self.replay_paused:

            def replay_tick():
                if self.replay_index < len(self.playback.frames) - 1:
                    self.replay_index += 1
                if self.replay_index == len(self.playback.frames) - 1:
                    self.state = "RESULTS"
                    return False
                return True

            return self.scheduler.consume(elapsed, replay_tick)
        if self.state != "RUNNING":
            return 0

        def tick():
            # A timestamped supplier lets deterministic input recordings target
            # physics ticks independently of the number of render frames.
            self.human.set_control(
                control(self.session.snapshot().tick) if callable(control) else control
            )
            self.session.step()
            self.state = self.session.state.value
            self.error = self.session.error_message
            return self.state == "RUNNING"

        return self.scheduler.consume(elapsed, tick)

    def run(self):
        import pygame

        from racing.render.input import held_control, key_bindings
        from racing.render.race_view import RaceView

        pygame.init()
        try:
            surface = pygame.display.set_mode(
                (self.game.width_px, self.game.height_px), pygame.RESIZABLE
            )
            pygame.display.set_caption("RL Racing — Human vs AI")
            view = RaceView(surface, self.game, self.physics)
            clock = pygame.time.Clock()
            bindings = key_bindings(self.game)
            while self.running:
                elapsed = clock.tick(self.game.render_fps) / 1000
                transitioned = False
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.action("quit")
                    elif event.type == pygame.VIDEORESIZE:
                        surface = pygame.display.set_mode(
                            (max(480, event.w), max(360, event.h)), pygame.RESIZABLE
                        )
                        view.surface = surface
                    elif event.type == pygame.KEYDOWN and not getattr(
                        event, "repeat", False
                    ):
                        if self.choosing_track:
                            action = {
                                pygame.K_LEFT: "chooser_prev",
                                pygame.K_UP: "chooser_prev",
                                pygame.K_RIGHT: "chooser_next",
                                pygame.K_DOWN: "chooser_next",
                                pygame.K_RETURN: "select_track",
                                pygame.K_ESCAPE: "chooser_back",
                            }.get(event.key)
                        else:
                            action = {
                                pygame.K_RETURN: "start"
                                if self.state == "MENU"
                                else "resume",
                                pygame.K_ESCAPE: "back"
                                if self.state == "MENU"
                                else "menu"
                                if self.state in ("PAUSED", "RESULTS")
                                else "resume"
                                if self.state == "REPLAY" and self.replay_paused
                                else "pause",
                                pygame.K_c: "choose_track",
                                pygame.K_p: "replay",
                                pygame.K_LEFTBRACKET: "model_prev",
                                pygame.K_RIGHTBRACKET: "model_next",
                                pygame.K_l: "swap_lanes",
                                pygame.K_q: "quit",
                                pygame.K_v: "toggle_camera",
                                pygame.K_d: "difficulty_next",
                            }.get(event.key)
                            pause_action = (
                                "back"
                                if self.state == "MENU"
                                else "menu"
                                if self.state in ("PAUSED", "RESULTS")
                                else "resume"
                                if self.state == "REPLAY" and self.replay_paused
                                else "pause"
                            )
                            for name, target in (
                                ("reset", "restart"),
                                ("debug", "debug"),
                                ("pause", pause_action),
                            ):
                                if event.key in bindings[name]:
                                    action = target
                        if action:
                            self.action(action)
                            transitioned |= action not in (
                                "debug",
                                "toggle_camera",
                                "difficulty_next",
                            )
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        # Rebuild after state/resize events in this batch.
                        view.draw(self)
                        buttons = (
                            self.chooser.draw(surface)
                            if self.choosing_track
                            else view.buttons
                        )
                        for action, rect in buttons.items():
                            if rect.collidepoint(event.pos):
                                self.action(action)
                                transitioned |= action not in (
                                    "debug",
                                    "toggle_camera",
                                    "difficulty_next",
                                )
                                break
                if not self.running:
                    break
                pressed = pygame.key.get_pressed()
                held = {
                    key for keys in bindings.values() for key in keys if pressed[key]
                }
                if not transitioned:
                    self.frame(elapsed, held_control(held, tuple(bindings.items())))
                view.draw(self)
                if self.choosing_track:
                    self.chooser.draw(surface)
                pygame.display.flip()
        finally:
            self._close_race("user_exit")
            pygame.quit()


def application_from_args(args, *, preferences):
    from racing.core.config import load_game
    from racing.render.chooser import TrackChooser

    if args.mode == "replay":
        app = RaceApplication(
            load_game(args.config),
            None,
            replay_path=project_path(args.replay),
            tolerant=args.tolerant_replay,
            preferences=preferences,
        )
    else:
        chooser = TrackChooser(args.split_manifest)
        track = chooser.tracks.get(args.track, chooser.tracks["demo"])
        app = RaceApplication(
            load_game(args.config),
            track,
            model_path=project_path(args.model) if args.model else None,
            config=replace(RaceConfig(), target_laps=args.laps),
            replay_output=project_path(args.replay_output)
            if args.replay_output
            else None,
            lane_swap=args.swap_lanes,
            preferences=preferences,
            model_roots=[project_path(root) for root in args.model_root]
            if args.model_root
            else None,
        )
        app.chooser, app.split_manifest = chooser, args.split_manifest
        if args.track not in chooser.tracks:
            app.error = f"{args.track}: unavailable in demo/train/validation catalog"
            app.choosing_track = True
            chooser.error = app.error
    return app


def run_from_args(args):
    from racing.render.desktop import run_desktop

    return run_desktop(args)
