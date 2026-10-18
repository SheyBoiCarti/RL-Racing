"""Desktop watch lifecycle over the exact shared RL environment."""

from dataclasses import replace

import numpy as np

from racing.envs.config import load_rl
from racing.envs.racing_env import RacingEnv
from racing.render.app import Application, AppState, FixedScheduler


class WatchApplication(Application):
    def __init__(self, game, controller, config=None, track=None):
        self.game, self.controller = game, controller
        self.config = config or load_rl()
        if track is not None:
            self.config = replace(self.config, default_track_id=track.track_id,
                                  allowed_split=track.split, track_sampling="fixed")
        catalog = {track.track_id: (track, track.split)} if track is not None else None
        self.env = RacingEnv(config=self.config, track_catalog=catalog)
        self.observation, _ = self.env.reset(seed=9000)
        self.physics, self.track = self.config.physics, self.env.track
        self.simulation = self.env.simulation
        self.scheduler = FixedScheduler(self.physics.dt_s, game.catch_up_limit)
        self.state, self.debug, self.best_lap_s = AppState.MENU, game.debug, None
        self.running, self.watch_mode = True, True
        self.exit_to_menu = False
        self.last_action = np.zeros(2, dtype=np.float32)
        self.model_label = f"PPO · step {controller.metadata['checkpoint_step']}"
        self.selected_policy = None
        self.chooser = None
        self.choosing_track = False
        self.split_manifest = "experiments/splits/v1.json"

    def select_track(self, track):
        config = replace(self.config, default_track_id=track.track_id,
                         allowed_split=track.split, track_sampling="fixed")
        env = RacingEnv(config=config, track_catalog={track.track_id: (track, track.split)})
        try:
            observation, _ = env.reset(seed=9000)
        except (ValueError, OSError):
            env.close()
            raise
        self.env.close()
        self.env, self.config, self.observation = env, config, observation
        self.track, self.simulation = track, env.simulation
        self.controller.reset(9000)
        self.last_action = np.zeros(2, dtype=np.float32)
        self.best_lap_s = None
        self.scheduler.clear()
        self.state = AppState.MENU

    def action(self, action):
        if action == "back" and self.state == AppState.MENU:
            self.exit_to_menu, self.running = True, False
            self.scheduler.clear()
        elif action == "reset":
            self.observation, _ = self.env.reset(seed=9000)
            self.controller.reset(9000)
            self.scheduler.clear()
            self.state = AppState.DRIVING
        else:
            super().action(action)

    def frame(self, elapsed, control):
        if self.state != AppState.DRIVING:
            return 0

        def tick():
            action = self.controller.act(self.observation, self.env.snapshot, self.track)
            self.last_action = np.asarray([action.steering, action.longitudinal], dtype=np.float32)
            self.observation, _, terminated, truncated, info = self.env.step(self.last_action)
            if terminated or truncated:
                if info["success"]:
                    lap = info["lap_time_s"]
                    self.best_lap_s = lap if self.best_lap_s is None else min(lap, self.best_lap_s)
                    self.state = AppState.ATTEMPT_FINISHED
                else:
                    self.state = AppState.ATTEMPT_FAILED
                return False
            return True

        return self.scheduler.consume(elapsed, tick)

    def run(self):
        try:
            return super().run()
        finally:
            self.env.close()
