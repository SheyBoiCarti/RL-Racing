"""One tick per action Gymnasium adapter over the shared simulator."""

from collections.abc import Mapping
from dataclasses import asdict

import gymnasium as gym
import numpy as np

from racing.core.models import CarControl, CarSpawn, CarStatus, EventKind
from racing.core.simulation import Simulation
from racing.envs.config import RLConfig
from racing.envs.observation import ObservationBuilder
from racing.envs.reward import RewardBreakdown, RewardCalculator
from racing.envs.termination import EpisodeOutcome, TerminationEvaluator
from racing.tracks.loader import Track


class RacingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self, config=None, track_catalog=None, render_mode=None):
        if render_mode not in (None, "human"):
            raise ValueError("render_mode: expected None or human")
        self.config = config or RLConfig()
        self.render_mode = render_mode
        # Catalog entries may be Track, or (Track, split). Unknown provenance
        # is rejected; the bundled demo has explicit known split metadata.
        self.track_catalog = (
            dict(track_catalog)
            if track_catalog is not None
            else {"demo": (Track.from_json("assets/tracks/demo.json"), "demo")}
        )
        if track_catalog is None and self.config.split_manifest:
            from racing.tracks.splits import load_split_catalog

            self.track_catalog = load_split_catalog(self.config.split_manifest, self.config.allowed_split)
        if self.config.track_sampling == "uniform_train":
            if not self.track_catalog or any(
                getattr(entry[0] if isinstance(entry, tuple) else entry, "split", None) != "train"
                for entry in self.track_catalog.values()
            ):
                raise ValueError("uniform_train requires a nonempty train-only catalog")
        self.action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-1, 1, (11,), dtype=np.float32)
        self.simulation = Simulation(self.config.physics)
        self.observations = ObservationBuilder(self.config)
        self.rewards = RewardCalculator(self.config)
        self._renderer = None
        self._done = True

    @property
    def snapshot(self):
        return self.simulation.snapshot()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options is None:
            options = {}
        if not isinstance(options, Mapping) or options.keys() - {
            "track_id",
            "spawn_id",
        }:
            raise ValueError("options: only track_id and spawn_id supported")
        if "track_id" not in options and self.config.track_sampling == "uniform_train":
            ids = sorted(self.track_catalog)
            track_id = ids[int(self.np_random.integers(len(ids)))]
        else:
            track_id = options.get("track_id", self.config.default_track_id)
        spawn_id = options.get("spawn_id", self.config.default_spawn_id)
        if not isinstance(track_id, str) or track_id not in self.track_catalog:
            raise ValueError("track_id: unknown track")
        entry = self.track_catalog[track_id]
        if isinstance(entry, tuple) and len(entry) == 2:
            track, split = entry
            if split != getattr(track, "split", None):
                raise ValueError("track_catalog: split conflicts with track metadata")
        else:
            track = entry
            split = getattr(track, "split", None)
        if split != self.config.allowed_split:
            raise ValueError(
                "track_id: outside allowed_split or missing split metadata"
            )
        if not isinstance(track, Track) or track.track_id != track_id:
            raise ValueError("track_catalog: Track ID mismatch")
        if not isinstance(spawn_id, str) or spawn_id not in track.spawns:
            raise ValueError("spawn_id: unknown spawn")
        self.track = track
        self.reset_seed = seed
        self.simulator_seed = int(self.np_random.integers(0, 2**32, dtype=np.uint32))
        self.simulation.reset(track, self.simulator_seed, [CarSpawn("agent", spawn_id)])
        self._steps = self._no_progress = 0
        self._done = False
        return self.observations.build(self.snapshot, track, "agent"), self._info(
            EpisodeOutcome(), RewardBreakdown(), False
        )

    def step(self, action):
        if self._done:
            raise RuntimeError("reset required before stepping a completed episode")
        try:
            raw = np.asarray(action)
            if (
                raw.dtype.kind not in "fiu"
                or raw.shape != (2,)
                or not np.isfinite(raw).all()
            ):
                raise ValueError("action: finite numeric shape (2,) required")
            values = np.clip(raw, -1, 1).astype(np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("action: finite numeric shape (2,) required") from error
        control = CarControl(float(values[0]), float(values[1]))
        result = self.simulation.step({"agent": control})
        car = result.snapshot.cars[0]
        self._steps += 1
        self._no_progress = (
            0
            if car.rewardable_delta_m > self.config.progress_threshold_m
            else self._no_progress + 1
        )
        outcome = TerminationEvaluator.check(
            car.status == CarStatus.DISQUALIFIED,
            car.completed_laps,
            self._steps,
            self._no_progress,
            self.config,
        )
        reward = self.rewards.score(car.rewardable_delta_m, control, outcome)
        self._done = outcome.terminated or outcome.truncated
        lap_completed = any(
            event.kind == EventKind.LAP_COMPLETED for event in result.events
        )
        observation = self.observations.build(self.snapshot, self.track, "agent")
        info = self._info(outcome, reward, lap_completed)
        return observation, reward.total, outcome.terminated, outcome.truncated, info

    def _info(self, outcome, reward, lap_completed):
        world, car = self.snapshot, self.snapshot.cars[0]
        fraction = car.high_water_m / (
            car.approach_distance_m + self.config.target_laps * self.track.length_m
        )
        return {
            "track_id": world.track_id,
            "reset_seed": self.reset_seed,
            "simulator_seed": self.simulator_seed,
            "tick": int(world.tick),
            "completed_laps": int(car.completed_laps),
            "success": bool(outcome.success),
            "lap_completed": bool(lap_completed),
            "off_track": car.status == CarStatus.DISQUALIFIED,
            "episode_reason": outcome.reason,
            "signed_delta_m": float(car.signed_delta_m),
            "rewardable_delta_m": float(car.rewardable_delta_m),
            "progress_m": float(car.high_water_m),
            "progress_fraction": 1.0
            if outcome.success
            else float(np.clip(fraction, 0, 1)),
            "no_progress_steps": int(self._no_progress),
            "lap_time_s": float(car.completed_lap_times_s[-1])
            if car.completed_lap_times_s
            else None,
            "reward_terms": asdict(reward),
            "projection_discontinuity": bool(car.projection_discontinuity),
        }

    def render(self):
        if self.render_mode is None:
            return None
        import pygame

        from racing.core.config import load_game
        from racing.envs.sensors import cast_rays
        from racing.render.renderer import Renderer
        from racing.render.style import Camera

        if self._renderer is None:
            pygame.init()
            game = load_game()
            surface = pygame.display.set_mode((game.width_px, game.height_px))
            self._renderer = Renderer(surface, game, self.config.physics)
        pygame.event.pump()
        self._renderer.draw(self.track, self.snapshot, "DRIVING", True, None, 0.0)
        game = self._renderer.config
        pose = self.snapshot.cars[0].state
        center = (
            (pose.x_m, pose.y_m)
            if game.follow_camera
            else tuple(
                (self.track.outer.min(axis=0) + self.track.outer.max(axis=0)) / 2
            )
        )
        camera = Camera(
            center, game.pixels_per_metre, self._renderer.surface.get_size()
        )
        distances = cast_rays(
            self.track, pose, self.config.ray_angles_deg, self.config.ray_range_m
        )
        origin = (pose.x_m, pose.y_m)
        for angle, distance in zip(self.config.ray_angles_deg, distances):
            heading = pose.heading_rad + np.radians(angle)
            end = (
                pose.x_m + distance * np.cos(heading),
                pose.y_m + distance * np.sin(heading),
            )
            pygame.draw.line(
                self._renderer.surface,
                (100, 225, 240),
                camera.to_screen(origin),
                camera.to_screen(end),
                1,
            )
        pygame.display.flip()

    def close(self):
        if self._renderer is not None:
            import pygame

            pygame.display.quit()
            self._renderer = None
