"""Measure production draw time on moving physics snapshots, including cold caches."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import pygame

from racing.core.config import load_game, load_physics
from racing.core.geometry import wrap
from racing.core.models import CarControl
from racing.render.app import Application
from racing.render.renderer import Renderer
from racing.tracks import Track


def moving_snapshots(app, frames):
    """Pure pursuit from the full-lap integration fixture; profiling only."""
    snapshots = []
    for _ in range(frames):
        car = app.simulation.snapshot().cars[0]
        target, _ = app.track.point_at(car.projection.s_m + 7)
        angle = wrap(math.atan2(target[1] - car.state.y_m, target[0] - car.state.x_m)
                     - car.state.heading_rad)
        distance = math.hypot(target[0] - car.state.x_m, target[1] - car.state.y_m)
        steer = math.atan2(2 * app.physics.wheelbase_m * math.sin(angle), distance) / app.physics.max_steer_rad
        throttle = .25 * (10 - car.state.speed_mps) + app.physics.rolling_drag_mps2 / app.physics.acceleration_mps2
        snapshots.append(app.simulation.step({"human": CarControl(steer, throttle)}).snapshot)
    return snapshots


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--output", type=Path, default=Path("docs/phase_reports/visual_redesign/profile.json"))
    args = parser.parse_args(argv)
    if args.frames <= 0:
        parser.error("--frames must be positive")
    app = Application(load_game(), load_physics(), Track.from_json("assets/tracks/demo.json"))
    snapshots = moving_snapshots(app, args.frames)
    pygame.init()
    try:
        surface = pygame.display.set_mode((1280, 720))
        pygame.display.set_caption("RL Racing render profile")
        renderer = Renderer(surface, app.game, app.physics)
        durations = []
        clock = pygame.time.Clock()
        for snapshot in snapshots:
            pygame.event.pump()
            started = time.perf_counter()
            renderer.draw(app.track, snapshot, "DRIVING", False, None, 0.)
            durations.append((time.perf_counter() - started) * 1000)
            pygame.display.flip()
            clock.tick(60)  # Excluded from draw duration.
        profile = {
            "python": platform.python_version(), "platform": platform.platform(),
            "dependencies": {name: importlib.metadata.version(name) for name in ("pygame", "numpy", "PyYAML", "jsonschema")},
            "video_driver": pygame.display.get_driver(), "viewport": [1280, 720], "frames": args.frames,
            "median_draw_ms": statistics.median(durations),
            "p95_draw_ms": float(np.percentile(durations, 95)),
            "max_draw_ms": max(durations), "first_draw_ms": durations[0],
            "cache": renderer.scene.cache_info(),
            "geometry_hash": hashlib.sha256(app.track.points.astype("<f8").tobytes()).hexdigest(),
            "first_tick": snapshots[0].tick, "last_tick": snapshots[-1].tick,
            "first_position": [snapshots[0].cars[0].state.x_m, snapshots[0].cars[0].state.y_m],
            "last_position": [snapshots[-1].cars[0].state.x_m, snapshots[-1].cars[0].state.y_m],
            "provenance": "physics-driven full-lap fixture inputs; no playable AI mode; cache construction included",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(profile, indent=2), encoding="utf-8")
        print(json.dumps(profile, indent=2))
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
