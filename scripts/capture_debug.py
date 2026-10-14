"""Reproducible captures through the production renderer."""

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from racing.core.config import load_game, load_physics, project_path
from racing.core.models import CarStatus
from racing.render.app import Application
from racing.render.renderer import Renderer
from racing.tracks import Track


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    pygame.init()
    try:
        config = load_game()
        app = Application(config, load_physics(), Track.from_json("assets/tracks/demo.json"))
        world = app.simulation.snapshot()
        if not args.suite:
            config = replace(config, follow_camera=False, pixels_per_metre=2.7)
            surface = pygame.display.set_mode((1280, 800))
            Renderer(surface, config, app.physics).draw(app.track, world, "DRIVING", True, None, 0.)
            path = args.output or project_path("docs/phase_reports/phase_01_debug.png")
            path.parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(surface, str(path))
            print(path)
            return
        output = args.output or project_path("docs/phase_reports/visual_redesign")
        output.mkdir(parents=True, exist_ok=True)
        manifest = {}
        cases = [("menu", "MENU"), ("driving", "DRIVING"), ("paused", "PAUSED"),
                 ("failed", "ATTEMPT_FAILED"), ("finished", "ATTEMPT_FINISHED"),
                 ("debug_overview", "DRIVING"), ("menu_compact", "MENU"),
                 ("driving_compact", "DRIVING")]
        for name, state in cases:
            size = (480, 360) if name.endswith("compact") else (1280, 720)
            surface = pygame.display.set_mode(size)
            debug = name == "debug_overview"
            settings = replace(config, follow_camera=False, pixels_per_metre=2.7) if debug else config
            fixture = world
            provenance = "seed 0 solo reset; production simulator snapshot"
            best = None
            if name == "finished":
                car = replace(world.cars[0], status=CarStatus.FINISHED, completed_laps=1,
                              completed_lap_times_s=(62.35,), lap_start_tick=1300)
                fixture = replace(world, tick=1300, time_s=65., cars=(car,))
                provenance = "synthetic completed result via dataclasses.replace; not a driven lap"
                best = 61.5
            elif name == "failed":
                car = replace(world.cars[0], status=CarStatus.DISQUALIFIED)
                fixture = replace(world, cars=(car,))
                provenance = "synthetic failure via dataclasses.replace; not a driven failure"
            Renderer(surface, settings, app.physics).draw(app.track, fixture, state, debug, best, 0.)
            pygame.image.save(surface, str(output / f"{name}.png"))
            manifest[name] = {"state": state, "viewport": size, "provenance": provenance,
                              "debug": debug, "tick": fixture.tick}
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(output)
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
