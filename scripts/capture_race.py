"""Capture actual native race states and replay through the production view.

Use --headless on CI. Captures exercise scripted human inputs, never real user
acceptance, and do not train or evaluate reserved test tracks.
"""

import argparse
import json
import os
from pathlib import Path

from racing.core.config import load_game, project_path
from racing.core.models import CarControl


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/phase_reports/phase_05/captures"))
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args(argv)
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import pygame
    from verify_race import ScriptedCar

    from racing.agents.policy import PolicyController
    from racing.race.replay import load_replay
    from racing.render.race import RaceApplication
    from racing.render.race_view import RaceView
    from racing.tracks import Track

    output = project_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    pygame.init()
    try:
        app = RaceApplication(load_game(), Track.from_json("assets/tracks/demo.json"),
                              controller=PolicyController(project_path(args.model)), model_path=args.model,
                              replay_output=project_path("runs/replays/capture-race.jsonl"))
        driver = ScriptedCar("human")
        manifest = {"display_driver": pygame.display.get_driver(), "human_acceptance": False, "captures": {}}

        def capture(name, current=app, size=(1280, 720)):
            surface = pygame.display.set_mode(size, pygame.RESIZABLE)
            pygame.display.set_caption("RL Racing — Phase 5 verification capture")
            view = RaceView(surface, current.game, current.physics)
            view.draw(current)
            pygame.display.flip()
            pygame.event.pump()
            pygame.image.save(surface, str(output / f"{name}.png"))
            manifest["captures"][name] = dict(state=current.state, tick=current.display_snapshot.tick,
                                               viewport=list(size), source="production application; actual driven/replayed state")

        capture("menu")
        capture("menu-compact", size=(480, 360))
        app.action("start")
        capture("countdown")
        for _ in range(60):
            app.frame(.05, CarControl(0., 0.))
        for _ in range(300):
            app.frame(.05, driver.act(None, app.session.snapshot(), app.track))
        capture("running-opponent")
        capture("running-compact", size=(480, 360))
        app.debug = True
        capture("sensors")
        app.debug = False
        app.action("pause")
        capture("paused")
        app.action("restart")
        capture("restarted-countdown")
        for _ in range(60):
            app.frame(.05, CarControl(0., 0.))
        while app.state == "RUNNING":
            app.frame(.05, driver.act(None, app.session.snapshot(), app.track))
        capture("results")
        capture("results-compact", size=(480, 360))
        load_replay(app.replay_path)
        app.action("replay")
        for _ in range(300):
            app.frame(.05, CarControl(0., 0.))
        capture("replay")
        app.action("pause")
        capture("replay-paused")
        app.action("quit")
        missing = RaceApplication(load_game(), app.track, model_path=project_path("runs/nonexistent-phase05-model.zip"))
        missing.action("start")
        capture("missing-model", missing)
        missing.action("quit")
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(manifest, indent=2))
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
