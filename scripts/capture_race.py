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
    parser.add_argument("--output", type=Path, default=Path("docs/phase_reports/phase_05_usability/captures"))
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--rules", choices=("legacy", "recovery"), default="recovery")
    parser.add_argument("--difficulty", choices=("easy", "medium", "hard"), default=None)
    args = parser.parse_args(argv)
    if args.rules == "legacy" and args.difficulty is not None:
        parser.error("--difficulty requires recovery rules")
    output = project_path(args.output)
    if output.exists():
        raise FileExistsError(f"Capture evidence already exists: {output}")
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import pygame

    from racing.agents.policy import PolicyController
    from racing.race.config import RaceRules
    from racing.race.replay import load_replay
    from racing.render.race import RaceApplication
    from racing.render.race_view import RaceView
    from racing.tracks import Track
    from scripts.verify_race import ScriptedCar

    output = project_path(args.output)
    output.mkdir(parents=True)
    pygame.init()
    try:
        app = RaceApplication(load_game(), Track.from_json("assets/tracks/demo.json"),
                              controller=PolicyController(project_path(args.model)), model_path=args.model,
                              replay_output=output / "capture-race.jsonl",
                              rules=RaceRules(difficulty=args.difficulty or "medium") if args.rules == "recovery" else None)
        driver = ScriptedCar("human")
        manifest = {"display_driver": pygame.display.get_driver(), "human_acceptance": False, "model_sha256": app.controller.metadata["model_sha256"],
                    "native_display": pygame.display.get_driver() != "dummy", "rules": args.rules,
                    "difficulty": args.difficulty or "medium" if args.rules == "recovery" else None,
                    "captures": {}}

        def capture(name, current=app, size=(1280, 720)):
            surface = pygame.display.set_mode(size, pygame.RESIZABLE)
            pygame.display.set_caption("RL Racing — Phase 5 verification capture")
            view = RaceView(surface, current.game, current.physics)
            view.draw(current)
            pygame.display.flip()
            pygame.event.pump()
            pygame.image.save(surface, str(output / f"{name}.png"))
            manifest["captures"][name] = dict(state=current.state, tick=current.display_snapshot.tick,
                                               viewport=list(size), camera=current.preferences.camera_mode,
                                               recovery={key: dict(remaining_ticks=r.remaining_ticks, count=r.count,
                                                   held_ticks=r.held_ticks, last_release_tick=r.last_release_tick)
                                                   for key, r in current.display_recovery.items()}, source="production application; actual driven/replayed state")

        capture("menu")
        capture("menu-compact", size=(480, 360))
        app.debug = True
        capture("menu-compact-debug", size=(480, 360))
        app.debug = False
        app.action("start")
        capture("countdown")
        for _ in range(60):
            app.frame(.05, CarControl(0., 0.))
        if args.rules == "recovery":
            for _ in range(200):
                app.frame(.05, CarControl(1., 1.))
                if app.display_recovery["human"].count:
                    break
            if app.display_recovery["human"].remaining_ticks != 20:
                raise RuntimeError("Actual off-track driving did not enter recovery")
            capture("recovery-countdown")
            for _ in range(20):
                app.frame(.05, CarControl(0., 0.))
            capture("recovery-go")
        capture("overhead")
        app.action("toggle_camera")
        capture("follow")
        app.action("toggle_camera")
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
