"""Desktop session preferences shared across successive application routes."""

from dataclasses import dataclass


@dataclass
class DesktopPreferences:
    camera_mode: str = "overhead"
    difficulty: str = "medium"

    def __post_init__(self):
        if self.camera_mode not in ("overhead", "follow"):
            raise ValueError("camera_mode: overhead or follow required")
        if self.difficulty not in ("easy", "medium", "hard"):
            raise ValueError("difficulty: easy, medium or hard required")


def run_desktop(args):
    """Own one application at a time and return to root without recursive runs."""
    from copy import copy

    from racing.core.config import load_game, load_physics
    from racing.render.app import Application
    from racing.render.chooser import TrackChooser
    from racing.render.race import application_from_args

    preferences = DesktopPreferences(camera_mode=args.camera or "overhead",
                                     difficulty=args.difficulty or "medium")
    game = load_game(args.config)
    chooser = TrackChooser(args.split_manifest)
    track = chooser.tracks.get(args.track, chooser.tracks["demo"])
    if args.track not in chooser.tracks:
        reason = f"; {chooser.error}" if chooser.error else ""
        chooser.error = f"{args.track}.json: track unavailable in demo/train/validation catalog{reason}"
    route, policy = args.mode, None
    first = True
    while True:
        current_route = route
        if route in ("race", "replay"):
            launch = copy(args)
            launch.mode, launch.track = route, track.track_id
            app = application_from_args(launch, preferences=preferences)
        elif route == "watch":
            from racing.agents.policy import PolicyController
            from racing.render.watch import WatchApplication
            try:
                app = WatchApplication(game, policy if policy is not None else PolicyController(args.model), track=track)
            except (ValueError, OSError, ImportError) as error:
                chooser.error = f"{args.model}: {error}"
                current_route = "drive"
                app = Application(game, load_physics(), track)
        else:
            app = Application(game, load_physics(), track)
        app.preferences = preferences
        app.chooser, app.split_manifest = chooser, args.split_manifest
        app.choosing_track = bool(chooser.error) or (first and route in ("drive", "watch"))
        if app.track.track_id in chooser.ids:
            chooser.index = chooser.ids.index(app.track.track_id)
        app.run()
        track = app.track
        first = False
        if current_route == "drive" and getattr(app, "selected_mode", None) == "race":
            route = "race"
        elif current_route == "drive" and getattr(app, "selected_policy", None) is not None:
            route, policy = "watch", app.selected_policy
        elif getattr(app, "exit_to_menu", False):
            route, policy = "drive", None
        else:
            break
        del app
