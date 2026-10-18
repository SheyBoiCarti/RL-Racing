"""Iterative routing contracts with native loops and policy loads replaced."""
from types import SimpleNamespace

import pytest

from racing.render import desktop


def arguments(mode="drive"):
    return SimpleNamespace(mode=mode, config="configs/game.yaml", track="demo",
        split_manifest="experiments/splits/v1.json", model="unused.zip" if mode == "watch" else None,
        laps=3, replay="unused.jsonl", replay_output=None, swap_lanes=False,
        model_root=None, tolerant_replay=False, camera=None, difficulty=None)


@pytest.mark.parametrize("initial", ["drive", "race", "replay", "watch"])
def test_routing_back_returns_to_root(initial, monkeypatch, demo_track):
    from racing.agents import policy
    from racing.render import app, race, watch
    routes, preferences = [], []
    class Root:
        def __init__(self, game, physics, track):
            self.game, self.track = game, track
            self.selected_mode = self.selected_policy = None
            self.exit_to_menu = False
        def run(self):
            routes.append("root")
            if initial == "drive" and len(routes) == 1:
                self.selected_mode = "race"
    def race_factory(args, *, preferences):
        assert args.track == "demo"
        preference_objects.append(preferences)
        return SimpleNamespace(track=demo_track, run=lambda: routes.append(args.mode),
                               exit_to_menu=True)
    class Watch:
        def __init__(self, game, controller, track):
            self.track, self.exit_to_menu = track, True
        def run(self):
            routes.append("watch")
    preference_objects = preferences
    monkeypatch.setattr(app, "Application", Root)
    monkeypatch.setattr(race, "application_from_args", race_factory)
    monkeypatch.setattr(watch, "WatchApplication", Watch)
    monkeypatch.setattr(policy, "PolicyController", lambda path: object())
    desktop.run_desktop(arguments(initial))
    assert routes == (["root", "race", "root"] if initial == "drive" else [initial, "root"])
    if preference_objects:
        assert preference_objects[0].difficulty == "medium"
        assert preference_objects[0].camera_mode == "overhead"


@pytest.mark.parametrize("argv", [
    ["--difficulty", "easy"], ["--mode", "watch", "--model", "m.zip", "--camera", "follow"],
    ["--mode", "replay", "--replay", "r.jsonl", "--difficulty", "hard"],
])
def test_cli_rejects_invalid_preference_modes_before_routing(monkeypatch, argv):
    from racing.render.app import main
    monkeypatch.setattr(desktop, "run_desktop", lambda args: pytest.fail("constructed invalid app"))
    with pytest.raises(SystemExit) as error:
        main(argv)
    assert error.value.code == 2


def test_watch_back_closes_environment(demo_track, monkeypatch):
    from racing.core.config import load_game
    from racing.render.app import Application
    from racing.render.watch import WatchApplication
    controller = SimpleNamespace(metadata={"checkpoint_step": 1})
    watch = WatchApplication(load_game(), controller, track=demo_track)
    closed = []
    original_close = watch.env.close
    def close():
        original_close()
        closed.append(True)
    monkeypatch.setattr(watch.env, "close", close)
    monkeypatch.setattr(Application, "run", lambda app: app.action("back"))
    watch.run()
    assert not watch.running and watch.exit_to_menu
    assert closed == [True]


def test_repeated_routing_preserves_preferences_and_selected_track(monkeypatch, demo_track):
    from racing.render import app, chooser, race
    routes, prefs = [], []
    selected = SimpleNamespace(track_id="train_000")
    monkeypatch.setattr(chooser, "TrackChooser", lambda path: SimpleNamespace(
        tracks={"demo": demo_track, "train_000": selected}, ids=["demo", "train_000"],
        index=0, error=None))
    class Root:
        def __init__(self, game, physics, track):
            self.track = track
            self.selected_mode = self.selected_policy = None
        def run(self):
            routes.append("root")
            if routes.count("root") <= 2:
                self.selected_mode, self.track = "race", selected
    def factory(args, *, preferences):
        assert args.track == "train_000"
        prefs.append(preferences)
        if len(prefs) == 1:
            preferences.camera_mode = "follow"
            preferences.difficulty = "easy"
        else:
            assert preferences.camera_mode == "follow" and preferences.difficulty == "easy"
        return SimpleNamespace(track=selected, exit_to_menu=True,
                               run=lambda: routes.append("race"))
    monkeypatch.setattr(app, "Application", Root)
    monkeypatch.setattr(race, "application_from_args", factory)
    desktop.run_desktop(arguments())
    assert routes == ["root", "race", "root", "race", "root"]
    assert prefs[0] is prefs[1]
