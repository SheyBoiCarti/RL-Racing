"""Desktop race lifecycle, playback and command dispatch without a live window."""

from dataclasses import replace

import pytest

from racing.core.config import load_game
from racing.core.models import CarControl
from racing.race.config import RaceConfig
from racing.race.replay import load_replay
from racing.render.race import RaceApplication
from racing.tracks import Track


class ConstantController:
    def __init__(self):
        self.calls = 0

    def reset(self, seed):
        self.calls = 0

    def act(self, observation, snapshot, track):
        self.calls += 1
        return CarControl(0., .1)


@pytest.fixture(scope="module")
def track():
    return Track.from_json("assets/tracks/demo.json")


def application(track, tmp_path, **kwargs):
    return RaceApplication(load_game(), track, controller=ConstantController(),
                           config=kwargs.pop("config", RaceConfig(countdown_seconds=0, race_max_steps=4)),
                           replay_output=tmp_path / "race.jsonl", **kwargs)


def test_countdown_pause_reset_and_abort_playback(track, tmp_path):
    app = application(track, tmp_path, config=RaceConfig(countdown_seconds=3, race_max_steps=4))
    app.action("start")
    assert app.state == "COUNTDOWN"
    for _ in range(60):
        app.frame(.05, CarControl(1., 1.))
    assert app.state == "RUNNING"
    assert app.session.snapshot().tick == 0
    assert app.controller.calls == 0
    app.frame(.075, CarControl(0., .1))
    assert app.session.snapshot().tick == 1
    app.action("pause")
    assert app.scheduler.accumulator == 0
    before = app.session.snapshot()
    app.frame(9, CarControl(1., 1.))
    assert app.session.snapshot() == before
    app.action("restart")
    assert load_replay(tmp_path / "race.jsonl").session.abort_reason == "restart"
    assert app.session.snapshot().tick == 0
    assert app.state == "COUNTDOWN"
    for _ in range(60):
        app.frame(.05, CarControl(0., 0.))
    app.action("quit")
    assert not app.running
    assert len(list(tmp_path.glob("*.jsonl"))) == 2
    assert all(load_replay(path).session.state == "RESULTS" for path in tmp_path.glob("*.jsonl"))


def test_results_and_model_free_playback(track, tmp_path):
    app = application(track, tmp_path)
    app.action("start")
    for _ in range(4):
        app.frame(.05, CarControl(0., .1))
    assert app.state == "RESULTS"
    result = app.session.snapshot()
    calls = app.controller.calls
    app.action("replay")
    assert app.state == "REPLAY"
    app.action("pause")
    app.frame(8, CarControl(0., 1.))
    assert app.replay_index == 0
    app.action("resume")
    for _ in range(4):
        app.frame(.05, CarControl(0., 1.))
    assert app.state == "RESULTS"
    assert app.display_snapshot == result
    assert app.controller.calls == calls
    other = RaceApplication(load_game(), track=None, replay_path=tmp_path / "race.jsonl")
    assert other.controller is None
    assert other.state == "REPLAY"
    other.action("quit")


def test_scheduler_invariance_and_model_error(track, tmp_path):
    snapshots = []
    for fps in (30, 60, 144):
        run = tmp_path / str(fps)
        app = application(track, run, config=RaceConfig(countdown_seconds=0, race_max_steps=100))
        app.action("start")
        for _ in range(fps * 2):
            app.frame(1 / fps, CarControl(0., .1))
        snapshots.append(app.session.snapshot())
        app.action("quit")
    assert snapshots[0] == snapshots[1] == snapshots[2]
    missing = RaceApplication(load_game(), track, model_path=tmp_path / "missing.zip")
    missing.action("start")
    assert missing.state == "MENU"
    assert "missing" in missing.error.lower()


def test_race_view_all_states_and_sizes(track, tmp_path, monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    import pygame

    from racing.render.race_view import RaceView

    pygame.init()
    try:
        app = application(track, tmp_path)
        for size in ((1280, 720), (640, 480), (480, 360)):
            surface = pygame.display.set_mode(size)
            view = RaceView(surface, app.game, app.physics)
            for state in ("MENU", "COUNTDOWN", "RUNNING", "PAUSED", "RESULTS", "REPLAY"):
                app.state = state
                before = app.session.snapshot()
                view.draw(app)
                assert app.session.snapshot() == before
                for rect in view.buttons.values():
                    assert surface.get_rect().contains(rect)
        assert len(view.car_labels) == 2
    finally:
        pygame.quit()


def test_cli_race_and_replay_dispatch(track, tmp_path, monkeypatch):
    from racing.render.app import main

    calls = []
    monkeypatch.setattr(RaceApplication, "run", lambda self: calls.append(self))
    main(["--mode", "race", "--model", str(tmp_path / "missing.zip"), "--track", "demo", "--laps", "2"])
    assert len(calls) == 1 and calls[0].config.target_laps == 2
    with pytest.raises(SystemExit):
        main(["--mode", "race", "--laps", "0"])
    with pytest.raises(SystemExit):
        main(["--mode", "replay"])
    with pytest.raises(SystemExit):
        main(["--mode", "drive", "--replay", "wrong.jsonl"])


def test_sdl_race_launch_quit(track, tmp_path, monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    import pygame

    create = pygame.display.set_mode

    def create_and_quit(*args, **kwargs):
        surface = create(*args, **kwargs)
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        return surface

    monkeypatch.setattr(pygame.display, "set_mode", create_and_quit)
    app = application(track, tmp_path)
    app.run()
    assert not pygame.get_init()


def test_manual_menu_exposes_race_ai(track):
    from racing.core.config import PhysicsConfig
    from racing.render.app import Application

    app = Application(load_game(), PhysicsConfig(), track)
    app.action("race")
    assert app.selected_mode == "race"
    assert not app.running


def test_timestamped_physics_inputs_equal_across_render_rates(track, tmp_path):
    worlds = []
    actions = []
    for fps in (30, 60, 144):
        app = application(track, tmp_path / str(fps),
                          config=RaceConfig(countdown_seconds=0, race_max_steps=100))
        app.action("start")
        seen = []

        def input_at_tick(tick):
            control = CarControl(.07 if tick < 17 else -.04, .3 if tick < 29 else .1)
            seen.append((tick, control))
            return control

        for _ in range(fps * 2):
            app.frame(1 / fps, input_at_tick)
        worlds.append(app.session.snapshot())
        actions.append(seen)
        app.action("quit")
    assert actions[0] == actions[1] == actions[2]
    assert worlds[0] == worlds[1] == worlds[2]
    assert worlds[0].tick == 40


def test_cancel_countdown_does_not_leave_incomplete_replay(track, tmp_path):
    app = application(track, tmp_path, config=RaceConfig(countdown_seconds=3))
    app.action("start")
    app.frame(.05, CarControl(0., 0.))
    app.action("quit")
    assert not list(tmp_path.glob("*.jsonl"))


def test_corrupt_policy_load_stays_at_visible_selection(track, tmp_path, monkeypatch):
    def invalid_archive(*args):
        raise AssertionError("No data found in the saved file")

    monkeypatch.setattr("racing.agents.policy.PolicyController", invalid_archive)
    app = RaceApplication(load_game(), track, model_path=tmp_path / "broken.zip")
    app.action("start")
    assert app.state == "MENU"
    assert "No data" in app.error


def test_playback_retains_recorded_lane_swap(track, tmp_path):
    app = application(track, tmp_path, lane_swap=True)
    app.action("start")
    for _ in range(4):
        app.frame(.05, CarControl(0., .1))
    playback = RaceApplication(load_game(), None, replay_path=app.replay_path)
    assert playback.session.lane_swap is True
    assert playback.lane_swap is True


def test_menu_releases_playback_frames_and_preserves_loaded_policy(track, tmp_path):
    app = application(track, tmp_path)
    app.action("start")
    for _ in range(4):
        app.frame(.05, CarControl(0., .1))
    app.action("replay")
    controller = app.controller
    app.action("menu")
    assert app.playback is None
    assert app.session.snapshot().tick == 0
    assert app.session.state == "MENU"
    assert app.controller is controller


def test_remapped_pause_key_is_respected(track, tmp_path, monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    import pygame

    app = application(track, tmp_path)
    app.game = replace(app.game, bindings=tuple((action, ("e",) if action == "pause" else keys)
                                               for action, keys in app.game.bindings))
    actions = []
    original_action = app.action

    def observe(action):
        original_action(action)
        actions.append((action, app.state))

    app.action = observe
    create = pygame.display.set_mode

    def window(*args, **kwargs):
        surface = create(*args, **kwargs)
        for key in (pygame.K_RETURN, pygame.K_e):
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key))
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        return surface

    monkeypatch.setattr(pygame.display, "set_mode", window)
    app.run()
    assert ("pause", "PAUSED") in actions


def test_debug_event_does_not_restore_elapsed_time_after_start(track, tmp_path, monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    import pygame

    app = application(track, tmp_path)
    create = pygame.display.set_mode

    def window(*args, **kwargs):
        surface = create(*args, **kwargs)
        for key in (pygame.K_RETURN, pygame.K_TAB):
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key))
        return surface

    class Clock:
        def tick(self, fps):
            return 50

    monkeypatch.setattr(pygame.display, "set_mode", window)
    monkeypatch.setattr(pygame.time, "Clock", Clock)
    monkeypatch.setattr(pygame.display, "flip", lambda: pygame.event.post(pygame.event.Event(pygame.QUIT)))
    app.run()
    assert app.session.snapshot().tick == 0
