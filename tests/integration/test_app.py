import copy
import os
import pickle
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from racing.core.config import PhysicsConfig, load_game
from racing.core.models import CarControl
from racing.render.app import Application, AppState
from racing.tracks import Track


@pytest.fixture(scope="module")
def track():
    return Track.from_json("assets/tracks/demo.json")


def test_application_lifecycle(track):
    app = Application(load_game(), PhysicsConfig(), track)
    assert app.state == AppState.MENU
    assert app.frame(10, CarControl(0.0, 1.0)) == 0
    app.action("reset")
    app.frame(0.075, CarControl(0.0, 1.0))
    assert app.simulation.snapshot().tick == 1
    app.action("pause")
    assert app.state == AppState.PAUSED and app.scheduler.accumulator == 0
    assert app.frame(20, CarControl(0.0, 1.0)) == 0
    app.action("resume")
    assert app.state == AppState.DRIVING
    app.frame(0.025, CarControl(0.0, 1.0))
    assert app.simulation.snapshot().tick == 1
    app.action("pause")
    app.action("pause")
    assert app.state == AppState.MENU
    for state in AppState:
        app.state = state
        app.action("reset")
        assert app.state == AppState.DRIVING and app.simulation.snapshot().tick == 0
    for _ in range(200):
        app.frame(0.05, CarControl(0.0, 1.0))
        if app.state == AppState.ATTEMPT_FAILED:
            break
    assert app.state == AppState.ATTEMPT_FAILED
    snapshot = app.simulation.snapshot()
    app.frame(100, CarControl(0.0, 1.0))
    assert app.simulation.snapshot() == snapshot
    app.action("debug")
    assert app.debug
    app.action("menu")
    assert app.state == AppState.MENU


def test_scheduler_states_equal_across_fps(track):
    snapshots = []
    for fps in (30, 60, 144):
        app = Application(load_game(), PhysicsConfig(), track)
        app.action("reset")
        for _ in range(fps * 2):
            app.frame(1 / fps, CarControl(0.1, 0.3))
        snapshots.append(app.simulation.snapshot())
    assert snapshots[0] == snapshots[1] == snapshots[2]
    assert snapshots[0].tick == 40
    assert snapshots[0].cars[0].state.speed_mps > 0


def test_sdl_launch_close_and_quit_finally():
    script = """
import pygame
from racing.core.config import load_game, load_physics
from racing.render.app import Application
from racing.tracks import Track
original = pygame.display.set_mode
def create(*args, **kwargs):
    surface = original(*args, **kwargs)
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    return surface
pygame.display.set_mode = create
Application(load_game(), load_physics(), Track.from_json("assets/tracks/demo.json")).run()
assert not pygame.get_init()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_renderer_resize_and_debug(track, monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    import pygame

    from racing.render.renderer import Renderer

    pygame.init()
    try:
        app = Application(load_game(), PhysicsConfig(), track)
        snapshot = app.simulation.snapshot()
        before = copy.deepcopy(snapshot)
        python_rng = random.getstate()
        numpy_rng = pickle.dumps(np.random.get_state())
        for size in ((1280, 720), (640, 480), (480, 360)):
            surface = pygame.display.set_mode(size)
            renderer = Renderer(surface, app.game, app.physics)
            for state in AppState:
                renderer.draw(
                    track, app.simulation.snapshot(), state.value, True, None, 0
                )
                expected = {
                    AppState.MENU: {"reset", "watch", "quit"},
                    AppState.PAUSED: {"resume", "reset", "menu"},
                    AppState.ATTEMPT_FAILED: {"reset", "menu"},
                    AppState.ATTEMPT_FINISHED: {"reset", "menu"},
                    AppState.DRIVING: set(),
                }[state]
                assert set(renderer.buttons) == expected
            assert renderer.buttons["reset"].right <= size[0]
        assert app.simulation.snapshot() == before
        assert snapshot == before
        assert random.getstate() == python_rng
        assert pickle.dumps(np.random.get_state()) == numpy_rng
    finally:
        pygame.quit()


def test_cli_paths_and_rejected_modes(tmp_path):
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(root / "scripts/play.py"), "--mode", "watch"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2 and "requires --model" in result.stderr
    script = "from racing.core.config import load_game; from racing.tracks import Track; assert Track.from_json('assets/tracks/demo.json').track_id == 'demo'; assert load_game().render_fps == 60"
    subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, check=True, capture_output=True
    )


@pytest.mark.parametrize("with_debug", [False, True])
def test_event_loop_uses_resized_buttons_and_freezes_pause(pygame_env, demo_track, monkeypatch, with_debug):
    import pygame

    from racing.render.style import layout

    app = Application(load_game(), PhysicsConfig(), demo_track)
    batches = [
        (50, [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN)]),
        (50, []),
        (50, [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)]),
        (20000, []),
        (20000, [pygame.event.Event(pygame.VIDEORESIZE, w=480, h=360),
                 pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1,
                                    pos=layout((480, 360), "PAUSED")["resume"].center)]),
        (50, []),
        (50, [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r, repeat=True)]),
        (50, [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r)]),
        (50, [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)]),
        (50, [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)]),
        (50, [pygame.event.Event(pygame.QUIT)]),
    ]
    observations = []
    if with_debug:
        batches[4][1].append(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_TAB))
        batches[7][1].append(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_TAB))
    current = []

    class Clock:
        def tick(self, fps):
            observations.append((app.state, app.simulation.snapshot()))
            elapsed, events = batches.pop(0)
            current[:] = events
            return elapsed

    monkeypatch.setattr(pygame.time, "Clock", Clock)
    monkeypatch.setattr(pygame.event, "get", lambda: current[:])
    app.run()
    assert observations[3][0] == AppState.PAUSED
    assert observations[3][1] == observations[4][1]
    assert observations[5][0] == AppState.DRIVING
    assert observations[6][1].tick == observations[3][1].tick + 1
    assert observations[7][1].tick == observations[6][1].tick + 1
    assert observations[8][1].tick == 0
    assert observations[10][0] == AppState.MENU
    assert not pygame.get_init()


def test_debug_keeps_player_car_visible(pygame_env, demo_track, world):
    import pygame

    from racing.render.renderer import Renderer
    from racing.render.style import PALETTE

    for size in ((1280, 720), (480, 360)):
        surface = pygame.Surface(size)
        Renderer(surface, load_game(), PhysicsConfig()).draw(demo_track, world,
                                                            "DRIVING", True, None, 0.)
        center = surface.get_rect().center
        pixels = pygame.surfarray.array3d(surface)[center[0]-6:center[0]+6,
                                                  center[1]-6:center[1]+6]
        assert np.any(np.all(pixels == PALETTE["accent"], axis=2))
