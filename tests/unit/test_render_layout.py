"""Catch inverted camera coordinates, clipped controls and overlapping HUD."""

from itertools import combinations

import pygame
import pytest

from racing.render.style import Camera, layout


def test_camera_clockwise_coordinates():
    camera = Camera((10., 20.), 18., (1280, 720))
    assert camera.to_screen((10., 20.)) == (640., 360.)
    assert camera.to_screen((11., 21.)) == (658., 378.)
    assert camera.to_world((658., 378.)) == (11., 21.)


@pytest.mark.parametrize("size", [(1280, 720), (640, 480), (480, 360)])
@pytest.mark.parametrize("state,actions", [
    ("MENU", {"reset", "watch", "race", "quit"}),
    ("PAUSED", {"resume", "reset", "menu"}),
    ("ATTEMPT_FAILED", {"reset", "menu"}),
    ("ATTEMPT_FINISHED", {"reset", "menu"}),
    ("DRIVING", set()),
])
def test_layout_fits_minimum_viewport(size, state, actions):
    rects = layout(size, state)
    hud = {"lap", "timer", "speed", "minimap", "hint"}
    assert set(rects) - hud - {"modal"} == actions
    viewport = pygame.Rect((0, 0), size)
    for rect in rects.values():
        assert viewport.contains(rect)
    for action in actions:
        assert rects[action].width >= 120 and rects[action].height >= 40
    for first, second in combinations(actions, 2):
        assert not rects[first].colliderect(rects[second])
    for first, second in combinations(hud, 2):
        assert not rects[first].colliderect(rects[second])
