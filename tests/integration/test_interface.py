"""Exercise real UI drawing at reference and compact sizes."""

from itertools import combinations

import pygame
import pytest

from racing.core.config import PhysicsConfig, load_game
from racing.render.interface import RaceInterface


@pytest.mark.parametrize("size", [(1280, 720), (480, 360)])
@pytest.mark.parametrize("state,actions", [
    ("MENU", {"reset", "watch", "quit"}), ("PAUSED", {"resume", "reset", "menu"}),
    ("ATTEMPT_FAILED", {"reset", "menu"}),
    ("ATTEMPT_FINISHED", {"reset", "menu"}), ("DRIVING", set()),
])
def test_interface_actions_fit_each_state(pygame_env, demo_track, world, size, state, actions):
    interface = RaceInterface(load_game(), PhysicsConfig())
    surface = pygame.Surface(size)
    before = world
    rects = interface.draw(surface, demo_track, world, state, None, 0.)
    assert set(rects) == actions
    for rect in rects.values():
        assert surface.get_rect().contains(rect)
        assert rect.width >= 120 and rect.height >= 40
    for first, second in combinations(rects.values(), 2):
        assert not first.colliderect(second)
    assert world == before
    assert interface.primary_font_px >= 18
    assert interface.secondary_font_px >= 14
