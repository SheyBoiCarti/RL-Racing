"""Rendering must be deterministic, immutable and bounded at custom scales."""

import copy
import math
import pickle
import random
from dataclasses import replace

import numpy as np
import pygame
import pytest

from racing.core.config import PhysicsConfig
from racing.render.scene import TrackScene
from racing.render.style import Camera


def test_scene_is_pure_and_repeatable(pygame_env, demo_track, world):
    before = copy.deepcopy(world)
    py_rng = random.getstate()
    np_rng = pickle.dumps(np.random.get_state())
    surfaces = [pygame.Surface((640, 480)) for _ in range(2)]
    for surface in surfaces:
        TrackScene(PhysicsConfig()).draw(surface, demo_track, world,
                                        Camera((0., 0.), 3., (640, 480)), True)
    assert world == before
    assert random.getstate() == py_rng
    assert pickle.dumps(np.random.get_state()) == np_rng
    assert pygame.image.tobytes(surfaces[0], "RGB") == pygame.image.tobytes(surfaces[1], "RGB")


def test_scene_cache_is_bounded_and_invalidates(pygame_env, demo_track, world, monkeypatch):
    scene = TrackScene(PhysicsConfig())
    surface = pygame.Surface((480, 360))
    allocate = pygame.Surface
    allocations = []

    def record(size, *args, **kwargs):
        allocations.append(size)
        return allocate(size, *args, **kwargs)

    monkeypatch.setattr(pygame, "Surface", record)
    for scale in (18., 500.):
        for index in range(40):
            scene.draw(surface, demo_track, world,
                       Camera((index * 512 / scale, 0.), scale, (480, 360)), False)
            info = scene.cache_info()
            assert info["tiles"] <= 32
            assert info["bytes"] <= 32 * 512 * 512 * 4
        generation = info["generation"]
    scene.draw(surface, demo_track, world, Camera((0., 0.), 18., (480, 360)), False)
    assert scene.cache_info()["generation"] > generation
    assert allocations and all(size == (512, 512) for size in allocations)
    generation = scene.cache_info()["generation"]
    scene.draw(surface, copy.copy(demo_track), world,
               Camera((0., 0.), 18., (480, 360)), False)
    assert scene.cache_info()["generation"] > generation


@pytest.mark.parametrize("heading,axis,sign", [(0., 0, 1), (math.pi / 2, 1, 1),
                                             (-math.pi / 2, 1, -1)])
def test_car_front_rotates_clockwise(pygame_env, world, heading, axis, sign):
    state = replace(world.cars[0].state, x_m=0., y_m=0., heading_rad=heading)
    before = copy.deepcopy(state)
    surface = pygame.Surface((200, 200))
    surface.fill((0, 0, 0))
    TrackScene(PhysicsConfig()).draw_car(surface, state, Camera((0., 0.), 40., (200, 200)))
    pixels = pygame.surfarray.array3d(surface)
    coords = np.argwhere(np.all(pixels == (255, 243, 191), axis=2))
    assert len(coords)
    assert (coords[:, axis].mean() - 100) * sign > 0
    assert state == before


@pytest.mark.parametrize("scale", [500., 10000.])
def test_large_scale_car_does_not_allocate_huge_sprite(pygame_env, world, monkeypatch, scale):
    surface = pygame.Surface((480, 360))
    original = pygame.Surface
    allocations = []

    def allocate(size, *args, **kwargs):
        allocations.append(size)
        return original(size, *args, **kwargs)

    monkeypatch.setattr(pygame, "Surface", allocate)
    state = replace(world.cars[0].state, x_m=0., y_m=0.)
    TrackScene(PhysicsConfig()).draw_car(surface, state, Camera((0., 0.), scale, surface.get_size()))
    assert all(max(size) <= 512 for size in allocations)


@pytest.mark.parametrize("scale", [10000., 1e6])
def test_extreme_scale_scene_clips_raster_polygons(pygame_env, demo_track, world, monkeypatch, scale):
    surface = pygame.Surface((480, 360))
    original = pygame.draw.polygon
    calls = []

    def bounded_polygon(destination, color, points, *args, **kwargs):
        width, height = destination.get_size()
        # SDL scanline work must depend on viewport, not projected world size.
        assert all(0 <= x <= width and 0 <= y <= height for x, y in points)
        calls.append(points)
        return original(destination, color, points, *args, **kwargs)

    monkeypatch.setattr(pygame.draw, "polygon", bounded_polygon)
    car = world.cars[0].state
    TrackScene(PhysicsConfig()).draw(surface, demo_track, world,
        Camera((car.x_m, car.y_m), scale, surface.get_size()), False)
    assert calls
