"""Pure race viewport allocation and fixed north-up full-circuit camera."""

from math import isfinite

import numpy as np
import pygame

from racing.render.style import Camera


def race_world_rect(viewport, *, state, button_rows, debug, min_height=120):
    width, height = viewport
    if (
        type(width) is not int
        or type(height) is not int
        or width < 480
        or height < 360
        or type(button_rows) is not int
        or button_rows < 0
        or type(min_height) is not int
        or min_height < 1
    ):
        raise ValueError("race layout: unsupported viewport or allocation")
    if state not in ("MENU", "COUNTDOWN", "RUNNING", "PAUSED", "RESULTS", "REPLAY"):
        raise ValueError("race layout: unsupported state")
    compact = width < 800 or height < 560
    top = 96 if compact else 124
    bottom = button_rows * 40 + 12 if button_rows else 32
    if debug and not compact:
        top += 96
        if height - top - bottom < min_height:
            top -= 96
    if height - top - bottom < min_height:
        top = 96
    if height - top - bottom < min_height:
        raise ValueError("race layout: insufficient world space")
    return pygame.Rect(12, top, width - 24, height - top - bottom)


def overhead_camera(track, viewport, world_rect, *, margin_m=2.0):
    if (
        type(margin_m) not in (int, float)
        or not isfinite(margin_m)
        or margin_m < 0
        or world_rect.width <= 0
        or world_rect.height <= 0
    ):
        raise ValueError("overhead camera: positive bounds and finite margin required")
    points = np.concatenate((track.outer, track.inner))
    lower, upper = points.min(axis=0) - margin_m, points.max(axis=0) + margin_m
    extent = upper - lower
    if not np.isfinite(points).all() or np.any(extent <= 0):
        raise ValueError("overhead camera: finite nondegenerate track required")
    scale = min(world_rect.width / extent[0], world_rect.height / extent[1])
    bounds_center = (lower + upper) / 2
    cx = bounds_center[0] - (world_rect.centerx - viewport[0] / 2) / scale
    cy = bounds_center[1] - (world_rect.centery - viewport[1] / 2) / scale
    return Camera(center=(float(cx), float(cy)), scale=float(scale), viewport=viewport)
