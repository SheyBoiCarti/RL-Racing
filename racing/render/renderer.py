"""Stable renderer adapter for world art and race presentation."""

import math

import pygame

from racing.render.interface import RaceInterface
from racing.render.scene import TrackScene
from racing.render.style import PALETTE as P
from racing.render.style import Camera, font


def debug_readout(track, snapshot, dropped_time):
    projection = snapshot.cars[0].projection
    i = projection.segment_index
    a, b = track.tangents[i - 1], track.tangents[i]
    turn = math.atan2(a[0] * b[1] - a[1] * b[0], float(a @ b))
    curvature = turn / ((track.lengths[i - 1] + track.lengths[i]) / 2)
    return (f"tick {snapshot.tick} · projection {projection.s_m:.1f}m · "
            f"segment {i}:{projection.segment_t:.2f} · lateral {projection.lateral_m:+.2f}m · "
            f"curvature {curvature:+.4f}/m · dropped {dropped_time:.3f}s")


class Renderer:
    def __init__(self, surface, config, physics):
        self.surface, self.config, self.physics = surface, config, physics
        self.scene = TrackScene(physics)
        self.interface = RaceInterface(config, physics)
        self.buttons = {}
        self.pointer = None
        self.policy_debug = None

    def draw(self, track, snapshot, state, debug, best_lap, dropped_time):
        low, high = track.outer.min(axis=0), track.outer.max(axis=0)
        viewport = self.surface.get_size()
        center = tuple((low + high) / 2)
        scale = self.config.pixels_per_metre
        if state == "MENU":
            scale = min((viewport[0] - 48) / (high[0] - low[0] + 12),
                        (viewport[1] - 48) / (high[1] - low[1] + 12))
        elif self.config.follow_camera:
            car = snapshot.cars[0].state
            center = (car.x_m, car.y_m)
        camera = Camera(center, scale, viewport)
        self.scene.draw(self.surface, track, snapshot, camera, debug)
        self.buttons = self.interface.draw(self.surface, track, snapshot, state,
                                           best_lap, 0., self.pointer)
        if debug:
            # Bottom gap is reserved between corner panels and control hint.
            # Modal screens put diagnostics below all actions instead.
            strip = pygame.Rect(8, viewport[1] - (54 if state == "DRIVING" else 24),
                                viewport[0] - 16, 20)
            pygame.draw.rect(self.surface, P["panel"], strip)
            text = debug_readout(track, snapshot, dropped_time)
            if self.policy_debug:
                text += f" · {self.policy_debug}"
            if font(14).size(text)[0] > strip.width - 8:
                # Keep projection and curvature visible together at minimum size.
                text = text[text.index("projection"):text.index(" · dropped")]
                text = text.replace("projection", "proj").replace("segment", "seg")
            size = 14
            while size > 9 and font(size).size(text)[0] > strip.width - 8:
                size -= 1
            self.surface.blit(font(size).render(text, True, P["secondary"]), (strip.x + 4, strip.y + 2))
