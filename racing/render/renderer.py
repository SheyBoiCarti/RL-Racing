"""Stable renderer adapter for world art and race presentation."""

import pygame

from racing.render.interface import RaceInterface
from racing.render.scene import TrackScene
from racing.render.style import PALETTE as P
from racing.render.style import Camera, font


class Renderer:
    def __init__(self, surface, config, physics):
        self.surface, self.config, self.physics = surface, config, physics
        self.scene = TrackScene(physics)
        self.interface = RaceInterface(config, physics)
        self.buttons = {}
        self.pointer = None

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
            car = snapshot.cars[0]
            text = f"tick {snapshot.tick} · s {car.projection.s_m:.1f}m · gate {car.next_checkpoint} · dropped {dropped_time:.3f}s"
            self.surface.blit(font(14).render(text, True, P["secondary"]), (strip.x + 4, strip.y + 2))
