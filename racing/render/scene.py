"""Deterministic circuit tiles and snapshot-only world annotations."""

import math
import random
from collections import OrderedDict

import numpy as np
import pygame

from racing.render.style import PALETTE as P
from racing.render.style import Camera, font

TILE = 512
MAX_TILES = 32


def clipped_polygon(surface, color, points):
    """Clip in float space before SDL scans projected, potentially huge edges."""
    points = list(points)
    width, height = surface.get_size()
    for axis, bound, sign in ((0, 0., 1), (0, float(width), -1),
                              (1, 0., 1), (1, float(height), -1)):
        if not points:
            return
        output = []
        previous = points[-1]
        previous_inside = (previous[axis] - bound) * sign >= 0
        for current in points:
            current_inside = (current[axis] - bound) * sign >= 0
            if current_inside != previous_inside:
                fraction = (bound - previous[axis]) / (current[axis] - previous[axis])
                intersection = [previous[i] + fraction * (current[i] - previous[i]) for i in (0, 1)]
                intersection[axis] = bound
                output.append(tuple(intersection))
            if current_inside:
                output.append(current)
            previous, previous_inside = current, current_inside
        points = output
    if len(points) >= 3:
        # Tiny rounding drift at an earlier clipping edge is harmless, but don't
        # let it escape the bounded raster contract.
        pygame.draw.polygon(surface, color, [(min(width, max(0., x)), min(height, max(0., y)))
                                             for x, y in points])


class TrackScene:
    def __init__(self, physics):
        self.physics = physics
        self._track = None
        self._scale = None
        self._tiles = OrderedDict()
        self._generation = 0
        self._decor = []
        self._fences = []

    def cache_info(self):
        return {"tiles": len(self._tiles), "bytes": sum(
            tile.get_pitch() * tile.get_height() for tile in self._tiles.values()),
            "generation": self._generation}

    def _prepare(self, track, scale):
        changed_track = self._track is not track
        if changed_track or self._scale != scale:
            self._tiles.clear()
            self._generation += 1
            self._track, self._scale = track, scale
        if changed_track:
            rng = random.Random(8128)
            low, high = track.outer.min(axis=0) - 35, track.outer.max(axis=0) + 35
            self._decor = []
            self._fences = []
            starts = np.array([a for a, b in track.boundary_segments])
            edges = np.array([b - a for a, b in track.boundary_segments])
            squared = np.sum(edges * edges, axis=1)

            def inside_loop(point, loop):
                a, b = loop, np.roll(loop, -1, axis=0)
                crosses = (a[:, 1] > point[1]) != (b[:, 1] > point[1])
                delta = b[:, 1] - a[:, 1]
                cut = a[:, 0] + (point[1] - a[:, 1]) * (b[:, 0] - a[:, 0]) / np.where(delta != 0, delta, 1)
                return bool(np.count_nonzero(crosses & (point[0] < cut)) % 2)

            for index in range(170):
                x, y = (rng.uniform(low[i], high[i]) for i in (0, 1))
                # Exclude legal road and a clear verge, using authoritative geometry.
                point = np.array((x, y))
                if not (inside_loop(point, track.outer) and not inside_loop(point, track.inner)):
                    t = np.clip(np.sum((point - starts) * edges, axis=1) / squared, 0, 1)
                    distances = np.linalg.norm(point - starts - t[:, None] * edges, axis=1)
                    if distances.min() > 4:
                        self._decor.append((x, y, rng.uniform(1.2, 2.6)))
                        if index % 17 == 0 and distances.min() > 12:
                            self._fences.append(((x - 4, y + 5), (x + 4, y + 5)))

    def _tile(self, track, scale, tx, ty):
        key = (tx, ty)
        if key in self._tiles:
            self._tiles.move_to_end(key)
            return self._tiles[key]
        tile = pygame.Surface((TILE, TILE), pygame.SRCALPHA)
        camera = Camera(((tx + .5) * TILE / scale, (ty + .5) * TILE / scale),
                        scale, (TILE, TILE))
        screen = camera.to_screen
        tile.fill(P["grass"])
        rng = random.Random(f"circuit:{tx}:{ty}:{scale:.12g}")
        for _ in range(150):
            x, y = rng.randrange(TILE), rng.randrange(TILE)
            pygame.draw.line(tile, rng.choice((P["grass_light"], P["grass_dark"])),
                             (x, y), (x + 5, y + 2))
        road_mask = pygame.Surface((TILE, TILE), pygame.SRCALPHA)
        clipped_polygon(road_mask, (*P["asphalt"], 255), [screen(p) for p in track.outer])
        clipped_polygon(road_mask, (0, 0, 0, 0), [screen(p) for p in track.inner])
        for _ in range(650):
            point = (rng.randrange(TILE), rng.randrange(TILE))
            if road_mask.get_at(point).a:
                road_mask.set_at(point, rng.choice((P["asphalt_dark"], P["asphalt_light"])))
        tile.blit(road_mask, (0, 0))
        for loop_index, loop in enumerate((track.outer, track.inner)):
            area = sum(a[0] * b[1] - b[0] * a[1]
                       for a, b in zip(loop, np.roll(loop, -1, axis=0)))
            direction = (1 if area > 0 else -1) * (1 if loop_index == 0 else -1)
            for a, b in zip(loop, np.roll(loop, -1, axis=0)):
                edge = b - a
                length = float(np.linalg.norm(edge))
                normal = direction * np.array((edge[1], -edge[0])) / length
                count = max(1, math.ceil(length / 2))
                for index in range(count):
                    start, end = a + edge * index / count, a + edge * (index + 1) / count
                    clipped_polygon(tile, P["red"] if index % 2 == 0 else P["ivory"],
                        [screen(p) for p in (start, end, end + normal * .6, start + normal * .6)])
            pygame.draw.lines(tile, P["ivory"], True, [screen(p) for p in loop],
                              max(1, min(100, round(.10 * scale))))
        _, tangent, a, b = track.gates[0]
        span = b - a
        count = max(2, math.ceil(float(np.linalg.norm(span)) / .7))
        for index in range(count):
            for row in range(2):
                start = a + span * index / count + tangent * (row - 1) * .65
                end = a + span * (index + 1) / count + tangent * (row - 1) * .65
                clipped_polygon(tile, P["ivory"] if (index + row) % 2 else P["asphalt_dark"],
                    [screen(p) for p in (start, end, end + tangent * .65, start + tangent * .65)])
        for x, y, radius in self._decor:
            center = screen((x, y))
            pixels = round(radius * scale)
            if -pixels <= center[0] <= TILE + pixels and -pixels <= center[1] <= TILE + pixels:
                pygame.draw.circle(tile, (40, 52, 34), center, pixels)
                pygame.draw.circle(tile, (63, 83, 47), (center[0] - pixels * .15,
                                                       center[1] - pixels * .15), round(pixels * .82))
        # Sparse spectator rails well beyond the verge, not a continuous wall.
        for a, b in self._fences:
            pygame.draw.line(tile, (143, 151, 132), screen(a), screen(b),
                             max(1, min(30, round(.07 * scale))))
            for fraction in (0., .5, 1.):
                x = a[0] + (b[0] - a[0]) * fraction
                pygame.draw.line(tile, (80, 87, 72), screen((x, a[1] - .3)),
                                 screen((x, a[1] + .3)), max(1, min(30, round(.10 * scale))))
        self._tiles[key] = tile
        while len(self._tiles) > MAX_TILES:
            self._tiles.popitem(last=False)
        return tile

    def draw(self, surface, track, snapshot, camera, debug):
        self._prepare(track, camera.scale)
        left, top = camera.to_world((0, 0))
        right, bottom = camera.to_world(surface.get_size())
        for ty in range(math.floor(top * camera.scale / TILE), math.floor(bottom * camera.scale / TILE) + 1):
            for tx in range(math.floor(left * camera.scale / TILE), math.floor(right * camera.scale / TILE) + 1):
                tile = self._tile(track, camera.scale, tx, ty)
                surface.blit(tile, tuple(round(v) for v in camera.to_screen(
                    (tx * TILE / camera.scale, ty * TILE / camera.scale))))
        self.draw_car(surface, snapshot.cars[0].state, camera)
        if debug:
            self._debug(surface, track, snapshot, camera)

    def draw_car(self, surface, state, camera):
        # Vector artwork is clipped by the destination surface. No scale-sized
        # sprites are allocated, even at extreme user-configured zoom levels.
        factor = self.physics.car_radius_m / .8
        cosine, sine = math.cos(state.heading_rad), math.sin(state.heading_rad)

        def point(x, y):
            return camera.to_screen((state.x_m + factor * (x * cosine - y * sine),
                                     state.y_m + factor * (x * sine + y * cosine)))

        def polygon(color, vertices):
            clipped_polygon(surface, color, [point(x, y) for x, y in vertices])

        def box(color, x1, y1, x2, y2):
            polygon(color, ((x1, y1), (x2, y1), (x2, y2), (x1, y2)))

        polygon((36, 40, 35), ((-.70, -.36), (.70, -.36), (.77, .36), (-.64, .43)))
        for x in (-.48, .38):
            for y in (-.41, .29):
                box((17, 21, 24), x - .13, y, x + .13, y + .12)
        polygon(P["accent"], ((-.70, -.27), (-.58, -.34), (.48, -.34),
                (.72, -.21), (.72, .21), (.48, .34), (-.58, .34), (-.70, .27)))
        box((255, 174, 85), -.5, -.25, .52, -.20)
        polygon((35, 56, 66), ((-.36, -.24), (.15, -.24), (.28, -.17),
                              (.28, .17), (.15, .24), (-.36, .24)))
        box((98, 134, 145), .07, -.20, .15, .20)
        box((218, 106, 36), .36, -.16, .58, .16)
        for y in (-.25, .14):
            box(P["headlight"], .60, y, .69, y + .11)
            box((210, 55, 44), -.69, y, -.61, y + .11)

    def _debug(self, surface, track, snapshot, camera):
        screen = camera.to_screen
        pygame.draw.lines(surface, (115, 172, 204), True, [screen(p) for p in track.points], 1)
        for index, (center, tangent, a, b) in enumerate(track.gates):
            pygame.draw.line(surface, P["accent"], screen(a), screen(b), 2)
            pygame.draw.line(surface, P["accent"], screen(center), screen(center + tangent * 3), 2)
            surface.blit(font(14).render(str(index), True, P["ivory"]), screen(center))
        car = snapshot.cars[0]
        position = (car.state.x_m, car.state.y_m)
        pygame.draw.circle(surface, (255, 110, 100), screen(position),
                           max(1, round(self.physics.car_radius_m * camera.scale)), 1)
        projected, _ = track.point_at(car.projection.s_m)
        pygame.draw.line(surface, (255, 110, 100), screen(position), screen(projected), 2)
        for spawn in track.spawns.values():
            pygame.draw.circle(surface, P["accent"], screen((spawn.x_m, spawn.y_m)), 4, 1)
