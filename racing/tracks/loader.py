"""Validated closed road ring and its shared finite boundary segments."""

import math

import numpy as np

from racing.core.config import project_path
from racing.core.geometry import (
    boundary_gate,
    inside,
    intersection_candidates,
    intersects,
    segment_distances,
    segment_project,
    segments,
    validate_loop,
    wrap,
)
from racing.core.models import CarState, Projection
from racing.tracks.schema import read_track


class Track:
    @classmethod
    def from_json(cls, path):
        path = project_path(path)
        try:
            return cls(read_track(path))
        except (ValueError, TypeError) as error:
            raise ValueError(f"{path}: {error}") from error

    def __init__(self, data):
        # Geometry-only callers may omit provenance; catalog selection must
        # distinguish those tracks from explicitly classified demo tracks.
        self.split = data.get("split")

        # Retain normalized source geometry for reproducible evaluation hashes.
        def number(value):
            return float(value) if value else 0.0

        self.geometry_payload = {
            "schema_version": data.get("schema_version", "track-v1"),
            "centerline": [[number(v) for v in point] for point in data["centerline"]],
            "width_m": number(data["width_m"]),
            "checkpoints": [
                {"index": int(g["index"]), "s_m": number(g["s_m"])}
                for g in data["checkpoints"]
            ],
            "spawns": [
                {
                    "spawn_id": s["spawn_id"],
                    "s_m": number(s["s_m"]),
                    "lateral_m": number(s["lateral_m"]),
                }
                for s in data["spawns"]
            ],
        }
        self.track_id = data["track_id"]
        self.name = data["name"]
        self.width_m = float(data["width_m"])
        self.points = np.array(data["centerline"], dtype=np.float64)
        self.edges = np.roll(self.points, -1, axis=0) - self.points
        self.lengths = np.linalg.norm(self.edges, axis=1)
        if np.any(self.lengths < 0.1):
            raise ValueError(
                "centerline: segment length below 0.1 m / repeated closing point"
            )
        validate_loop(self.points, "centerline")
        self.tangents = self.edges / self.lengths[:, None]
        self.normals = np.column_stack((-self.tangents[:, 1], self.tangents[:, 0]))
        self.cumulative = np.r_[0.0, np.cumsum(self.lengths)]
        self.length_m = float(self.cumulative[-1])
        previous = np.roll(self.normals, 1, axis=0)
        denominator = 1 + np.sum(previous * self.normals, axis=1)
        if np.any(denominator <= 1e-9):
            raise ValueError("miter: reversing/undefined")
        offsets = self.width_m / 2 * (previous + self.normals) / denominator[:, None]
        if np.any(np.linalg.norm(offsets, axis=1) > self.width_m):
            raise ValueError("miter: displacement exceeds width")
        loops = (self.points + offsets, self.points - offsets)
        for loop in loops:
            validate_loop(loop, "boundary offset fold")
        candidates = intersection_candidates(loops[0], np.roll(loops[0], -1, axis=0),
                                             loops[1], np.roll(loops[1], -1, axis=0))
        for i, j in zip(*np.nonzero(candidates)):
            if intersects(loops[0][i], loops[0][(i+1) % len(self.points)],
                          loops[1][j], loops[1][(j+1) % len(self.points)]):
                raise ValueError("boundary: loops intersect")
        self.inner, self.outer = loops if inside(loops[0][0], loops[1]) else loops[::-1]
        if not all(inside(p, self.outer) for p in self.inner):
            raise ValueError("boundary: loops not nested")

        def area(loop):
            return abs(sum(a[0] * b[1] - a[1] * b[0] for a, b in segments(loop))) / 2

        if area(self.outer) <= area(self.inner):
            raise ValueError("boundary: nonpositive road area")
        self.boundary_segments = segments(self.outer) + segments(self.inner)
        self._boundary_a = np.array([a for a, _ in self.boundary_segments])
        self._boundary_b = np.array([b for _, b in self.boundary_segments])
        self._boundary_a.flags.writeable = False
        self._boundary_b.flags.writeable = False
        checkpoints = data["checkpoints"]
        stations = [g["s_m"] for g in checkpoints]
        if (
            stations[0] != 0
            or [g["index"] for g in checkpoints] != list(range(len(stations)))
            or any(b - a < 20 for a, b in zip(stations, stations[1:] + [self.length_m]))
        ):
            raise ValueError("checkpoint: finish/index/cyclic spacing invariant")
        self.gates = []
        for s in stations:
            center, tangent = self.point_at(s)
            if not self.contains_disk(*center, 0):
                raise ValueError("checkpoint: center outside road")
            a, b = boundary_gate(center, tangent, loops)
            for p, q in self.boundary_segments:
                if (
                    intersects(a, b, p, q)
                    and min(segment_project(a, p, q)[2], segment_project(b, p, q)[2])
                    > 1e-6
                ):
                    raise ValueError("checkpoint: intermediate boundary crossing")
            self.gates.append((center, tangent, a, b))
        self.gates = tuple(self.gates)
        self.spawns = {}
        for spawn in data["spawns"]:
            sid, s, lateral = spawn["spawn_id"], spawn["s_m"], spawn["lateral_m"]
            if sid in self.spawns or not 0 <= s < self.length_m:
                raise ValueError("spawn: duplicate ID or invalid station")
            center, tangent = self.point_at(s)
            position = center + lateral * np.array((-tangent[1], tangent[0]))
            if not self.contains_disk(*position, 0.8):
                raise ValueError("spawn: full disk outside road")
            self.spawns[sid] = CarState(
                float(position[0]),
                float(position[1]),
                wrap(math.atan2(tangent[1], tangent[0])),
                0.0,
                0.0,
            )
        required = {"solo": 0.0, "grid_left": -1.5, "grid_right": 1.5}
        for sid, lateral in required.items():
            matches = [s for s in data["spawns"] if s["spawn_id"] == sid]
            if (
                not matches
                or abs(matches[0]["s_m"] - (self.length_m - 5)) > 1e-6
                or matches[0]["lateral_m"] != lateral
            ):
                raise ValueError(f"spawn.{sid}: required pose 5 m before finish")
        for array in (
            self.points,
            self.edges,
            self.lengths,
            self.tangents,
            self.normals,
            self.cumulative,
            *loops,
        ):
            array.flags.writeable = False

    def point_at(self, s_m):
        if not math.isfinite(s_m):
            raise ValueError("station: finite arclength required")
        s = s_m % self.length_m
        i = int(np.searchsorted(self.cumulative, s, side="right") - 1)
        return self.points[i] + (s - self.cumulative[i]) * self.tangents[
            i
        ], self.tangents[i].copy()

    def project(self, x, y, last_s=None):
        if not all(math.isfinite(v) for v in (x, y)) or (
            last_s is not None and not math.isfinite(last_s)
        ):
            raise ValueError("projection: finite coordinates and hint required")
        point = np.array((x, y))

        def cyclic(s):
            return abs(
                (s - last_s + self.length_m / 2) % self.length_m - self.length_m / 2
            )

        def nearest(indices):
            best = None
            for i in indices:
                foot, t, distance = segment_project(
                    point, self.points[i], self.points[(i + 1) % len(self.points)]
                )
                s = float((self.cumulative[i] + t * self.lengths[i]) % self.length_m)
                key = (cyclic(s) if last_s is not None else 0, i)
                candidate = (
                    distance,
                    key,
                    i,
                    t,
                    s,
                    float((point - foot) @ self.normals[i]),
                )
                if (
                    best is None
                    or distance < best[0] - 1e-9
                    or (abs(distance - best[0]) <= 1e-9 and key < best[1])
                ):
                    best = candidate
            return best

        selected = None
        if last_s is not None:
            indices = [
                i
                for i in range(len(self.points))
                if any(
                    self.cumulative[i] <= last_s + 10 + shift
                    and self.cumulative[i + 1] >= last_s - 10 + shift
                    for shift in (-self.length_m, 0, self.length_m)
                )
            ]
            selected = nearest(indices)
        if selected is None or selected[0] > self.width_m:
            selected = nearest(range(len(self.points)))
        distance, _, i, t, s, lateral = selected
        return Projection(
            i,
            t,
            s,
            lateral,
            math.atan2(self.tangents[i, 1], self.tangents[i, 0]),
            distance,
        )

    def contains_disk(self, x, y, radius):
        if not all(math.isfinite(v) for v in (x, y, radius)) or radius < 0:
            raise ValueError("disk: finite coordinates and nonnegative radius required")
        point = np.array((x, y))
        if not hasattr(self, '_boundary_a'):
            self._boundary_a = np.array([a for a, _ in self.boundary_segments])
            self._boundary_b = np.array([b for _, b in self.boundary_segments])
        return (
            inside(point, self.outer)
            and not inside(point, self.inner)
            and float(np.min(segment_distances(point, self._boundary_a, self._boundary_b)))
            >= radius - 1e-6
        )
