"""Structural audit only: never performs policy inference."""
import math

import numpy as np

from racing.tracks.loader import Track


def validate_generated(data, config):
    points = np.array(data['centerline'], dtype=float)
    if not np.isfinite(points).all():
        raise ValueError('centerline: nonfinite')
    if not np.array_equal(points, np.round(points, 6)):
        raise ValueError('quantization: six decimal centerline required')
    edges = np.roll(points, -1, axis=0)-points
    lengths = np.linalg.norm(edges, axis=1)
    length = float(lengths.sum())
    if not config.length_min_m <= length <= config.length_max_m:
        raise ValueError('length: outside configured bounds')
    if data['width_m'] != config.width_m:
        raise ValueError('width: mismatch')
    headings = np.arctan2(edges[:, 1], edges[:, 0])
    turns = np.abs((headings-np.roll(headings, 1)+np.pi) % (2*np.pi)-np.pi)
    curvature = turns/((lengths+np.roll(lengths, 1))/2)
    if max(turns) > config.max_heading_change_rad:
        raise ValueError('heading: excessive turn')
    if max(curvature) > config.max_abs_curvature_per_m:
        raise ValueError('curvature: excessive curvature')
    # Vectorized O(N²) endpoint projections; crossing pairs are independently
    # rejected by Track's centerline topology validation below.
    a, b = points, np.roll(points, -1, axis=0)
    def distances(p):
        delta = p[:, None, :]-a[None, :, :]
        t = np.clip(np.sum(delta*edges[None, :, :], axis=2)/(lengths**2)[None, :], 0, 1)
        return np.linalg.norm(delta-t[:, :, None]*edges[None, :, :], axis=2)
    da, db = distances(a), distances(b)
    distance = np.minimum.reduce([da, db, da.T, db.T])
    mid = np.cumsum(lengths)-lengths/2
    separation = np.abs(mid[:, None]-mid[None, :])
    separation = np.minimum(separation, length-separation)
    eligible = separation > 2*config.width_m
    clearance = float(distance[eligible].min())-config.width_m
    if clearance < config.min_road_clearance_m:
        raise ValueError('clearance: nonlocal road clearance too small')
    track = Track(data)
    for center, tangent, a, b in track.gates:
        # Interior samples avoid boundary classification ambiguity at endpoints.
        count = math.ceil(float(np.linalg.norm(b-a))/.25)
        for fraction in np.arange(1, count)/count:
            p = a+fraction*(b-a)
            if not track.contains_disk(*p, 0):
                raise ValueError('gate: interior outside road')
    left, right = track.spawns['grid_left'], track.spawns['grid_right']
    if math.hypot(left.x_m-right.x_m, left.y_m-right.y_m) <= 1.6:
        raise ValueError('spawn: grid separation')
    return track, dict(length_m=length, max_curvature=float(max(curvature)),
                       max_heading_change_rad=float(max(turns)), min_clearance_m=clearance,
                       validator_version='geometry-v1')
