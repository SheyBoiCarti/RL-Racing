"""Float64 geometry; orientation tolerance 1e-9, distance tolerance 1e-6 m."""

import math

import numpy as np

ORIENTATION_EPS = 1e-9
DISTANCE_EPS = 1e-6


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def segment_project(point, a, b):
    point, a, b = (np.asarray(v, dtype=np.float64) for v in (point, a, b))
    edge = b - a
    denominator = float(edge @ edge)
    t = float(np.clip((point - a) @ edge / denominator, 0, 1)) if denominator else 0.0
    foot = a + t * edge
    return foot, t, float(np.linalg.norm(point - foot))


def intersects(a, b, c, d):
    edge, other = b - a, d - c
    denominator = cross(edge, other)
    if abs(denominator) <= ORIENTATION_EPS:
        if abs(cross(c - a, edge)) > ORIENTATION_EPS:
            return False
        size = float(edge @ edge)
        if size == 0:
            return segment_project(a, c, d)[2] <= DISTANCE_EPS
        lo, hi = sorted((float((c - a) @ edge / size), float((d - a) @ edge / size)))
        return max(lo, 0) <= min(hi, 1) + ORIENTATION_EPS
    t, u = cross(c - a, other) / denominator, cross(c - a, edge) / denominator
    return (
        -ORIENTATION_EPS <= t <= 1 + ORIENTATION_EPS
        and -ORIENTATION_EPS <= u <= 1 + ORIENTATION_EPS
    )


def segments(loop):
    return tuple(zip(loop, np.roll(loop, -1, axis=0)))


def inside(point, loop):
    """Even-odd half-open ray rule; boundary points are classified inside."""
    a = np.asarray(loop, dtype=np.float64)
    b = np.roll(a, -1, axis=0)
    if np.any(segment_distances(point, a, b) <= DISTANCE_EPS):
        return True
    crossing = (a[:, 1] > point[1]) != (b[:, 1] > point[1])
    aa, bb = a[crossing], b[crossing]
    cuts = aa[:, 0] + (point[1]-aa[:, 1])*(bb[:, 0]-aa[:, 0])/(bb[:, 1]-aa[:, 1])
    return bool(np.count_nonzero(point[0] < cuts) % 2)


def segment_distances(point, a, b):
    """Distances to a batch of finite segments, including point segments."""
    edge = b-a
    size = np.sum(edge*edge, axis=-1)
    delta = np.asarray(point)-a
    t = np.zeros_like(size)
    np.divide(np.sum(delta*edge, axis=-1), size, out=t, where=size != 0)
    return np.linalg.norm(delta-np.clip(t, 0, 1)[..., None]*edge, axis=-1)


def intersection_candidates(a, b, c, d):
    """Broad phase that preserves exact scalar intersection semantics."""
    # Scalar intersects admits a dimensionless endpoint tolerance. Expand
    # boxes by that tolerance times segment extent plus distance epsilon.
    pad = DISTANCE_EPS + ORIENTATION_EPS*(np.linalg.norm(b-a, axis=-1)[:, None]
                                       +np.linalg.norm(d-c, axis=-1)[None, :])
    return np.all(np.maximum(np.minimum(a, b)[:, None, :], np.minimum(c, d)[None, :, :])
                  <= np.minimum(np.maximum(a, b)[:, None, :], np.maximum(c, d)[None, :, :])+pad[:, :, None], axis=-1)


def validate_loop(loop, label):
    edges = segments(loop)
    candidates = intersection_candidates(np.asarray(loop), np.roll(loop, -1, axis=0),
                                         np.asarray(loop), np.roll(loop, -1, axis=0))
    for i, (a, b) in enumerate(edges):
        c = loop[(i + 2) % len(loop)]
        if abs(cross(b - a, c - b)) <= ORIENTATION_EPS and float((b - a) @ (c - b)) < 0:
            raise ValueError(f"{label}: adjacent overlap")
        for j in np.flatnonzero(candidates[i]):
            if j <= i:
                continue
            if j == i + 1 or (i == 0 and j == len(edges) - 1):
                continue
            if intersects(a, b, *edges[j]):
                raise ValueError(f"{label}: nonadjacent intersection")


def boundary_gate(center, tangent, loops):
    normal = np.array((-tangent[1], tangent[0]))
    hits = []
    for loop_id, loop in enumerate(loops):
        for a, b in segments(loop):
            edge = b - a
            denominator = cross(normal, edge)
            if abs(denominator) <= ORIENTATION_EPS:
                if abs(cross(a - center, normal)) <= ORIENTATION_EPS:
                    raise ValueError("checkpoint: collinear boundary")
                continue
            t, u = (
                cross(a - center, edge) / denominator,
                cross(a - center, normal) / denominator,
            )
            if -ORIENTATION_EPS <= u <= 1 + ORIENTATION_EPS:
                if not any(
                    abs(t - h[0]) <= DISTANCE_EPS and loop_id == h[1] for h in hits
                ):
                    hits.append((t, loop_id))
    positive = sorted((h for h in hits if h[0] > DISTANCE_EPS))
    negative = sorted((h for h in hits if h[0] < -DISTANCE_EPS), reverse=True)
    if not positive or not negative or positive[0][1] == negative[0][1]:
        raise ValueError("checkpoint: endpoints must hit opposite boundary loops")
    return center + negative[0][0] * normal, center + positive[0][0] * normal
