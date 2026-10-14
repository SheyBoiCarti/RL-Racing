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
    result = False
    for a, b in segments(loop):
        if segment_project(point, a, b)[2] <= DISTANCE_EPS:
            return True
        if (a[1] > point[1]) != (b[1] > point[1]):
            cut = a[0] + (point[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if point[0] < cut:
                result = not result
    return result


def validate_loop(loop, label):
    edges = segments(loop)
    for i, (a, b) in enumerate(edges):
        c = loop[(i + 2) % len(loop)]
        if abs(cross(b - a, c - b)) <= ORIENTATION_EPS and float((b - a) @ (c - b)) < 0:
            raise ValueError(f"{label}: adjacent overlap")
        for j in range(i + 1, len(edges)):
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
