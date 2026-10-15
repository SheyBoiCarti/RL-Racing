"""Pure boundary-only center-origin ray sensors."""

import numpy as np


def ray_segment_distance(origin, direction, a, b, max_distance=40.0):
    ox, oy = origin
    dx, dy = direction
    ex, ey = b[0] - a[0], b[1] - a[1]
    if ex * ex + ey * ey <= 1e-24:
        return None
    qx, qy = a[0] - ox, a[1] - oy
    denominator = dx * ey - dy * ex
    if abs(denominator) > 1e-9:
        distance = (qx * ey - qy * ex) / denominator
        u = (qx * dy - qy * dx) / denominator
        if distance < -1e-6 or not -1e-6 <= u <= 1 + 1e-6:
            return None
        distance = max(0.0, distance)
    else:
        if abs(qx * dy - qy * dx) > 1e-9:
            return None
        lo, hi = sorted((qx * dx + qy * dy, (b[0] - ox) * dx + (b[1] - oy) * dy))
        if hi < 0:
            return None
        distance = max(0.0, lo)
    return float(distance) if distance <= max_distance else None


def cast_rays(track, pose, angles, max_distance):
    headings = pose.heading_rad + np.radians(angles)
    directions = np.column_stack((np.cos(headings), np.sin(headings)))
    segments = np.asarray(track.boundary_segments)
    edges = segments[:, 1] - segments[:, 0]
    q = segments[:, 0] - np.array((pose.x_m, pose.y_m))
    denominator = (
        directions[:, 0, None] * edges[None, :, 1]
        - directions[:, 1, None] * edges[None, :, 0]
    )
    cross_q_d = (
        q[None, :, 0] * directions[:, 1, None] - q[None, :, 1] * directions[:, 0, None]
    )
    nonparallel = np.abs(denominator) > 1e-9
    safe = np.where(nonparallel, denominator, 1.0)
    distance = (q[:, 0] * edges[:, 1] - q[:, 1] * edges[:, 0])[None, :] / safe
    u = cross_q_d / safe
    valid = nonparallel & (distance >= -1e-6) & (u >= -1e-6) & (u <= 1 + 1e-6)
    distances = np.where(valid, np.maximum(distance, 0.0), np.inf)
    # Collinear segments use projected endpoint intervals, including origin overlap.
    a = directions @ q.T
    b = a + directions @ edges.T
    collinear = ~nonparallel & (np.abs(cross_q_d) <= 1e-9) & (np.maximum(a, b) >= 0)
    collinear &= np.sum(edges * edges, axis=1)[None, :] > 1e-24
    distances = np.minimum(
        distances, np.where(collinear, np.maximum(0.0, np.minimum(a, b)), np.inf)
    )
    distances[:, np.sum(edges * edges, axis=1) <= 1e-24] = np.inf
    return np.minimum(max_distance, distances.min(axis=1))
