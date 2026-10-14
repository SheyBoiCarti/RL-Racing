import copy
import json
import math

import numpy as np
import pytest

from racing.core.geometry import boundary_gate, segment_project
from racing.tracks.loader import Track


@pytest.fixture(scope="module")
def track():
    return Track.from_json("assets/tracks/demo.json")


def test_projection_primitive():
    foot, t, distance = segment_project((4, 3), (0, 0), (10, 0))
    assert tuple(foot) == (4, 0)
    assert t == 0.4 and distance == 3


def test_ring_and_vertices(track):
    assert not track.contains_disk(0, 0, 0.8)
    ring = Track.__new__(Track)
    ring.outer = np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]])
    ring.inner = np.array([[6.0, 6.0], [14.0, 6.0], [14.0, 14.0], [6.0, 14.0]])
    from racing.core.geometry import segments

    ring.boundary_segments = segments(ring.outer) + segments(ring.inner)
    assert ring.contains_disk(10, 0.8, 0.8)
    assert not ring.contains_disk(10, 0.799, 0.8)
    center, tangent = track.point_at(track.cumulative[3])
    assert np.array_equal(center, track.points[3])
    assert np.array_equal(tangent, track.tangents[3])
    p = track.project(*center)
    assert p.distance_m == pytest.approx(0)


def test_corner_gate():
    # Independently specified finite incoming/outgoing boundaries, width 12.
    k = 6 * math.tan(math.pi / 8)
    inner = np.array(
        [[-20, 6], [-k, 6], [10 - 6 / math.sqrt(2), 10 + 6 / math.sqrt(2)], [-20, 25.0]]
    )
    outer = np.array(
        [
            [-20, -6],
            [k, -6],
            [10 + 6 / math.sqrt(2), 10 - 6 / math.sqrt(2)],
            [25, -20.0],
        ]
    )
    a, b = boundary_gate(
        np.array([0.0, 0.0]), np.array([1.0, 1.0]) / math.sqrt(2), (inner, outer)
    )
    assert a == pytest.approx([6 / math.sqrt(2), -6 / math.sqrt(2)])
    assert b == pytest.approx([-6, 6])
    assumed = np.array([-6 / math.sqrt(2), 6 / math.sqrt(2)])
    assert (
        min(
            segment_project(assumed, p, q)[2]
            for loop in (inner, outer)
            for p, q in zip(loop, np.roll(loop, -1, axis=0))
        )
        > 0.1
    )


@pytest.mark.parametrize(
    "change,reason",
    [
        (
            lambda d: d["centerline"].__setitem__(1, d["centerline"][0]),
            "segment length",
        ),
        (lambda d: d["centerline"].__setitem__(1, d["centerline"][75]), "centerline"),
        (lambda d: d.__setitem__("width_m", 400), "boundary|miter"),
        (lambda d: d["spawns"][0].__setitem__("lateral_m", 20), "spawn"),
        (lambda d: d["checkpoints"][1].__setitem__("s_m", 1), "checkpoint"),
    ],
)
def test_rejected_invariants(change, reason):
    from racing.core.config import project_path

    data = copy.deepcopy(
        json.loads(project_path("assets/tracks/demo.json").read_text())
    )
    change(data)
    with pytest.raises(ValueError, match=reason):
        Track(data)
