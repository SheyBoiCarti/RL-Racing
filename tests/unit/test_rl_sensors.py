import pytest

from racing.envs.sensors import ray_segment_distance


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ((5, -1), (5, 1), 5),
        ((-5, -1), (-5, 1), None),
        ((41, -1), (41, 1), None),
        ((0, 1), (5, 1), None),
        ((-1, 0), (1, 0), 0),
        ((2, 0), (4, 0), 2),
        ((-4, 0), (-2, 0), None),
        ((2, 0), (2, 0), None),
        ((0, -1), (0, 1), 0),
        ((5, 0), (5, 2), 5),
    ],
)
def test_intersections(a, b, expected):
    assert ray_segment_distance((0, 0), (1, 0), a, b, 40) == expected
    # Exercise the vectorized production path against the same literal fixtures.
    from types import SimpleNamespace

    from racing.envs.sensors import cast_rays

    pose = SimpleNamespace(x_m=0.0, y_m=0.0, heading_rad=0.0)
    track = SimpleNamespace(boundary_segments=[(a, b)])
    assert cast_rays(track, pose, [0.0], 40)[0] == (
        40 if expected is None else expected
    )


def test_vertical_ray():
    assert ray_segment_distance((0, 0), (0, 1), (-1, 3), (1, 3), 40) == 3


def test_cast_rays_matches_scalar_intersections():
    from math import cos, radians, sin

    from racing.envs.config import RAY_ANGLES
    from racing.envs.sensors import cast_rays
    from racing.tracks.loader import Track

    track = Track.from_json("assets/tracks/demo.json")
    pose = track.spawns["solo"]
    expected = []
    for angle in RAY_ANGLES:
        heading = pose.heading_rad + radians(angle)
        hits = [
            ray_segment_distance(
                (pose.x_m, pose.y_m), (cos(heading), sin(heading)), a, b, 40
            )
            for a, b in track.boundary_segments
        ]
        expected.append(min([40] + [hit for hit in hits if hit is not None]))
    assert cast_rays(track, pose, RAY_ANGLES, 40) == pytest.approx(expected)
