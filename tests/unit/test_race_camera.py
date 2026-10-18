import numpy as np
import pytest


def test_compact_setup_keeps_track_when_debug_enabled():
    from racing.render.race_camera import race_world_rect

    rect = race_world_rect((480, 360), state="MENU", button_rows=3, debug=True)
    assert tuple(rect) == (12, 96, 456, 132)


@pytest.mark.parametrize("viewport", [(480, 360), (1280, 720), (800, 560)])
def test_expanded_track_fits_usable_rectangle(demo_track, viewport):
    from racing.render.race_camera import overhead_camera, race_world_rect

    rect = race_world_rect(viewport, state="MENU", button_rows=3, debug=True)
    margin = 2.8
    camera = overhead_camera(demo_track, viewport, rect, margin_m=margin)
    points = np.concatenate((demo_track.outer, demo_track.inner))
    lower, upper = points.min(axis=0) - margin, points.max(axis=0) + margin
    for x in (lower[0], upper[0]):
        for y in (lower[1], upper[1]):
            sx, sy = camera.to_screen((x, y))
            assert rect.left - 1e-6 <= sx <= rect.right + 1e-6
            assert rect.top - 1e-6 <= sy <= rect.bottom + 1e-6
            assert camera.to_world((sx, sy)) == pytest.approx((x, y))
    assert camera == overhead_camera(demo_track, viewport, rect, margin_m=margin)


@pytest.mark.parametrize(
    "viewport,rows", [((100, 100), 0), ((480, 360), 9), ((480, 360), -1), ((0, 720), 0)]
)
def test_invalid_layout_is_explicit(viewport, rows):
    from racing.render.race_camera import race_world_rect

    with pytest.raises(ValueError):
        race_world_rect(viewport, state="MENU", button_rows=rows, debug=True)


def test_resize_recomputes_camera(demo_track):
    from racing.render.race_camera import overhead_camera, race_world_rect

    cameras = [
        overhead_camera(
            demo_track,
            viewport,
            race_world_rect(viewport, state="RUNNING", button_rows=0, debug=False),
        )
        for viewport in ((480, 360), (1280, 720))
    ]
    assert cameras[0].scale < cameras[1].scale
