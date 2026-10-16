"""Desktop selection stays within the public catalog and resets the selected circuit."""

import pytest

from racing.core.config import load_game, load_physics
from racing.render.app import Application, AppState
from racing.tracks import Track


@pytest.fixture(scope="module")
def frozen_chooser():
    from racing.render.chooser import TrackChooser

    return TrackChooser()


def test_application_exposes_track_chooser():
    app = Application(load_game(), load_physics(), Track.from_json("assets/tracks/demo.json"))
    assert hasattr(app, "chooser")
    app.action("choose_track")
    assert app.choosing_track and app.state == AppState.MENU


def test_catalog_filters_test_and_keeps_load_error():
    from racing.render.chooser import TrackChooser

    calls = []
    demo = Track.from_json("assets/tracks/demo.json")

    def load(path, split):
        calls.append(split)
        if split == "validation":
            raise ValueError("broken-track.json: invalid boundary")
        return {"test_hidden": demo, "train_000": demo}

    chooser = TrackChooser("broken-manifest.json", catalog_loader=load)
    assert calls == ["train", "validation"]
    assert chooser.ids == ["demo"]  # Atomic catalog: never present partial integrity.
    assert "broken-manifest.json" in chooser.error
    assert "invalid boundary" in chooser.error


def test_navigation_wraps_and_selection_resets_manual_track():
    from racing.render.chooser import TrackChooser

    demo = Track.from_json("assets/tracks/demo.json")
    chooser = TrackChooser(catalog_loader=lambda path, split: {})
    chooser.move(-1)
    assert chooser.selected_id == "demo"
    app = Application(load_game(), load_physics(), demo)
    app.chooser = chooser
    app.action("choose_track")
    app.action("select_track")
    assert not app.choosing_track and app.track.track_id == "demo"
    app.action("reset")
    assert app.simulation.snapshot().track_id == chooser.selected_id


def test_chooser_rejects_test_even_from_injected_catalog():
    from racing.render.chooser import TrackChooser

    demo = Track.from_json("assets/tracks/demo.json")
    chooser = TrackChooser(catalog_loader=lambda path, split: {"test_000": demo})
    assert chooser.ids == ["demo"]


def test_chooser_renders_clickable_cards_at_small_sizes(pygame_env):
    import pygame

    from racing.render.chooser import TrackChooser

    chooser = TrackChooser(catalog_loader=lambda path, split: {})
    for size in ((1280, 720), (640, 480), (480, 360)):
        surface = pygame.Surface(size)
        buttons = chooser.draw(surface)
        assert {"select_track", "chooser_back", "chooser_prev", "chooser_next"} <= buttons.keys()
        assert "track:demo" in buttons
        assert all(surface.get_rect().contains(rect) for rect in buttons.values())


def test_debug_readout_includes_shared_projection_and_curvature():
    from racing.render.renderer import debug_readout

    demo = Track.from_json("assets/tracks/demo.json")
    app = Application(load_game(), load_physics(), demo)
    text = debug_readout(demo, app.simulation.snapshot(), 0.0)
    assert "projection" in text and "curvature" in text
    assert "lateral" in text and "segment" in text


def test_frozen_catalog_and_selected_track_drive_and_watch(frozen_chooser):
    from racing.render.watch import WatchApplication

    assert frozen_chooser.error is None
    assert len(frozen_chooser.ids) == 76
    assert not any(key.startswith("test_") for key in frozen_chooser.ids)

    class Controller:
        metadata = {"checkpoint_step": 64}

        def reset(self, seed):
            self.seed = seed

    controller = Controller()
    watch = WatchApplication(load_game(), controller)
    manual = Application(load_game(), load_physics(), frozen_chooser.tracks["demo"])
    try:
        for key in ("train_000", "validation_000", "demo"):
            for app in (manual, watch):
                app.chooser = frozen_chooser
                app.action("menu")
                app.action("choose_track")
                app.action("track:" + key)
                app.action("select_track")
                app.action("reset")
                assert app.track.track_id == key
                assert app.simulation.snapshot().track_id == key
                assert not app.choosing_track
            assert watch.config.default_track_id == key
            assert watch.config.track_sampling == "fixed"
        assert controller.seed == 9000
    finally:
        watch.env.close()


def test_generated_previews_are_distinct_and_clickable(pygame_env, frozen_chooser):
    import pygame

    surface = pygame.Surface((1280, 720))
    frozen_chooser.index = 0
    buttons = frozen_chooser.draw(surface)
    cards = [buttons[f"track:train_{i:03}"] for i in range(3)]
    # Compare preview pixels only, excluding different names and lengths.
    images = [pygame.image.tostring(surface.subsurface(rect.x + 6, rect.y + 6,
                                                      rect.width - 12, rect.height - 56), "RGB")
              for rect in cards]
    assert len(set(images)) == 3
    frozen_chooser.move(-1)
    assert frozen_chooser.selected_id == "validation_014"
    frozen_chooser.move(1)
    assert frozen_chooser.selected_id == "demo"


def test_cli_unavailable_track_returns_to_chooser(tmp_path, monkeypatch):
    from racing.render.app import main

    seen = []
    monkeypatch.setattr(Application, "run", lambda app: seen.append(app))
    main(["--track", "test_000", "--split-manifest", str(tmp_path / "missing.json")])
    app = seen[0]
    assert app.choosing_track and app.track.track_id == "demo"
    assert "test_000.json" in app.chooser.error
    assert "unavailable" in app.chooser.error
