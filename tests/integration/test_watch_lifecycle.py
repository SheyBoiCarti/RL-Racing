from racing.core.config import load_game
from racing.render.watch import WatchApplication
from racing.tracks import Track


def test_watch_draws_a_frame_before_quitting(pygame_env, monkeypatch):
    import pygame

    class Controller:
        metadata = {"checkpoint_step": 1}

    app = WatchApplication(
        load_game(), Controller(), track=Track.from_json("assets/tracks/demo.json")
    )
    batches = iter([[], [pygame.event.Event(pygame.QUIT)]])
    monkeypatch.setattr(pygame.event, "get", lambda: next(batches))
    app.run()
    assert not app.running
