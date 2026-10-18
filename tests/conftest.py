"""Opt-in display setup; importing core tests never initializes SDL."""

import pytest

from racing.core.config import PhysicsConfig
from racing.core.models import CarSpawn
from racing.core.simulation import Simulation
from racing.tracks import Track


@pytest.fixture
def pygame_env(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    import pygame

    pygame.init()
    yield pygame
    pygame.quit()


@pytest.fixture
def demo_track():
    return Track.from_json("assets/tracks/demo.json")


@pytest.fixture
def world(demo_track):
    simulation = Simulation(PhysicsConfig())
    return simulation.reset(demo_track, 0, (CarSpawn("human", "solo"),))


@pytest.fixture
def recovery_race(demo_track):
    from racing.race.config import RaceConfig, RaceRules
    from racing.race.session import RaceSession

    race = RaceSession(
        demo_track, rules=RaceRules(), config=RaceConfig(countdown_seconds=0)
    )
    race.start()
    return race
