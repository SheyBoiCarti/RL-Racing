"""Catch fictitious prestart timers, rollover and restarted success timers."""

from dataclasses import replace

from racing.core.config import PhysicsConfig
from racing.render.interface import format_time, race_readout


def test_prestart_and_completed_lap_readouts(world):
    physics = PhysicsConfig()
    prestart = race_readout(world, physics, None)
    assert prestart.time == "—" and prestart.best == "—"
    assert prestart.phase == "Reach the start line"
    car = replace(world.cars[0], lap_start_tick=0)
    running = race_readout(replace(world, tick=247, cars=(car,)), physics, None)
    assert running.time == "00:12.35"
    car = replace(car, completed_laps=1, completed_lap_times_s=(62.35,), lap_start_tick=247)
    completed = race_readout(replace(world, tick=247, cars=(car,)), physics, 61.5)
    assert completed.time == "01:02.35"
    assert completed.lap == "01 / 01"
    assert completed.best == "01:01.50"
    car = replace(car, state=replace(car.state, speed_mps=20.))
    assert race_readout(replace(world, cars=(car,)), physics, None).speed == "072"
    assert format_time(59.999) == "01:00.00"
