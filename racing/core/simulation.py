"""Multi-car ghost deterministic simulator; snapshots contain only immutable values."""

from collections.abc import Mapping
from dataclasses import replace

from racing.core.models import (
    CarControl,
    CarSnapshot,
    CarStatus,
    EventKind,
    RaceEvent,
    StepResult,
    WorldSnapshot,
)
from racing.core.physics import integrate_car, validate_control
from racing.core.progress import advance


class Simulation:
    def __init__(self, config):
        self.config = config
        self._world = None

    def reset(self, track, seed, cars):
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed: uint32 required")
        if not cars:
            raise ValueError("cars: nonempty roster required")
        ids = [spawn.car_id for spawn in cars]
        if any(not isinstance(car_id, str) or not car_id for car_id in ids):
            raise ValueError("car_id: nonempty string required")
        if len(set(ids)) != len(ids):
            raise ValueError("car_id: duplicate ID")
        snapshots = []
        for spawn in sorted(cars, key=lambda item: item.car_id):
            if spawn.spawn_id not in track.spawns:
                raise ValueError("spawn_id: unknown spawn")
            state = track.spawns[spawn.spawn_id]
            if not track.contains_disk(state.x_m, state.y_m, self.config.car_radius_m):
                raise ValueError("spawn: configured disk outside road")
            projection = track.project(state.x_m, state.y_m)
            snapshots.append(
                CarSnapshot(
                    spawn.car_id,
                    state,
                    CarStatus.ACTIVE,
                    projection,
                    False,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    (track.length_m - projection.s_m) % track.length_m,
                    0,
                    False,
                    1,
                    None,
                    (),
                )
            )
        self.track, self.seed = track, seed
        self._world = WorldSnapshot(0, 0.0, track.track_id, tuple(snapshots))
        return self._world

    def snapshot(self):
        if self._world is None:
            raise RuntimeError("reset required")
        return self._world

    def step(self, controls):
        world = self.snapshot()
        ids = {car.car_id for car in world.cars}
        if not isinstance(controls, Mapping) or any(key not in ids for key in controls):
            raise ValueError("controls: unknown car ID or invalid mapping")
        for control in controls.values():
            validate_control(control)
        tick = world.tick + 1
        events = []
        cars = []
        for car in world.cars:
            if car.status == CarStatus.ACTIVE:
                state = integrate_car(
                    car.state,
                    controls.get(car.car_id, CarControl(0.0, 0.0)),
                    self.config,
                )
                projection = self.track.project(
                    state.x_m, state.y_m, car.projection.s_m
                )
                if not self.track.contains_disk(
                    state.x_m, state.y_m, self.config.car_radius_m
                ):
                    car = replace(
                        car,
                        state=state,
                        projection=projection,
                        status=CarStatus.DISQUALIFIED,
                        projection_discontinuity=False,
                        signed_delta_m=0.0,
                        rewardable_delta_m=0.0,
                    )
                    events.append(
                        RaceEvent(
                            EventKind.OFF_TRACK,
                            car.car_id,
                            tick,
                            tick * self.config.dt_s,
                            None,
                            None,
                            None,
                        )
                    )
                else:
                    car, car_events = advance(
                        car, state, projection, self.track, tick, self.config
                    )
                    events.extend(car_events)
            cars.append(car)
        self._world = WorldSnapshot(
            tick, tick * self.config.dt_s, self.track.track_id, tuple(cars)
        )
        return StepResult(
            self._world,
            tuple(
                sorted(
                    events,
                    key=lambda e: (
                        e.tick,
                        e.car_id,
                        -1 if e.crossing_fraction is None else e.crossing_fraction,
                        e.kind.value,
                    ),
                )
            ),
        )

    def retire_car(self, car_id, status):
        world = self.snapshot()
        car = next((car for car in world.cars if car.car_id == car_id), None)
        if car is None or status not in (
            CarStatus.FINISHED,
            CarStatus.TIMED_OUT,
            CarStatus.ABORTED,
        ):
            raise ValueError("retirement: invalid ID/status")
        if car.status == status:
            return world
        if car.status != CarStatus.ACTIVE:
            raise ValueError("retirement: terminal status cannot change")
        self._world = replace(
            world,
            cars=tuple(
                replace(item, status=status, signed_delta_m=0.0, rewardable_delta_m=0.0)
                if item.car_id == car_id
                else item
                for item in world.cars
            ),
        )
        return self._world
