"""Multi-car ghost deterministic simulator; snapshots contain only immutable values."""

from collections.abc import Collection, Mapping
from dataclasses import replace
from math import isfinite

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

    def step(self, controls, *, held_ids: Collection[str] = ()):
        world = self.snapshot()
        ids = {car.car_id for car in world.cars}
        if not isinstance(controls, Mapping) or any(key not in ids for key in controls):
            raise ValueError("controls: unknown car ID or invalid mapping")
        for control in controls.values():
            validate_control(control)
        if isinstance(held_ids, (str, bytes)) or not isinstance(held_ids, Collection):
            raise ValueError("held_ids: collection of IDs required")
        if any(not isinstance(key, str) or not key for key in held_ids):
            raise ValueError("held_ids: nonempty string IDs required")
        held_ids = frozenset(held_ids)
        active_ids = {
            car.car_id for car in world.cars if car.status == CarStatus.ACTIVE
        }
        if not held_ids <= active_ids or any(
            key not in controls or controls[key] != CarControl(0.0, 0.0)
            for key in held_ids
        ):
            raise ValueError(
                "held_ids: active cars with explicit zero controls required"
            )
        tick = world.tick + 1
        events = []
        cars = []
        for car in world.cars:
            if car.car_id in held_ids:
                car = replace(car, signed_delta_m=0.0, rewardable_delta_m=0.0)
            elif car.status == CarStatus.ACTIVE:
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

    def recover_car(self, safe: CarSnapshot, *, expected_tick: int) -> WorldSnapshot:
        world = self.snapshot()
        if type(expected_tick) is not int or expected_tick != world.tick:
            raise ValueError("recovery: stale expected tick")
        if not isinstance(safe, CarSnapshot) or safe.status != CarStatus.ACTIVE:
            raise ValueError("recovery: active safe snapshot required")
        current = next((car for car in world.cars if car.car_id == safe.car_id), None)
        if current is None or current.status != CarStatus.DISQUALIFIED:
            raise ValueError("recovery: disqualified matching car required")
        history = (
            "net_travel_m",
            "high_water_m",
            "approach_distance_m",
            "completed_laps",
            "finish_armed",
            "next_checkpoint",
            "lap_start_tick",
            "completed_lap_times_s",
        )
        if any(getattr(current, key) != getattr(safe, key) for key in history):
            raise ValueError("recovery: stale progress history")
        values = (
            safe.state.x_m,
            safe.state.y_m,
            safe.state.heading_rad,
            safe.state.speed_mps,
            safe.state.steering_rad,
            safe.projection.tangent_heading,
        )
        if any(type(v) not in (int, float) or not isfinite(v) for v in values):
            raise ValueError("recovery: finite pose required")
        if not self.track.contains_disk(
            safe.state.x_m, safe.state.y_m, self.config.car_radius_m
        ):
            raise ValueError("recovery: unsafe disk")
        restored = replace(
            safe,
            state=replace(
                safe.state,
                heading_rad=safe.projection.tangent_heading,
                speed_mps=0.0,
                steering_rad=0.0,
            ),
            signed_delta_m=0.0,
            rewardable_delta_m=0.0,
            projection_discontinuity=False,
        )
        self._world = replace(
            world,
            cars=tuple(
                restored if car.car_id == safe.car_id else car for car in world.cars
            ),
        )
        return self._world

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
