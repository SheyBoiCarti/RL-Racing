"""Projection continuity, high-water credit and finite swept gate transitions."""

from dataclasses import replace
from math import cos, hypot, pi

import numpy as np

from racing.core.geometry import wrap
from racing.core.models import EventKind, RaceEvent


def signed_delta(old_s, new_s, length):
    return (new_s - old_s + length / 2) % length - length / 2


def continuous(track, old, new, displacement, physics):
    delta = signed_delta(old.s_m, new.s_m, track.length_m)
    if displacement > physics.max_speed_mps * physics.dt_s + 1e-6:
        return False
    theta = 0.0
    for i, station in enumerate(track.cumulative[:-1]):
        distance = (
            (station - old.s_m) % track.length_m
            if delta >= 0
            else (old.s_m - station) % track.length_m
        )
        # Outgoing tangent is already selected at the old vertex; include a new
        # vertex for forward travel, and the old vertex for reverse travel.
        traversed = (0 < distance <= delta) if delta >= 0 else (0 <= distance < -delta)
        if traversed:
            previous = track.tangents[i - 1]
            current = track.tangents[i]
            theta += abs(
                wrap(
                    float(
                        np.arctan2(current[1], current[0])
                        - np.arctan2(previous[1], previous[0])
                    )
                )
            )
    return (
        theta < pi / 2
        and abs(delta)
        <= (displacement + old.distance_m + new.distance_m) / cos(theta / 2) + 1e-6
    )


def gate_crossing(previous, current, gate):
    center, tangent, a, b = gate
    previous, current = np.asarray(previous), np.asarray(current)
    movement = current - previous
    old, new = float((previous - center) @ tangent), float((current - center) @ tangent)
    if not (old < 0 <= new and float(movement @ tangent) > 0):
        return None
    fraction = old / (old - new)
    position = previous + fraction * movement
    edge = b - a
    length = float(np.linalg.norm(edge))
    along = float((position - a) @ edge / length)
    return fraction if -1e-6 <= along <= length + 1e-6 else None


def advance(old, state, projection, track, tick, physics):
    displacement = hypot(state.x_m - old.state.x_m, state.y_m - old.state.y_m)
    valid = continuous(track, old.projection, projection, displacement, physics)
    delta = (
        signed_delta(old.projection.s_m, projection.s_m, track.length_m)
        if valid
        else 0.0
    )
    net = old.net_travel_m + delta
    rewardable = max(0.0, net - old.high_water_m)
    car = replace(
        old,
        state=state,
        projection=projection,
        projection_discontinuity=not valid,
        signed_delta_m=delta,
        rewardable_delta_m=rewardable,
        net_travel_m=net,
        high_water_m=max(old.high_water_m, net),
    )
    events = []
    if not valid:
        return car, events
    crossings = []
    for index, gate in enumerate(track.gates):
        fraction = gate_crossing(
            (old.state.x_m, old.state.y_m), (state.x_m, state.y_m), gate
        )
        if fraction is not None:
            crossings.append((fraction, index))
    for fraction, index in sorted(crossings):
        kind = None
        if index == 0 and not car.finish_armed:
            car = replace(
                car, finish_armed=True, next_checkpoint=1, lap_start_tick=tick
            )
            kind = EventKind.LAP_STARTED
        elif car.finish_armed and index == car.next_checkpoint:
            if index == 0:
                time = (tick - car.lap_start_tick) * physics.dt_s
                car = replace(
                    car,
                    completed_laps=car.completed_laps + 1,
                    completed_lap_times_s=car.completed_lap_times_s + (time,),
                    lap_start_tick=tick,
                    next_checkpoint=1,
                )
                kind = EventKind.LAP_COMPLETED
            else:
                car = replace(car, next_checkpoint=(index + 1) % len(track.gates))
                kind = EventKind.CHECKPOINT_CROSSED
        if kind:
            lap = (
                car.completed_laps
                if kind == EventKind.LAP_COMPLETED
                else car.completed_laps + 1
            )
            events.append(
                RaceEvent(
                    kind, car.car_id, tick, tick * physics.dt_s, index, lap, fraction
                )
            )
    return car, events
