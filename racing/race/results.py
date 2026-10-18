"""Race outcomes and net-distance ranking over immutable core snapshots."""

from dataclasses import dataclass

from racing.core.models import CarStatus


@dataclass(frozen=True, slots=True)
class CarResult:
    car_id: str
    status: CarStatus
    reason: str
    completed_laps: int
    lap_times_s: tuple[float, ...]
    best_lap_s: float | None
    finish_tick: int | None
    finish_time_s: float | None
    recovery_count: int = 0
    recovery_held_ticks: int = 0


def car_results(snapshot, reasons, finish_ticks, dt_s, *, recovery_by_car=None):
    return {
        car.car_id: CarResult(
            car.car_id,
            car.status,
            reasons[car.car_id],
            car.completed_laps,
            car.completed_lap_times_s,
            min(car.completed_lap_times_s, default=None),
            finish_ticks[car.car_id],
            finish_ticks[car.car_id] * dt_s
            if finish_ticks[car.car_id] is not None
            else None,
            recovery_by_car[car.car_id].count if recovery_by_car else 0,
            recovery_by_car[car.car_id].held_ticks if recovery_by_car else 0,
        )
        for car in snapshot.cars
    }


def positions(snapshot):
    """Competition ranks; progress ties within one micrometre share a rank."""
    groups = {CarStatus.FINISHED: 0, CarStatus.ACTIVE: 1}

    def ahead(first, second):
        a, b = groups.get(first.status, 2), groups.get(second.status, 2)
        return a < b or (
            a == b and a < 2 and first.net_travel_m > second.net_travel_m + 1e-6
        )

    return {
        car.car_id: 1 + sum(ahead(other, car) for other in snapshot.cars)
        for car in snapshot.cars
    }
