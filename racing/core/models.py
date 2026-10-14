"""Immutable public simulator values from Shared Contracts section 1."""

from dataclasses import dataclass
from enum import Enum


class CarStatus(str, Enum):
    ACTIVE = "active"
    FINISHED = "finished"
    DISQUALIFIED = "disqualified"
    TIMED_OUT = "timed_out"
    ABORTED = "aborted"


class EventKind(str, Enum):
    OFF_TRACK = "off_track"
    LAP_STARTED = "lap_started"
    CHECKPOINT_CROSSED = "checkpoint_crossed"
    LAP_COMPLETED = "lap_completed"


@dataclass(frozen=True, slots=True)
class CarControl:
    steering: float
    longitudinal: float


@dataclass(frozen=True, slots=True)
class CarState:
    x_m: float
    y_m: float
    heading_rad: float
    speed_mps: float
    steering_rad: float


@dataclass(frozen=True, slots=True)
class CarSpawn:
    car_id: str
    spawn_id: str


@dataclass(frozen=True, slots=True)
class Projection:
    segment_index: int
    segment_t: float
    s_m: float
    lateral_m: float
    tangent_heading: float
    distance_m: float


@dataclass(frozen=True, slots=True)
class CarSnapshot:
    car_id: str
    state: CarState
    status: CarStatus
    projection: Projection  # selected by core; do not reproject in an adapter
    projection_discontinuity: bool
    signed_delta_m: float  # accepted current tick; zero at reset/retirement/failure
    rewardable_delta_m: float
    net_travel_m: float  # accepted signed travel from spawn
    high_water_m: float  # max(0, all accepted historical net travel)
    approach_distance_m: float  # forward distance from spawn to first finish
    completed_laps: int
    finish_armed: bool
    next_checkpoint: int  # 1 initially; 0 means all nonfinish gates passed
    lap_start_tick: int | None
    completed_lap_times_s: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class WorldSnapshot:
    tick: int
    time_s: float
    track_id: str
    cars: tuple[CarSnapshot, ...]  # sorted car_id, including retired cars


@dataclass(frozen=True, slots=True)
class RaceEvent:
    kind: EventKind
    car_id: str
    tick: int
    time_s: float
    checkpoint_index: int | None
    lap_number: int | None
    crossing_fraction: float | None


@dataclass(frozen=True, slots=True)
class StepResult:
    snapshot: WorldSnapshot
    events: tuple[RaceEvent, ...]
