"""Headless human/policy race and deterministic explicit-action reducer."""

from collections.abc import Mapping
from dataclasses import replace
from enum import Enum
from math import atan2, ceil, hypot

from racing.core.config import PhysicsConfig
from racing.core.geometry import wrap
from racing.core.models import CarControl, CarSpawn, CarStatus, StepResult
from racing.core.physics import validate_control
from racing.core.simulation import Simulation
from racing.race.config import RaceConfig, RaceRules
from racing.race.difficulty import govern_ai
from racing.race.recovery import RecoveryState
from racing.race.results import car_results, positions


class RaceState(str, Enum):
    MENU = "MENU"
    COUNTDOWN = "COUNTDOWN"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    RESULTS = "RESULTS"


class RaceSession:
    def __init__(
        self,
        track,
        physics=None,
        config=None,
        controllers=None,
        seed=0,
        lane_swap=False,
        *,
        rules: RaceRules | None = None,
    ):
        if rules is not None and not isinstance(rules, RaceRules):
            raise ValueError("rules: RaceRules or None required")
        self.rules = rules
        self.track = track
        self.physics = physics or PhysicsConfig()
        self.config = config or RaceConfig()
        if not isinstance(self.physics, PhysicsConfig) or not isinstance(
            self.config, RaceConfig
        ):
            raise ValueError("race: PhysicsConfig and RaceConfig required")
        if type(lane_swap) is not bool:
            raise ValueError("lane_swap: boolean required")
        self.seed, self.lane_swap = seed, lane_swap
        self.controllers = dict(controllers or {})
        if self.controllers.keys() - {"ai", "human"}:
            raise ValueError("controllers: unknown car ID")
        self._validate_grid()
        left, right = (
            ("grid_right", "grid_left") if lane_swap else ("grid_left", "grid_right")
        )
        self.spawns = (CarSpawn("ai", right), CarSpawn("human", left))
        self.simulation = Simulation(self.physics)
        self.observation_builder = None
        self._listeners = []
        self._reset()
        self.state = RaceState.MENU

    def _validate_grid(self):
        states = []
        for sid, lateral in (("grid_left", -1.5), ("grid_right", 1.5)):
            if sid not in self.track.spawns:
                raise ValueError(f"grid: missing {sid}")
            state = self.track.spawns[sid]
            center, tangent = self.track.point_at(self.track.length_m - 5)
            expected = (
                center[0] - lateral * tangent[1],
                center[1] + lateral * tangent[0],
            )
            if (
                hypot(state.x_m - expected[0], state.y_m - expected[1]) > 1e-6
                or abs(wrap(state.heading_rad - atan2(tangent[1], tangent[0]))) > 1e-6
                or state.speed_mps != 0
                or state.steering_rad != 0
                or not self.track.contains_disk(
                    state.x_m, state.y_m, self.physics.car_radius_m
                )
            ):
                raise ValueError(f"grid: illegal {sid} pose")
            states.append(state)
        if (
            hypot(states[0].x_m - states[1].x_m, states[0].y_m - states[1].y_m)
            < 2 * self.physics.car_radius_m
        ):
            raise ValueError("grid: overlapping car disks")

    def _reset(self):
        self.simulation.reset(self.track, self.seed, self.spawns)
        self.recovery_by_car = (
            {car.car_id: RecoveryState(car) for car in self.snapshot().cars}
            if self.rules is not None
            else {}
        )
        self.error_message = ""
        self.no_progress_by_car = {s.car_id: 0 for s in self.spawns}
        self.finish_tick_by_car = {s.car_id: None for s in self.spawns}
        self.reason_by_car = {s.car_id: "running" for s in self.spawns}
        self.last_controls = {}
        self.last_observations = {}
        self.winner = "no_winner"
        self.abort_reason = None
        self.countdown_ticks_remaining = ceil(self.config.countdown_seconds * 20)

    def snapshot(self):
        return self.simulation.snapshot()

    @property
    def results(self):
        return car_results(
            self.snapshot(),
            self.reason_by_car,
            self.finish_tick_by_car,
            self.physics.dt_s,
            recovery_by_car=self.recovery_by_car,
        )

    @property
    def position_by_car(self):
        return positions(self.snapshot())

    @property
    def gap_m(self):
        cars = {c.car_id: c for c in self.snapshot().cars}
        return cars["ai"].net_travel_m - cars["human"].net_travel_m

    def add_listener(self, callback):
        """Observe callback(session, event, payload) after each reducer transition."""
        self._listeners.append(callback)

    def remove_listener(self, callback):
        self._listeners.remove(callback)

    def _notify(self, event, **payload):
        for callback in tuple(self._listeners):
            callback(self, event, payload)

    def start(self):
        if self.state != RaceState.MENU:
            raise ValueError("start: MENU required")
        for controller in self.controllers.values():
            controller.reset(self.seed)
        self.state = RaceState.COUNTDOWN
        if self.countdown_ticks_remaining == 0:
            self._begin_running()
        return self.snapshot()

    def _begin_running(self):
        self.state = RaceState.RUNNING
        self._notify("start")

    def countdown_tick(self):
        if self.state == RaceState.COUNTDOWN:
            self.countdown_ticks_remaining -= 1
            if self.countdown_ticks_remaining == 0:
                self._begin_running()
        return self.snapshot()

    def _validate_controls(self, controls, active_ids):
        if not isinstance(controls, Mapping) or set(controls) != active_ids:
            raise ValueError("controls: exactly all active car IDs required")
        clipped = {}
        for car_id in sorted(controls):
            control = controls[car_id]
            validate_control(control)
            clipped[car_id] = CarControl(
                float(min(1, max(-1, control.steering))),
                float(min(1, max(-1, control.longitudinal))),
            )
        return clipped

    def step(self, controls=None):
        if self.state != RaceState.RUNNING:
            if controls is not None:
                raise ValueError("step: RUNNING required")
            return StepResult(self.snapshot(), ())
        before = self.snapshot()
        active = [c for c in before.cars if c.status == CarStatus.ACTIVE]
        active_ids = {c.car_id for c in active}
        held = {
            key
            for key, recovery in self.recovery_by_car.items()
            if recovery.remaining_ticks > 0 and key in active_ids
        }
        movable = [car for car in active if car.car_id not in held]
        if controls is None:
            try:
                if self.observation_builder is None:
                    from racing.envs.config import RLConfig
                    from racing.envs.observation import ObservationBuilder

                    self.observation_builder = ObservationBuilder(
                        RLConfig(physics=self.physics)
                    )
                observations = {
                    c.car_id: self.observation_builder.build(
                        before, self.track, c.car_id
                    )
                    for c in movable
                }
                controls = {
                    c.car_id: self.controllers[c.car_id].act(
                        observations[c.car_id], before, self.track
                    )
                    for c in movable
                }
                for key in held:
                    controls[key] = CarControl(0.0, 0.0)
                if self.rules is not None and "ai" in active_ids - held:
                    ai = next(c for c in active if c.car_id == "ai")
                    controls["ai"] = govern_ai(
                        controls["ai"],
                        ai.state.speed_mps,
                        self.physics.max_speed_mps,
                        self.rules.difficulty,
                    )
                clipped = self._validate_controls(controls, active_ids)
            except Exception:
                self.abort("inference_error")
                return StepResult(self.snapshot(), ())
            self.last_observations = observations
        else:
            clipped = self._validate_controls(controls, active_ids)
        if any(clipped[key] != CarControl(0.0, 0.0) for key in held):
            raise ValueError("recovery: held cars require zero controls")
        result = self.simulation.step(clipped, held_ids=held)
        self.last_controls = clipped
        previous = {c.car_id: c for c in before.cars}
        for car in result.snapshot.cars:
            car_id = car.car_id
            if previous[car_id].status != CarStatus.ACTIVE:
                continue
            recovering = False
            if self.rules is not None:
                recovery = self.recovery_by_car[car_id]
                if car_id in held:
                    remaining = recovery.remaining_ticks - 1
                    self.recovery_by_car[car_id] = replace(
                        recovery,
                        remaining_ticks=remaining,
                        held_ticks=recovery.held_ticks + 1,
                        safe=car if remaining == 0 else recovery.safe,
                        last_release_tick=result.snapshot.tick
                        if remaining == 0
                        else recovery.last_release_tick,
                    )
                    recovering = True
                elif car.status == CarStatus.DISQUALIFIED:
                    try:
                        restored = self.simulation.recover_car(
                            recovery.safe, expected_tick=result.snapshot.tick
                        )
                    except ValueError as exc:
                        self.error_message = f"Recovery failed for {car_id}: {exc}"
                        self.reason_by_car[car_id] = "recovery_error"
                        self._abort("recovery_error", notify=False)
                        break
                    car = next(c for c in restored.cars if c.car_id == car_id)
                    self.recovery_by_car[car_id] = replace(
                        recovery,
                        remaining_ticks=self.rules.recovery_hold_ticks,
                        count=recovery.count + 1,
                        last_release_tick=None,
                    )
                    recovering = True
                elif car.status == CarStatus.ACTIVE:
                    self.recovery_by_car[car_id] = replace(recovery, safe=car)
            increment = car.high_water_m - previous[car_id].high_water_m
            self.no_progress_by_car[car_id] = (
                0
                if recovering or increment > self.config.progress_threshold_m
                else self.no_progress_by_car[car_id] + 1
            )
            if car.status == CarStatus.DISQUALIFIED:
                self.reason_by_car[car_id] = "off_track"
            elif car.completed_laps >= self.config.target_laps:
                self.finish_tick_by_car[car_id] = result.snapshot.tick
                self.reason_by_car[car_id] = "finished"
                self.simulation.retire_car(car_id, CarStatus.FINISHED)
            elif result.snapshot.tick >= self.config.race_max_steps:
                self.reason_by_car[car_id] = "race_max_steps"
                self.simulation.retire_car(car_id, CarStatus.TIMED_OUT)
            elif self.no_progress_by_car[car_id] >= self.config.no_progress_steps:
                self.reason_by_car[car_id] = "no_progress"
                self.simulation.retire_car(car_id, CarStatus.TIMED_OUT)
        finishes = {
            key: tick
            for key, tick in self.finish_tick_by_car.items()
            if tick is not None
        }
        if self.abort_reason == "recovery_error":
            pass
        elif finishes:
            earliest = min(finishes.values())
            winners = [key for key, tick in finishes.items() if tick == earliest]
            self.winner = winners[0] if len(winners) == 1 else "draw"
            self.state = RaceState.RESULTS
        elif all(c.status != CarStatus.ACTIVE for c in self.snapshot().cars):
            self.state = RaceState.RESULTS
        result = StepResult(self.snapshot(), result.events)
        self._notify("tick", controls=dict(clipped), result=result)
        return result

    def pause(self):
        if self.state != RaceState.RUNNING:
            raise ValueError("pause: RUNNING required")
        self.state = RaceState.PAUSED
        self._notify("pause")

    def resume(self):
        if self.state != RaceState.PAUSED:
            raise ValueError("resume: PAUSED required")
        self.state = RaceState.RUNNING
        self._notify("resume")

    def abort(self, reason):
        return self._abort(reason)

    def _abort(self, reason, *, notify=True):
        if self.state not in (RaceState.RUNNING, RaceState.PAUSED):
            raise ValueError("abort: RUNNING or PAUSED required")
        allowed = ("inference_error", "user_exit", "restart")
        if self.rules is not None:
            allowed += ("recovery_error",)
        if reason not in allowed:
            raise ValueError("abort: expected inference_error, user_exit, or restart")
        for car in self.snapshot().cars:
            if car.status == CarStatus.ACTIVE:
                self.simulation.retire_car(car.car_id, CarStatus.ABORTED)
                self.reason_by_car[car.car_id] = reason
        self.abort_reason, self.winner, self.state = (
            reason,
            "no_winner",
            RaceState.RESULTS,
        )
        if reason == "recovery_error":
            for car in self.snapshot().cars:
                if car.status == CarStatus.DISQUALIFIED:
                    self.reason_by_car[car.car_id] = reason
        if notify:
            self._notify("abort", reason=reason)
        return self.snapshot()

    def restart(self):
        if self.state in (RaceState.RUNNING, RaceState.PAUSED):
            self.abort("restart")
        self._reset()
        self.state = RaceState.MENU
        return self.start()
