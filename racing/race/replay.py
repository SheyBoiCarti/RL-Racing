"""Self-contained action replay: record applied ticks and reconstruct the reducer."""

import copy
import platform
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from racing.core.config import PhysicsConfig
from racing.core.models import CarControl
from racing.race.config import RaceConfig
from racing.race.replay_state import (
    VERSIONS,
    canonical_json,
    compare_state,
    parse_record,
    plain,
    state_hash,
    validate_record,
    verification_state,
    verify_state,
)
from racing.race.session import RaceSession
from racing.tracks import Track


def _dependencies():
    packages = {}
    for package in ("numpy", "pygame", "PyYAML", "jsonschema", "gymnasium", "stable_baselines3", "torch"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not-installed"
    return dict(python=platform.python_version(), platform=platform.platform(),
                packages=packages, git_commit=None)


class ReplayRecorder:
    """Attach before countdown; every file is exclusive and flushed per record.

    Session listeners run after outcomes, so action records never describe a tick
    that failed inference. A terminal tick or abort writes the footer immediately.
    A recorder belongs to exactly one race; restart needs a new recorder/path.
    """

    def __init__(self, path, session, *, model_sha256=None, model_label=None):
        if session.snapshot().tick or session.state not in ("MENU", "COUNTDOWN", "RUNNING"):
            raise ValueError("replay recorder must attach before race start/tick zero")
        self.path, self.session = Path(path), session
        self.model_sha256, self.model_label = model_sha256, model_label
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("x", encoding="utf-8", newline="\n")
        self.seq = 0
        self.started = False
        self.closed = False
        session.add_listener(self._on_event)
        if session.state == "RUNNING":
            self._header()

    def _write(self, record):
        record = dict(record, seq=self.seq)
        validate_record(record)
        self.file.write(canonical_json(record) + "\n")
        self.file.flush()
        self.seq += 1

    def _header(self):
        if self.started or self.closed:
            return
        session = self.session
        physics, race = plain(session.physics), plain(session.config)
        track = getattr(session.track, "source_payload", None)
        if track is None:
            track = dict(session.track.geometry_payload, track_id=session.track.track_id,
                         name=session.track.name, split=session.track.split or "demo", generator=None)
        initial = verification_state(session)
        try:
            self._write(dict(record_type="header", schema_version="replay-v1", fixture_only=False,
                             physics_config=physics, race_config=race, versions=VERSIONS,
                             config_hashes=dict(physics=state_hash(physics), race=state_hash(race)),
                             track=track, track_geometry_sha256=state_hash(session.track.geometry_payload),
                             simulator_seed=session.seed, spawns=plain(session.spawns),
                             model_sha256=self.model_sha256, model_label=self.model_label,
                             dependencies=_dependencies(), hash_interval=100,
                             initial_state=initial, initial_state_sha256=state_hash(initial)))
        except Exception:
            self.file.close()
            self.closed = True
            raise
        self.started = True

    def _on_event(self, session, event, payload):
        if self.closed:
            return
        if event == "start":
            self._header()
        elif event == "tick":
            self.record_tick()
            if session.state == "RESULTS":
                self.close()
        elif event in ("pause", "resume", "abort"):
            self.record_event(event, payload.get("reason"))
            if event == "abort":
                self.close()

    def record_tick(self):
        if not self.started:
            raise ValueError("replay: header must precede ticks")
        tick = self.session.snapshot().tick
        record = dict(record_type="tick", tick=tick, controls=plain(self.session.last_controls))
        if tick % 100 == 0:
            state = verification_state(self.session)
            record.update(verification_state=state, state_sha256=state_hash(state))
        self._write(record)

    def record_event(self, event, reason=None):
        if not self.started:
            raise ValueError("replay: countdown excluded; header required before session events")
        state = verification_state(self.session)
        self._write(dict(record_type="session_event", after_tick=self.session.snapshot().tick,
                         event=event, reason=reason, verification_state=state, state_sha256=state_hash(state)))

    def close(self):
        if self.closed:
            return
        if not self.started:
            # Countdown has no running replay. No header/ticks means no driven race.
            self.file.close()
            self.path.unlink()
            self.closed = True
            return
        if self.session.state != "RESULTS":
            raise ValueError("replay: finalize via session.abort before closing active race")
        state = verification_state(self.session)
        self._write(dict(record_type="footer", status="aborted" if self.session.abort_reason else "completed",
                         tick_count=self.session.snapshot().tick, winner=self.session.winner,
                         abort_reason=self.session.abort_reason, final_state=state,
                         final_state_sha256=state_hash(state)))
        self.file.close()
        self.closed = True


@dataclass(frozen=True)
class ReplayFrame:
    snapshot: object
    state: str
    controls: dict
    results: dict
    position_by_car: dict
    gap_m: float


@dataclass(frozen=True)
class ReplayPlayback:
    header: dict
    session: RaceSession
    frames: tuple[ReplayFrame, ...]


def _frame(session):
    return ReplayFrame(session.snapshot(), plain(session.state), dict(session.last_controls),
                       dict(session.results), dict(session.position_by_car), session.gap_m)


def load_replay(path, *, tolerant=False):
    """Validate/reconstruct all records before playback, without policy inference."""
    records = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                raise ValueError(f"replay line {line_number}: empty record")
            record = parse_record(line)
            validate_record(record)
            if record["seq"] != len(records):
                raise ValueError(f"replay line {line_number}: noncontiguous seq")
            records.append(record)
    if not records or records[0]["record_type"] != "header":
        raise ValueError("replay: header required first")
    if len(records) < 2 or records[-1]["record_type"] != "footer":
        raise ValueError("replay: missing footer (incomplete recording)")
    header = records[0]
    if header["fixture_only"]:
        raise ValueError("replay: specification fixture is not a runnable recording")
    if header["versions"] != VERSIONS or header["hash_interval"] != 100:
        raise ValueError("replay: unsupported versions/hash interval")
    physics = PhysicsConfig(**header["physics_config"])
    config = RaceConfig(**header["race_config"])
    for name, value in (("physics", physics), ("race", config)):
        if state_hash(plain(value)) != header["config_hashes"][name]:
            raise ValueError(f"replay: {name} config hash mismatch")
    track = Track(header["track"])
    if state_hash(track.geometry_payload) != header["track_geometry_sha256"]:
        raise ValueError("replay: track geometry hash mismatch")
    expected_spawns = [{"car_id": "ai", "spawn_id": "grid_right"},
                       {"car_id": "human", "spawn_id": "grid_left"}]
    swapped = [{"car_id": "ai", "spawn_id": "grid_left"},
               {"car_id": "human", "spawn_id": "grid_right"}]
    if header["spawns"] not in (expected_spawns, swapped):
        raise ValueError("replay: ordered unique human/ai grid spawns required")
    session = RaceSession(track, physics=physics, config=config, seed=header["simulator_seed"],
                          lane_swap=header["spawns"] == swapped)
    session.start()
    while session.state == "COUNTDOWN":
        session.countdown_tick()
    verify_state(header["initial_state"], header["initial_state_sha256"], session, tolerant=tolerant)
    frames = [_frame(session)]
    for index, record in enumerate(records[1:], 1):
        kind = record["record_type"]
        tick = session.snapshot().tick
        if kind == "tick":
            if session.state != "RUNNING" or record["tick"] != tick + 1:
                raise ValueError(f"replay seq {index}: illegal state or noncontiguous tick {record['tick']}")
            controls = {key: CarControl(**value) for key, value in record["controls"].items()}
            session.step(controls)
            if record["tick"] % 100 == 0 and "verification_state" not in record:
                raise ValueError(f"replay tick {record['tick']}: missing scheduled state/hash")
            if "verification_state" in record:
                verify_state(record["verification_state"], record["state_sha256"], session, tolerant=tolerant)
            frames.append(_frame(session))
        elif kind == "session_event":
            if record["after_tick"] != tick:
                raise ValueError(f"replay seq {index}: event after_tick mismatch")
            event, reason = record["event"], record["reason"]
            if event == "abort":
                if reason is None or session.state not in ("RUNNING", "PAUSED"):
                    raise ValueError(f"replay seq {index}: illegal abort")
                session.abort(reason)
                if "verification_state" not in record:
                    raise ValueError(f"replay seq {index}: abort requires state/hash")
            else:
                if reason is not None:
                    raise ValueError(f"replay seq {index}: pause/resume reason must be null")
                if event == "pause" and session.state == "RUNNING":
                    session.pause()
                elif event == "resume" and session.state == "PAUSED":
                    session.resume()
                else:
                    raise ValueError(f"replay seq {index}: illegal {event}")
            if "verification_state" in record:
                verify_state(record["verification_state"], record["state_sha256"], session, tolerant=tolerant)
            # Session events change the current tick's UI state without creating physics time.
            frames[-1] = _frame(session)
        elif kind == "footer":
            if index != len(records) - 1 or session.state != "RESULTS":
                raise ValueError(f"replay seq {index}: premature/duplicate footer")
            expected = dict(tick_count=tick, winner=session.winner, abort_reason=session.abort_reason,
                            status="aborted" if session.abort_reason else "completed")
            compare_state(expected, {key: record[key] for key in expected}, tick=tick, path="footer")
            verify_state(record["final_state"], record["final_state_sha256"], session, tolerant=tolerant)
        else:
            raise ValueError(f"replay seq {index}: unexpected {kind}")
    return ReplayPlayback(copy.deepcopy(header), session, tuple(frames))
