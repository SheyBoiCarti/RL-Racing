"""Canonical labeled replay state, offline record schemas and divergence paths."""

import hashlib
import json
import math
import re
from dataclasses import fields, is_dataclass
from enum import Enum
from functools import lru_cache

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from racing.core.config import project_path

VERSIONS = dict(physics="physics-v1", track="track-v1", observation="obs-rays7-v1",
                action="action-v1", reward="reward-v1", evaluation="evaluation-v1",
                selection="selection-v1")

VERSIONS_V2 = {**VERSIONS, "rules": "race-recovery-v1"}

FLOAT_FIELDS = frozenset(("time_s", "x_m", "y_m", "heading_rad", "speed_mps", "steering_rad",
                          "segment_t", "s_m", "lateral_m", "tangent_heading", "distance_m",
                          "signed_delta_m", "rewardable_delta_m", "net_travel_m", "high_water_m",
                          "approach_distance_m"))


def plain(value, *, hex_floats=False):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {f.name: plain(float(getattr(value, f.name))
                              if f.type in (float, "float") else getattr(value, f.name),
                              hex_floats=hex_floats) for f in fields(value)}
    if isinstance(value, dict):
        return {k: plain(v, hex_floats=hex_floats) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v, hex_floats=hex_floats) for v in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("replay: finite float required")
        return value.hex() if hex_floats else (0.0 if value == 0 else value)
    if value is None or type(value) in (str, int, bool):
        return value
    raise ValueError(f"replay: unsupported value {type(value).__name__}")


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def state_hash(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def verification_state(session):
    state = dict(world=plain(session.snapshot(), hex_floats=True), session=dict(
        no_progress_by_car=dict(session.no_progress_by_car),
        finish_tick_by_car=dict(session.finish_tick_by_car),
        session_state=plain(session.state), winner=session.winner,
        abort_reason=session.abort_reason))
    if session.rules is not None:
        state["session"].update(race_rules=plain(session.rules),
                                recovery_by_car=plain(session.recovery_by_car, hex_floats=True))
    return state


@lru_cache(maxsize=2)
def record_validator(schema_version="replay-v1"):
    if schema_version not in ("replay-v1", "replay-v2"):
        raise ValueError("replay: unsupported schema version")
    root = project_path("docs/specs/schemas")
    schemas = [json.loads((root / name).read_text(encoding="utf-8"))
               for name in ("common.schema.json", "track.schema.json", "replay.schema.json",
                            "common-v2.schema.json", "replay-v2.schema.json")]
    registry = Registry().with_resources((s["$id"], Resource.from_contents(s)) for s in schemas)
    return Draft202012Validator(schemas[2 if schema_version == "replay-v1" else 4], registry=registry)


def validate_record(record, *, schema_version="replay-v1"):
    error = next(record_validator(schema_version).iter_errors(record), None)
    if error:
        # oneOf's top-level message prints the whole record; show a concise path instead.
        details = list(error.context)
        preferred = [e for e in details if e.validator != "enum"]
        detail = next(iter(preferred or details), error)
        raise ValueError(f"replay seq {record.get('seq')}: schema {list(detail.path)}: {detail.message}")


def parse_record(line):
    def mapping(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"replay: duplicate JSON field {key}")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError(f"replay: finite numbers required ({value})")

    try:
        result = json.loads(line, object_pairs_hook=mapping, parse_constant=nonfinite)
        plain(result)  # Also catches numeric overflow such as 1e999.
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"replay: invalid JSON: {error}") from error
    if not isinstance(result, dict):
        raise ValueError("replay: record must be an object")
    return result


def compare_state(expected, actual, *, tick, tolerant=False, path="state"):
    def fail():
        raise ValueError(f"replay divergence at tick {tick}, {path}: expected {expected!r}, actual {actual!r}")

    if type(expected) is not type(actual):
        fail()
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            fail()
        for key in expected:
            compare_state(expected[key], actual[key], tick=tick, tolerant=tolerant, path=f"{path}.{key}")
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            fail()
        for i, (first, second) in enumerate(zip(expected, actual, strict=True)):
            compare_state(first, second, tick=tick, tolerant=tolerant, path=f"{path}[{i}]")
    elif tolerant and isinstance(expected, str) and (
        path.rsplit(".", 1)[-1] in FLOAT_FIELDS or re.search(r"\.completed_lap_times_s\[\d+\]$", path)
    ):
        try:
            first, second = float.fromhex(expected), float.fromhex(actual)
        except (ValueError, OverflowError):
            fail()
        if not (math.isfinite(first) and math.isfinite(second)) or abs(first - second) > 1e-6:
            fail()
    elif expected != actual:
        fail()


def verify_state(expected, digest, session, *, tolerant=False):
    if state_hash(expected) != digest:
        raise ValueError(f"replay tick {session.snapshot().tick}: stored state hash mismatch")
    actual = verification_state(session)
    compare_state(expected, actual, tick=session.snapshot().tick, tolerant=tolerant)
