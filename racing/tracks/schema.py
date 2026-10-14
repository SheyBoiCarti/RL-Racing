"""Offline track-v1 validation; reject nonfinite JSON and duplicate keys."""

import json
import math
from pathlib import Path

from jsonschema import Draft202012Validator


def read_track(path):
    def mapping(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{key}: duplicate field")
            result[key] = value
        return result

    def finite(value, field="root"):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{field}: nonfinite number")
        if isinstance(value, dict):
            for key, child in value.items():
                finite(child, f"{field}.{key}")
        if isinstance(value, list):
            for i, child in enumerate(value):
                finite(child, f"{field}[{i}]")

    data = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=mapping)
    finite(data)
    schema = json.loads(Path(__file__).with_name("track.schema.json").read_text())
    errors = sorted(
        Draft202012Validator(schema).iter_errors(data), key=lambda e: str(e.path)
    )
    if errors:
        error = errors[0]
        raise ValueError(f"{'.'.join(map(str, error.path))}: {error.message}")
    return data
