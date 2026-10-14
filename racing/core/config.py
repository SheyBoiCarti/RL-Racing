"""Frozen configuration with filename/field-specific validation."""

import sys
from dataclasses import dataclass, fields
from math import isfinite
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if not (PROJECT_ROOT / "pyproject.toml").is_file():
    PROJECT_ROOT = Path(sys.prefix) / "share" / "ai-racing"


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True, slots=True)
class PhysicsConfig:
    dt_s: float = 0.05
    max_speed_mps: float = 22.0
    max_steer_rad: float = 0.55
    wheelbase_m: float = 2.6
    acceleration_mps2: float = 6.0
    braking_mps2: float = 10.0
    rolling_drag_mps2: float = 0.25
    steering_slew_radps: float = 2.5
    car_radius_m: float = 0.8

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if (
                type(value) not in (float, int)
                or not isfinite(value)
                or value < 0
                or (value == 0 and field.name != "rolling_drag_mps2")
            ):
                raise ValueError(f"{field.name}: expected finite positive number")
        if self.max_steer_rad >= 1.5707963267948966:
            raise ValueError("max_steer_rad: must be below pi/2")


@dataclass(frozen=True, slots=True)
class GameConfig:
    width_px: int
    height_px: int
    render_fps: int
    pixels_per_metre: float
    follow_camera: bool
    debug: bool
    catch_up_limit: int
    bindings: tuple[tuple[str, tuple[str, ...]], ...]


class _UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if key in result:
            raise ValueError(f"{key}: duplicate field")
        result[key] = loader.construct_object(value_node)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _read(path, cls):
    path = project_path(path)
    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueLoader)
        if not isinstance(data, dict):
            raise TypeError("root: expected mapping")
        unknown = data.keys() - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"{min(unknown)}: unknown field")
        return path, data
    except (ValueError, TypeError, yaml.YAMLError) as error:
        raise ValueError(f"{path}: {error}") from error


def load_physics(path="configs/physics.yaml") -> PhysicsConfig:
    filename, data = _read(path, PhysicsConfig)
    try:
        config = PhysicsConfig(**data)
        return PhysicsConfig(
            **{f.name: float(getattr(config, f.name)) for f in fields(config)}
        )
    except (ValueError, TypeError) as error:
        raise ValueError(f"{filename}: {error}") from error


def load_game(path="configs/game.yaml") -> GameConfig:
    filename, data = _read(path, GameConfig)
    try:
        for field in fields(GameConfig):
            if field.name not in data:
                raise ValueError(f"{field.name}: missing field")
        for name in ("width_px", "height_px", "render_fps", "catch_up_limit"):
            if type(data[name]) is not int or data[name] <= 0:
                raise ValueError(f"{name}: expected positive integer")
        if data["catch_up_limit"] > 5:
            raise ValueError("catch_up_limit: maximum is 5")
        for name in ("follow_camera", "debug"):
            if type(data[name]) is not bool:
                raise ValueError(f"{name}: expected boolean")
        scale = data["pixels_per_metre"]
        if type(scale) not in (float, int) or not isfinite(scale) or scale <= 0:
            raise ValueError("pixels_per_metre: expected finite positive number")
        bindings = data["bindings"]
        actions = {
            "left",
            "right",
            "accelerate",
            "brake",
            "brake_override",
            "pause",
            "reset",
            "debug",
        }
        if not isinstance(bindings, dict) or bindings.keys() != actions:
            raise ValueError("bindings: expected all eight action names")
        used = set()
        for action, keys in bindings.items():
            if not isinstance(keys, list) or not keys:
                raise ValueError(f"bindings.{action}: expected nonempty key list")
            for key in keys:
                aliases = {"esc": "escape", "enter": "return"}
                canonical = (
                    aliases.get(key.lower(), key.lower())
                    if isinstance(key, str)
                    else None
                )
                if not isinstance(key, str) or not key or canonical in used:
                    raise ValueError(
                        f"bindings.{action}: invalid or conflicting key {key!r}"
                    )
                used.add(canonical)
        data["bindings"] = tuple((a, tuple(k)) for a, k in bindings.items())
        data["pixels_per_metre"] = float(scale)
        return GameConfig(**data)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{filename}: {error}") from error
