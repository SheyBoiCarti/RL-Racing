"""Portable provenance and canonical artifact helpers (never credentials)."""

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from racing.training.evaluation import _plain, _write_json, canonical_hash

ROOT = Path(__file__).resolve().parents[2]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, data)


def now():
    return datetime.now(timezone.utc).isoformat()


def dependencies():
    packages = {}
    for name in ("numpy", "gymnasium", "stable_baselines3", "torch", "PyYAML", "tensorboard"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not-installed"
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return dict(python=platform.python_version(), platform=platform.platform(), packages=packages, git_commit=revision)


def source_identity():
    paths = []
    for name in ("racing", "scripts", "configs", "experiments", "assets/tracks", "docs/specs/schemas"):
        paths.extend(p for p in (ROOT / name).rglob("*") if p.suffix in (".py", ".yaml", ".json")
                     and not any(part.startswith(".") for part in p.relative_to(ROOT).parts))
    paths.extend(ROOT.glob("requirements*lock.txt"))
    paths.append(ROOT / "pyproject.toml")
    return canonical_hash({p.relative_to(ROOT).as_posix(): file_hash(p) for p in sorted(paths)})


def contracts(config):
    physics = _plain(config.physics)
    observation = dict(version=config.observation_version, ray_angles_deg=list(config.ray_angles_deg),
                       ray_range_m=config.ray_range_m, dtype="float32", dimension=11)
    action = dict(version=config.action_version, dimension=2, dtype="float32", order=["steering", "longitudinal"])
    env = {k: getattr(config, k) for k in ("target_laps", "max_steps", "no_progress_steps", "progress_threshold_m",
           "reward_version", "steering_penalty", "progress_coefficient", "step_cost", "off_track_penalty", "lap_bonus")}
    return dict(physics=physics, observation=observation, action=action, env=env)


def versions(config):
    return dict(physics="physics-v1", track="track-v1", observation=config.observation_version,
                action=config.action_version, reward=config.reward_version, evaluation="evaluation-v1", selection="selection-v1")
