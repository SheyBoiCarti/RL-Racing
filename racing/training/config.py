"""Strict fixed-track PPO and operational session configuration."""

from dataclasses import asdict, dataclass, field, fields
from math import isfinite
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TrainConfig:
    algo: str = "ppo"
    policy: str = "MlpPolicy"
    device: str = "cpu"
    seed: int = 42
    total_timesteps: int = 1_000_000
    n_envs: int = 1
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    learning_rate: float = .0003
    gamma: float = .99
    gae_lambda: float = .95
    clip_range: float = .2
    ent_coef: float = 0.
    vf_coef: float = .5
    max_grad_norm: float = .5
    policy_kwargs: dict = field(default_factory=lambda: {"net_arch": {"pi": [64, 64], "vf": [64, 64]}})
    eval_every_steps: int = 10000
    eval_seeds: list = field(default_factory=lambda: [9000, 9001, 9002, 9003, 9004])
    target_laps: int = 1
    track_id: str = "demo"
    track_sampling: str = "fixed"
    split_manifest: str | None = None
    evaluation_split: str | None = None
    condition: str | None = None
    execution_profile_registration: str | None = None

    def __post_init__(self):
        if self.track_sampling not in ("fixed", "uniform_train"):
            raise ValueError("track_sampling: expected fixed or uniform_train")
        if self.evaluation_split not in (None, "validation"):
            raise ValueError("evaluation_split: only validation supported; test requires Phase 6")
        if (self.track_sampling == "uniform_train" or self.evaluation_split) and not self.split_manifest:
            raise ValueError("split_manifest required")
        if self.split_manifest and self.evaluation_split != "validation":
            raise ValueError("split_manifest requires validation evaluation")
        if self.condition not in (None, "single", "multi"):
            raise ValueError("condition: expected single or multi")
        if self.condition and (self.evaluation_split != "validation" or
                self.track_sampling != ("uniform_train" if self.condition == "multi" else "fixed")):
            raise ValueError("condition sampling/validation mismatch")
        if self.algo != "ppo" or self.policy != "MlpPolicy":
            raise ValueError("Phase 3 requires ppo / MlpPolicy")
        if self.device not in ("cpu", "cuda", "auto"):
            raise ValueError("device: expected cpu, cuda or auto")
        if self.n_envs != 1 or type(self.n_envs) is not int:
            raise ValueError("Phase 3 requires one worker")
        if self.target_laps != 1 or self.track_id != "demo":
            raise ValueError("Phase 3 requires demo and one target lap")
        for name in ("total_timesteps", "n_steps", "batch_size", "n_epochs", "eval_every_steps", "target_laps"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name}: positive integer required")
        if self.batch_size < 2 or self.n_steps < 2 or self.n_steps % self.batch_size:
            raise ValueError("batch_size must divide rollout size and be at least two")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed: uint32 required")
        if not isinstance(self.eval_seeds, list) or not self.eval_seeds or any(
            type(s) is not int or not 0 <= s < 2**32 for s in self.eval_seeds
        ) or len(set(self.eval_seeds)) != len(self.eval_seeds) or self.seed in self.eval_seeds:
            raise ValueError("eval_seeds: distinct uint32 seeds outside the training stream required")
        for name in ("learning_rate", "gamma", "gae_lambda", "clip_range", "ent_coef", "vf_coef", "max_grad_norm"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not isfinite(value) or value < 0:
                raise ValueError(f"{name}: finite nonnegative number required")
            if name in ("learning_rate", "clip_range", "max_grad_norm") and value == 0:
                raise ValueError(f"{name}: positive value required")
            if name in ("gamma", "gae_lambda", "clip_range") and value > 1:
                raise ValueError(f"{name}: must be <= 1")
            object.__setattr__(self, name, float(value))
        architecture = self.policy_kwargs
        if not isinstance(architecture, dict) or set(architecture) != {"net_arch"}:
            raise ValueError("policy_kwargs: only net_arch supported; activation is Tanh")
        layers = architecture["net_arch"]
        if not isinstance(layers, dict) or set(layers) != {"pi", "vf"} or any(
            not isinstance(v, list) or not v or any(type(n) is not int or n <= 0 for n in v)
            for v in layers.values()
        ):
            raise ValueError("net_arch: nonempty pi/vf positive integer layers required")


@dataclass(frozen=True)
class ExecutionConfig:
    platform: str = "local"
    device: str = "cpu"
    torch_threads: int = 1
    max_session_seconds: float = 43200
    shutdown_margin_seconds: float = 900
    output_root: str = "runs"

    def __post_init__(self):
        if self.platform not in ("local", "kaggle") or self.device not in ("cpu", "cuda"):
            raise ValueError("execution platform/device unsupported")
        if type(self.torch_threads) is not int or self.torch_threads != 1:
            raise ValueError("torch_threads must be 1")
        for name in ("max_session_seconds", "shutdown_margin_seconds"):
            v = getattr(self, name)
            if type(v) not in (int, float) or not isfinite(v) or v <= 0:
                raise ValueError(f"{name}: finite positive value required")
        if self.max_session_seconds <= self.shutdown_margin_seconds:
            raise ValueError("session budget must exceed shutdown margin")
        if not isinstance(self.output_root, str) or not self.output_root or "/kaggle/input" in self.output_root.replace("\\", "/"):
            raise ValueError("output_root must be writable, outside /kaggle/input")


def _load(path, cls, overrides):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict) or set(data) - {f.name for f in fields(cls)}:
        raise ValueError(f"{path}: unknown configuration fields")
    data.update({k: v for k, v in overrides.items() if v is not None})
    try:
        return cls(**data)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: {error}") from error


def load_train(path="configs/train_ppo.yaml", **overrides):
    return _load(path, TrainConfig, overrides)


def load_execution(path):
    return _load(path, ExecutionConfig, {})


def resume_identity(config):
    return {k: v for k, v in asdict(config).items() if k not in ("total_timesteps", "execution_profile_registration")}
