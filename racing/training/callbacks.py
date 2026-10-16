"""Post-update evaluation, isolated inference and portable training state."""

import csv
import random
import time
from contextlib import contextmanager

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from racing.core.models import CarControl
from racing.training.checkpoints import publish_alias, save_checkpoint
from racing.training.evaluation import evaluate, evaluate_split
from racing.training.manifest import read_json, write_json
from racing.training.selection import choose_best

RESUMABLE_STOP = 75


class ControlledStop(Exception):
    """A verified checkpoint was committed at a safe update boundary."""


@contextmanager
def isolated_evaluation(policy):
    import torch

    state = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
             torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)
    modes = [(module, module.training) for module in policy.modules()]
    try:
        yield
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])
        if state[3] is not None:
            torch.cuda.set_rng_state_all(state[3])
        for module, mode in modes:
            module.training = mode


class LivePolicy:
    config = {}

    def __init__(self, model, seed):
        self.model, self.training_seed = model, seed

    def reset(self, seed):
        pass

    def act(self, observation, snapshot, track):
        values, _ = self.model.predict(observation, deterministic=True)
        if np.asarray(values).shape != (2,) or not np.isfinite(values).all():
            raise ValueError("nonfinite or malformed live policy prediction")
        values = np.clip(values, -1, 1)
        return CarControl(float(values[0]), float(values[1]))


class EpisodeMetrics(BaseCallback):
    def __init__(self, coordinator):
        super().__init__()
        self.coordinator = coordinator

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode" in info:
                episode = info["episode"]
                self.coordinator.episode_index += 1
                append_csv(self.coordinator.run_dir / "metrics/train.csv", dict(
                    timestep=self.num_timesteps, episode_index=self.coordinator.episode_index,
                    episode_return=episode["r"], episode_length=episode["l"], success=info["success"],
                    progress=info["progress_fraction"], reason=info["episode_reason"],
                    wall_seconds=time.monotonic() - self.coordinator.started,
                    track_id=info["track_id"], reset_seed=info["reset_seed"], simulator_seed=info["simulator_seed"]))
        return True


def append_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow({k: "" if v is None else v for k, v in row.items()})
        file.flush()


class Coordinator:
    def __init__(self, config, env_config, run_dir, track_hash, execution, stop_requested=None):
        self.config, self.env_config, self.run_dir = config, env_config, run_dir
        self.track_hash, self.execution = track_hash, execution
        self.update_count, self.episode_index = 0, 0
        self.next_due, self.ledger, self.last_evaluated = config.eval_every_steps, [], None
        self.started, self.evaluation_seconds = time.monotonic(), 0.
        self.stop_requested = stop_requested or (lambda: False)
        self.last_checkpoint = None

    def state(self):
        return dict(update_count=self.update_count, episode_index=self.episode_index, next_due=self.next_due,
                    ledger=self.ledger, last_evaluated=self.last_evaluated,
                    requested_transitions=self.config.total_timesteps)

    def restore(self, state):
        for name in ("update_count", "episode_index", "next_due", "ledger", "last_evaluated"):
            setattr(self, name, state[name])

    def evaluate(self, model, diagnostic=False):
        before = time.monotonic()
        pair = [int(model.num_timesteps), self.update_count]
        if self.last_evaluated == pair:
            return
        label = f"step_{model.num_timesteps:09d}_update_{self.update_count:06d}"
        with isolated_evaluation(model.policy):
            if self.config.evaluation_split:
                summary = evaluate_split("ppo", agent=LivePolicy(model, self.config.seed),
                               split=self.config.evaluation_split, split_manifest=self.config.split_manifest,
                               episodes_per_track=len(self.config.eval_seeds), reset_seeds=self.config.eval_seeds,
                               seed=self.config.eval_seeds[0], output=self.run_dir / "evaluations" / label,
                               config=self.env_config)
            else:
                summary = evaluate("ppo", agent=LivePolicy(model, self.config.seed),
                               track_id=self.config.track_id, episodes=len(self.config.eval_seeds),
                               reset_seeds=self.config.eval_seeds, seed=self.config.eval_seeds[0],
                               output=self.run_dir / "evaluations" / label, config=self.env_config)
        self.evaluation_seconds += time.monotonic() - before
        all_success = summary["completed"] == summary["attempted"]
        # Deterministic demo attempts have one lap duration; derive the mean
        # from attempts rather than substituting a median in future datasets.
        with (self.run_dir / "evaluations" / label / "attempts.csv").open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        lap_mean = sum(float(row["lap_time_s"]) for row in rows) / len(rows) if all_success else None
        metrics = dict(macro_completion=summary.get("macro_completion", summary["completion_rate"]), macro_progress=summary.get("macro_progress", summary["mean_progress_fraction"]),
                       all_success=all_success, macro_mean_lap_time_s=lap_mean,
                       checkpoint_step=int(model.num_timesteps), update_count=self.update_count)
        self.last_evaluated = pair
        if not diagnostic:
            entry = save_checkpoint(model, self, metrics, selectable=True)
            self.last_checkpoint = entry
            best = choose_best(self.ledger)
            publish_alias(self.run_dir, "best_model", best["path"])
            write_json(self.run_dir / "validation_ledger.json", self.ledger)
        append_csv(self.run_dir / "metrics/evaluations.csv", dict(
            timestep=model.num_timesteps, update_count=self.update_count, attempted=summary["attempted"],
            completed=summary["completed"], completion=summary["completion_rate"],
            lap_median=summary["lap_time_median_s"], lap_iqr=summary["lap_time_iqr_s"],
            mean_progress=summary["mean_progress_fraction"], off_track=summary["off_track_count"],
            timeouts=summary["timeout_count"], diagnostic=diagnostic,
            selected=False if diagnostic else best["path"] == entry["path"], path=f"evaluations/{label}"))
        return metrics

    def after_update(self, model):
        self.update_count += 1
        due = model.num_timesteps >= self.next_due
        final = model.num_timesteps >= self.config.total_timesteps
        if due:
            while self.next_due <= model.num_timesteps:
                self.next_due += self.config.eval_every_steps
        if due or final:
            self.evaluate(model)
        self.flush_progress(model)
        stop = self.stop_requested() or time.monotonic() - self.started >= (
            self.execution.max_session_seconds - self.execution.shutdown_margin_seconds)
        if stop and not final:
            if not (due or final):
                metrics = dict(macro_completion=0., macro_progress=0., all_success=False,
                               macro_mean_lap_time_s=None, checkpoint_step=int(model.num_timesteps), update_count=self.update_count)
                self.last_checkpoint = save_checkpoint(model, self, metrics, selectable=False)
            raise ControlledStop("session limit reached at a safe post-update checkpoint")

    def flush_progress(self, model):
        path = self.run_dir / "manifest.json"
        manifest = read_json(path)
        elapsed = time.monotonic() - self.started
        manifest.update(actual_transitions=int(model.num_timesteps), update_count=self.update_count,
                        evaluation_wall_seconds=self.evaluation_seconds,
                        training_wall_seconds=max(0., elapsed - self.evaluation_seconds))
        manifest["training_steps_per_second"] = (
            (model.num_timesteps - manifest["starting_transitions"]) / manifest["training_wall_seconds"]
            if manifest["training_wall_seconds"] > 0 else None)
        write_json(path, manifest)
